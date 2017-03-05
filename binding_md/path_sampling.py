import openpathsampling as paths
from openpathsampling.engines.openmm.tools import trajectory_to_mdtraj
from contact_map import ContactFrequency
from openpathsampling.netcdfplus import StorableNamedObject


class StableContactsState(StorableNamedObject):
    def __init__(self, topology, ligand_groups, n_frames, n_contacts,
                 frequency=0.95):
        """
        Parameters
        ----------
        topology : :class:`mdtraj.Topology`
        ligand_groups : dict of {str: list of int}
        """
        self.topology = topology
        self.ligand_groups = ligand_groups
        self.n_contacts = n_contacts
        self.n_frames = n_frames
        self.frequency = frequency
        self.query = sum(self.ligand_groups.values(), [])
        self.haystack = topology.select("protein and symbol != 'H'")

        self.cutoff = 0.4

    def check_start(self, trajectory):
        """
        Parameters
        ----------
        trajectory : :class:`openpathsampling.Trajectory`
        """
        traj = trajectory_to_mdtraj(trajectory, md_topology=self.topology)
        if len(trajectory) < self.n_frames:
            return False
        subtraj = traj[0:self.n_frames]
        contacts = ContactFrequency(subtraj,
                                    query=self.query,
                                    haystack=self.haystack,
                                    cutoff=self.cutoff)
        most_common = \
                contacts.residue_contacts.most_common()[:self.n_contacts]
        if len(most_common) < self.n_contacts:
            return False
        if most_common[-1][1] >= self.frequency:
            return True
        else:
            return False

    def check_end(self, trajectory):
        if len(trajectory) < self.n_frames:
            return False
        else:
            return self.check_start(trajectory[-self.n_frames:])


class MultipleBindingEnsemble(paths.Ensemble):
    """
    Ensemble for sampling unknown binding sites.

    The stopping conditions (for both appending and prepending) are when the
    path either reaches a known stable state, or has a set of contacts that
    are stable for a fixed time period and during that time period is
    *never* in a given excluded volume.

    The definition of the contacts being stable is determined by the
    :class:`.StableContactState` object.
    """
    def __init__(self, initial_state, known_states, stable_contact_state,
                 excluded_volume=None, window_offset=None):
        self.initial_state = initial_state
        self.known_states = known_states
        self.states = paths.join_volumes(set([initial_state]+known_states))
        self.stable_contact_state = stable_contact_state
        if excluded_volume is None:
            excluded_volume = paths.EmptyVolume()
        self.excluded_volume = excluded_volume
        self.excluded_volume_ensemble = \
                paths.AllOutXEnsemble(self.excluded_volume)
        if window_offset is None:
            window_offset = self.stable_contact_state.n_frames / 2
        self.window_offset = window_offset

    def _check_length(self, trajectory):
        return len(trajectory) >= self.stable_contact_state.n_frames

    def _window_edge(self, trajectory):
        return (len(trajectory) % self.window_offset == 0)

    def _should_check_stable_contacts(self, trajectory):
        return (self._check_length(trajectory)
                and self._window_edge(trajectory))

    def _check_start(self, trajectory):
        start = self.states(trajectory[0])
        if not start and self._should_check_stable_contacts(trajectory):
            start_subtraj = trajectory[0:self.stable_contact_state.n_frames]
            start = (
                self.stable_contact_state.check_start(trajectory)
                and self.excluded_volume_ensemble(start_subtraj)
                # first part needs whole trajectory to ensure that we check
                # only on the right windows
            )
        return start

    def _check_end(self, trajectory):
        end = self.states(trajectory[-1])
        if not end and self._should_check_stable_contacts(trajectory):
            end_subtraj = trajectory[-self.stable_contact_state.n_frames:]
            end = (
                self.stable_contact_state.check_end(trajectory)
                and self.excluded_volume_ensemble(end_subtraj)
                # first part needs whole trajectory to ensure that we check
                # only on the right windows
            )
        return end

    def _all_subtrajectory_windows(self, trajectory):
        n_frames = self.stable_contact_state.n_frames
        window_offset = self.window_offset
        len_max = len(trajectory) - n_frames
        return [trajectory[offset : offset + n_frames]
                for offset in range(0, len_max, window_offset)]

    def _check_continue_one_frame(self, trajectory):
        frame = trajectory[0]
        if self.states(frame) and not self.initial_state(frame):
            return False  # one frame in a final state
        else:
            return True  # either 1 frame in initial, or 1 frame

    def can_append(self, trajectory, trusted=None):
        if len(trajectory) == 1:
            return self._check_continue_one_frame(trajectory)
        end = self._check_end(trajectory)
        if trusted:
            return not end
        else:
            pass  # TODO

    def can_prepend(self, trajectory, trusted=None):
        if len(trajectory) == 1:
            return self._check_continue_one_frame(trajectory)
        start = self._check_start(trajectory)
        if trusted:
            return not start
        else:
            pass  # TODO

    def __call__(self, trajectory, trusted=None, candidate=False):
        candidate_check = (self.initial_state(trajectory[0])
                           and self._check_end(trajectory))
        if candidate:
            return candidate_check
        elif candidate_check and not self.initial_state(trajectory[-1]):
            subtrajectories = self._all_subtrajectory_windows(trajectory)
            return not any([self._check_end(subtraj)
                            for subtraj in subtrajectories])
        else:
            return False

    def strict_can_append(self, trajectory, trusted=False):
        if not self.initial_state(trajectory[0]):
            return False
        return self.can_append(trajectory, trusted)

    def strict_can_prepend(self, trajectory, trusted=False):
        if not self._check_end(trajectory):
            return False
        return self.can_prepend(trajectory, trusted)


class MultipleBindingShootingPointSelector(paths.ShootingPointSelector):
    def __init__(self, multiple_binding_ensemble, subselector=None):
        self.multiple_binding_ensemble = multiple_binding_ensemble
        if subselector is None:
            subselector = paths.UniformSelector()
        self.subselector = subselector
        # cache the previously tested trajectory and the resulting
        # subtrajectory. This avoids wasteful recalc for most cases of
        # multiple use of `f`, `pick`, `sum_bias` for the same input traj
        self._cached_subtraj = paths.Trajectory([])
        self._cached_traj = paths.Trajectory([])

    def _get_subtrajectory(self, trajectory):
        if trajectory == self._cached_traj:
            return self._cached_subtraj
        else:
            ens = self.multiple_binding_ensemble  # convenience
            # note that we assume that the trajectory satisfies the
            # ensemble; if not, something has gone wrong in sampling
            if ens.states(trajectory[-1]):
                return trajectory
            else:
                # the +1 keeps the padding correct
                return trajectory[0:-ens.stable_contact_state.n_frames+1]

    def f(self, snapshot, trajectory):
        subtraj = self._get_subtrajectory(trajectory)
        return self.subselector.f(snapshot, subtraj)

    def pick(self, trajectory):
        subtraj = self._get_subtrajectory(trajectory)
        return self.subselector.pick(subtraj)

    def sum_bias(self, trajectory):
        subtraj = self._get_subtrajectory(trajectory)
        return self.subselector.sum_bias(subtraj)


# TODO: move these two into OPS: or better yet, change the current
# implementation in OPS base classes so that they work like this
class SingleEnsembleTransition(paths.Transition):
    def __init__(self, ensemble, stateA, stateB):
        super(SingleEnsembleTransition, self).__init__(stateA, stateB)
        self.ensembles = [ensemble]

class NetworkFromTransitions(paths.TransitionNetwork):
    # this is really how paths.TransitionNetwork should work
    def __init__(self, sampling_transitions, transitions):
        super(NetworkFromTransitions, self).__init__()
        self._sampling_transitions = sampling_transitions
        self.transitions=transitions
