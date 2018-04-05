import openpathsampling as paths
from openpathsampling.engines.openmm.tools import trajectory_to_mdtraj
from contact_map import ContactFrequency
from openpathsampling.netcdfplus import StorableNamedObject

import logging
logger = logging.Logger(__name__)

# conveniences for handling directions
FWD = +1
BKWD = -1

def clean_direction(direction):
    if direction > 0:
        return FWD
    elif direction < 0:
        return BKWD
    else:
        raise ValueError("Bad direction: " + str(direction))


class StableContactsState(StorableNamedObject):
    def __init__(self, topology, ligand_groups, n_frames, n_contacts,
                 frequency=0.95):
        """
        Parameters
        ----------
        topology : :class:`openpathsampling.engines.openmm.MDTrajTopology`
        ligand_groups : dict of {str: list of int}
        """
        self.topology = topology
        self.md_topology = topology.mdtraj
        self.ligand_groups = ligand_groups
        self.n_contacts = n_contacts
        self.n_frames = n_frames
        self.frequency = frequency
        self.query = sum(self.ligand_groups.values(), [])
        self.haystack = self.md_topology.select("protein and symbol != 'H'")
        self.cutoff = 0.4

    def make_contact_frequency(self, trajectory):
        """Convenience for making a ContactFrequency object.
        """
        return ContactFrequency(
            trajectory=trajectory_to_mdtraj(trajectory),
            query=self.query,
            haystack=self.haystack,
            cutoff=self.cutoff
        )

    def subtraj_for_direction(self, trajectory, direction):
        direction = clean_direction(direction)
        my_slice = {FWD: slice(-self.n_frames, None),
                    BKWD: slice(0, self.n_frames)}[direction]
        return trajectory[my_slice]

    def _check_one_side(self, trajectory, direction, cache=None):
        if len(trajectory) < self.n_frames:
            return False
        traj = trajectory_to_mdtraj(trajectory=trajectory,
                                    md_topology=self.md_topology)
        if not cache:
            subtraj = self.subtraj_for_direction(trajectory, direction)
            contacts = self.make_contact_frequency(subtraj)
        else:
            contacts = cache.updated_contact_frequency(trajectory,
                                                       direction)

        most_common = \
                contacts.residue_contacts.most_common()[:self.n_contacts]

        return (len(most_common) >= self.n_contacts
                and most_common[-1][1] >= self.frequency)


    def check_start(self, trajectory, cache=None):
        """
        Parameters
        ----------
        trajectory : :class:`openpathsampling.Trajectory`
        """
        return self._check_one_side(trajectory, direction=-1, cache=cache)

    def check_end(self, trajectory, cache=None):
        return self._check_one_side(trajectory, direction=+1, cache=cache)


class MultipleBindingEnsembleCache(object):
    """Cache of contact frequency that can be updated frame-by-frame.

    Parameters
    ----------
    ensemble : :class:`.MultipleBindingEnsemble`
    direction : +1 or -1
    """
    def __init__(self, ensemble, direction=+1):
        self.ensemble = ensemble
        self.direction = direction
        self.initial_frame = None
        self.final_frame = None
        self.last_trajectory_length = 0
        self.contact_frequency = None

    @property
    def _make_contact_freq(self):
        return self.ensemble.stable_contact_state.make_contact_frequency

    def updated_contact_frequency(self, trajectory, direction):
        if direction != self.direction:
            raise RuntimeError(
                "Wrong cache for direction: cache is %d; direction is %d",
                self.direction, direction
            )

        if not self.is_valid(trajectory, direction):
            self.reset(trajectory)
        else:
            self._update(trajectory)

        return self.contact_frequency

    def is_valid(self, trajectory, direction):
        """check whether the trajectory is compatible with cached info
        """
        if len(trajectory) < 2:
            return False

        direction = clean_direction(direction)
        prev_initial_frame, prev_final_frame = {
            FWD: (trajectory[0], trajectory[-2]),
            BKWD: (trajectory[1], trajectory[-1])
        }[direction]
        return (self.initial_frame == prev_initial_frame
                and self.final_frame == prev_final_frame)

    def reset(self, trajectory):
        """Reset the cache for a new trajectory
        """
        self.last_trajectory_length = len(trajectory)
        if self.last_trajectory_length == 0:
            self.initial_frame = None
            self.final_frame = None
        else:
            self.initial_frame = trajectory[0]
            self.final_frame = trajectory[-1]

        subtraj = self.ensemble.stable_contact_state.subtraj_for_direction(
            trajectory=trajectory,
            direction=self.direction
        )
        if len(subtraj) == 0:
            self.contact_frequency = None
        else:
            self.contact_frequency = self._make_contact_freq(subtraj)

    def _update(self, trajectory, direction=None):
        if direction is None:
            direction = self.direction
        direction = clean_direction(direction)
        n_frames = self.ensemble.stable_contact_state.n_frames

        self.initial_frame = trajectory[0]
        self.final_frame = trajectory[-1]

        add_frame = {FWD: trajectory[-1], BKWD: trajectory[0]}[direction]
        add_freq = self._make_contact_freq(paths.Trajectory([add_frame]))
        logger.debug("Adding frame to contact frequency")
        self.contact_frequency.add_contact_frequency(add_freq)

        if len(trajectory) > n_frames:
            sub_frame = {FWD: trajectory[-n_frames],
                         BKWD: trajectory[n_frames - 1]}[direction]
            sub_freq = self._make_contact_freq(paths.Trajectory([sub_frame]))
            logger.debug("Subtracting frame from contact frequency")
            self.contact_frequency.subtract_contact_frequency(sub_freq)

        logger.debug("Contact frequency based on length %d",
                     self.contact_frequency.n_frames)



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
        super(MultipleBindingEnsemble, self).__init__()
        self.initial_state = initial_state
        self.known_states = known_states
        self.final_state = paths.join_volumes(set(known_states)
                                              - set([initial_state]))
        self.states = paths.join_volumes(set([initial_state]+known_states))
        self.stable_contact_state = stable_contact_state
        if excluded_volume is None:
            excluded_volume = paths.EmptyVolume()
        self.excluded_volume = excluded_volume
        self.excluded_volume_ensemble = \
                paths.AllOutXEnsemble(self.excluded_volume)
        # if window_offset is None:
            # window_offset = self.stable_contact_state.n_frames / 2
        # self.window_offset = window_offset
        self.cache = {d: MultipleBindingEnsembleCache(self, direction=d)
                      for d in [FWD, BKWD]}

    # def _check_length(self, trajectory):
        # return len(trajectory) >= self.stable_contact_state.n_frames

    # def _window_edge(self, trajectory):
        # return (len(trajectory) % self.window_offset == 0)

    # def _should_check_stable_contacts(self, trajectory):
        # return (self._check_length(trajectory)
                # and self._window_edge(trajectory))


    def _trusted_analysis(self, trajectory, state, direction, is_check,
                          cache=None):
        """Generic function for testing checking and continuing criteria

        Parameters
        ----------
        trajectory : :class:`openpathsampling.Trajectory`
        direction : +1 or -1
        is_check : bool
            if True, this checks inclusion; if false, this is can_append or
            can_prepend
        state : :class:`openpathsampling.Volume`
            a single volume joining all relevant states
        cache : :class:`.MultipleBindingEnsembleCache`

        Return
        ------
        bool
        """
        direction = clean_direction(direction)
        if len(trajectory) == 1:
            return (self._check_continue_one_frame(trajectory, direction)
                    and not is_check)

        frame_idx = {FWD: -1, BKWD: 0}[direction]
        n_frames = self.stable_contact_state.n_frames
        subtraj_slice = {FWD: slice(-n_frames, None),
                         BKWD: slice(0, n_frames)}[direction]

        # length convenience
        ex_vol_ens = self.excluded_volume_ensemble
        excluded_volume_check = {
            (False, FWD): lambda traj: \
                ex_vol_ens.can_append(traj, trusted=True),
            (False, BKWD): lambda traj: \
                ex_vol_ens.can_prepend(traj, trusted=True),
            (True, FWD): lambda traj: \
                ex_vol_ens(traj, candidate=True),
            (True, BKWD): lambda traj: \
                ex_vol_ens.check_reverse(traj, candidate=True)
        }[is_check, direction]

        stable_contact_check = {
            FWD: self.stable_contact_state.check_end,
            BKWD: self.stable_contact_state.check_start
        }[direction]

        # conditions in which we cannot extend -- either
        #   1. endpoint is in a state
        #   2. subtraj has stable contacts and is outside excluded volume
        # the order here should make it so we short-circuit to avoid the
        # most expensive steps
        in_ensemble = (
            state(trajectory[frame_idx])
            or (
                excluded_volume_check(trajectory[subtraj_slice])
                and stable_contact_check(trajectory, cache=cache)
            )
        )
        # if is_check is False (i.e., doing can_append/prepend) then the
        # test is successful if we *are not* in the ensemble. If is_check is
        # True, then the test is successful if we *are* in the ensemble
        return (in_ensemble == is_check)

    def _untrusted_can_continue(self, trajectory, state, direction):
        """Can-append/prepend for untrusted trajectories

        Internally, we just create a cache and build up the trajectory using
        the approach with ``trusted=True``.

        Returns
        -------
        subtraj : :class:`.openpathsampling.Trajectory`
            the longest subtrajectory that allowed can-append
        cache : :class:`.MultipleBindingEnsembleCache`
            cache with details of the analysis of ``subtraj``
        """
        direction = clean_direction(direction)
        cache = MultipleBindingEnsembleCache(ensemble=self,
                                             direction=direction)
        n_frames = self.stable_contact_state.n_frames

        n_frames = 1
        can_continue = True
        while can_continue and n_frames <= len(trajectory):
            subtraj_slice = {FWD: slice(0, n_frames),
                             BKWD: slice(-n_frames, None)}[direction]
            subtraj = trajectory[subtraj_slice]
            can_continue = self._trusted_analysis(trajectory=subtraj,
                                                  state=state,
                                                  direction=direction,
                                                  is_check=False,
                                                  cache=cache)
            n_frames += 1
        return (subtraj, cache)

    def _check_continue_one_frame(self, trajectory, direction):
        (allowed, forbidden) = {
            FWD: (self.initial_state, self.final_state),
            BKWD: (self.final_state, self.initial_state)
        }[direction]
        frame = trajectory[0]
        return allowed(frame) or not forbidden(frame)

    def can_append(self, trajectory, trusted=None):
        if trusted:
            result = self._trusted_analysis(trajectory=trajectory,
                                            state=self.states,
                                            direction=FWD,
                                            is_check=False,
                                            cache=self.cache[FWD])
        else:
            subtraj, cache = self._untrusted_can_continue(
                trajectory=trajectory,
                state=self.states,
                direction=FWD
            )
            result = (len(subtraj) == len(trajectory)
                      and self._trusted_analysis(trajectory=subtraj,
                                                 state=self.states,
                                                 direction=FWD,
                                                 is_check=False,
                                                 cache=cache))
        return result

    def can_prepend(self, trajectory, trusted=None):
        pass
        # TODO

    def __call__(self, trajectory, trusted=None, candidate=False):
        if candidate:
            result = self._trusted_analysis(trajectory=trajectory,
                                            states=self.final_state,
                                            direction=FWD,
                                            is_check=True,
                                            cache=self.cache[FWD])
        else:
            subtraj, cache = self._untrusted_can_continue(
                trajectory=trajecory,
                state=self.states,
                direction=FWD
            )
            result = self._trusted_analysis(trajectory=subtraj,
                                            states=self.final_state,
                                            direction=FWD,
                                            is_check=True,
                                            cache=cache)
        return result

    def strict_can_append(self, trajectory, trusted=False):
        return (self.initial_state(trajectory[0])
                and self.can_append(trajectory, trusted))

    def strict_can_prepend(self, trajectory, trusted=False):
        pass #TODO


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


class StableContactsCommittorSimulation(paths.ShootFromSnapshotsSimulation):
    def __init__(self, storage, multiple_binding_ensemble, randomizer,
                 engine, initial_snapshots):
        starting_volume = ~multiple_binding_ensemble.states
        super(StableContactsCommittorSimulation, self).__init__(
            storage=storage,
            engine=engine,
            starting_volume=starting_volume,
            forward_ensemble=multiple_binding_ensemble,
            backward_ensemble=multiple_binding_ensemble,
            randomizer=randomizer,
            initial_snapshots=initial_snapshots
        )

