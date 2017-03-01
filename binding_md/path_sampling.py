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
    def __init__(self, known_states, stable_contact_state,
                 excluded_volume=None):
        self.known_states = known_states
        self.states = paths.join_volumes(known_states)
        self.stable_contact_state = stable_contact_state
        if excluded_volume is None:
            excluded_volume = paths.EmptyVolume()
        self.excluded_volume = excluded_volume
        self.excluded_volume_ensemble = \
                paths.AllOutXEnsemble(self.excluded_volume)

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
                self.excluded_volume_ensemble(start_subtraj)
                and self.stable_contact_state.check_start(start_subtraj)
            )
        return start

    def _check_end(self, trajectory):
        end = self.states(trajectory[-1])
        if not end and self._should_check_stable_contacts(trajectory):
            end_subtraj = trajectory[-self.stable_contact_state.n_frames:]
            end = (
                self.excluded_volume_ensemble(end_subtraj)
                and self.stable_contact_state.check_end(end_subtraj)
            )
        return end

    def can_append(self, trajectory, trusted=None):
        if trusted:
            return self._check_end(trajectory)
        else:
            pass  # TODO

    def can_prepend(self, trajectory, trusted=None):
        if trusted:
            return self._check_start(trajectory)
        else:
            pass  # TODO

    def __call__(self, trajectory, trusted=None, candidate=False):
        if candidate:
            start = self._check_start(trajectory)
            end = self._check_end(trajectory)
            return start and end
        else:
            pass  # TODO

    def strict_can_append(self, trajectory):
        pass  # TODO

    def strict_can_prepend(self, trajectory):
        pass  # TODO

