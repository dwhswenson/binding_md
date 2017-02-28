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
        if most_common[-1][1] >= self.frequency:
            return True
        else:
            return False

    def check_end(self, trajectory):
        if len(trajectory) < self.n_frames:
            return False
        else:
            return self.check_start(trajectory[-self.n_frames:])

