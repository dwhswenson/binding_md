import itertools
import openpathsampling as paths
import mdtraj as md
from .utils import *
from binding_md.path_sampling import *


def setup_module():
    global bound_frame, unbound_frame, contact_frame, excl_vol_frame, \
            other_frame, topology, stable_contact_state
    fname = find_testfile("trajectory.pdb")
    traj = paths.engines.openmm.tools.ops_load_trajectory(fname)
    topology = traj.topology
    bound_frame = traj[0]
    excl_vol_frame = traj[1]
    contact_frame = traj[2]
    unbound_frame = traj[3]
    other_frame = traj[4]
    stable_contact_state = StableContactsState(
        topology=topology,
        ligand_groups={'ligand': [4, 5]},
        n_contacts=1,
        n_frames=4,
        frequency=0.7
    )
    stable_contact_state.haystack = [0, 1, 2, 3]
    stable_contact_state.cutoff = 0.075

def make_trajectory(string_representation):
    str2frame = {"B": bound_frame,
                 "U": unbound_frame,
                 "C": contact_frame,
                 "X": excl_vol_frame,
                 "O": other_frame}
    return paths.Trajectory([str2frame[char]
                             for char in string_representation.upper()])

def test_clean_direction():
    pytest.skip("Not implemented")

class TestStableContactsState(object):
    def setup(self):
        self.state = stable_contact_state

    def test_make_contact_frequency(self):
        pytest.skip("Not implemented")

    def test_subtraj_for_direction(self):
        pytest.skip("Not implemented")

    @pytest.mark.parametrize(
        'traj, result', [('ooo', False),
                         ('ccc', False),
                         ('ccoocc', False),
                         ('cccou', True),
                         ('ccxcob', True)]
    )
    def test_check_start(self, traj, result):
        trajectory = make_trajectory(traj)
        assert self.state.check_start(trajectory) == result

    @pytest.mark.parametrize(
        'traj, result', [('bocxcc', True),
                         ('boccc', True),
                         ('ccooocc', False),
                         ('ccc', False),
                         ('ooo', False)]
    )
    def test_check_end(self, traj, result):
        trajectory = make_trajectory(traj)
        assert self.state.check_end(trajectory) == result

    def test_call(self):
        assert self.state(make_trajectory('ccxc')) == True
        assert self.state(make_trajectory('oooo')) == False
        with pytest.raises(RuntimeError):
            self.state(make_trajectory('ccccc'))
        with pytest.raises(RuntimeError):
            self.state(make_trajectory('ccc'))


class TestMultipleBindingEnsemble(object):
    def setup(self):
        distance_pairs = list(itertools.product(range(4), range(4, 6)))
        distance = {
            frozenset(pair): paths.MDTrajFunctionCV(
                name="dist" + str(pair),
                f=md.compute_distances,
                topology=topology,
                atom_pairs=[list(pair)]
            )
            for pair in distance_pairs
        }
        # bound state
        # unbound state
        # stable contact state
        # excluded volume
        # self.ensemble = MultipleBindingEnsemble(
        # )
        self.accept = {key: make_trajectory(key)
                       for key in ['bocxu', 'bxcocc']}
        self.reject = {key: make_trajectory(key)
                       for key in ['bocxcc']}

    def test_can_append_accept(self):
        pass

    def test_can_append_reject(self):
        pass

    def test_can_prepend(self):
        pass

    def test_check_forward(self):
        pass

    def test_check_reverse(self):
        pass

    def test_strict_can_append(self):
        pass

    def test_strict_can_prepend(self):
        pass