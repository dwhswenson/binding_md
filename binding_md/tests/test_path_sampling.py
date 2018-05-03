import itertools
import openpathsampling as paths
import mdtraj as md
from .utils import *
from binding_md.path_sampling import *

logger.addHandler(logging.StreamHandler())
ops_logger = logging.getLogger('openpathsampling')
ops_logger.setLevel(logging.INFO)

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
    logging.basicConfig()

def make_trajectory(string_representation):
    str2frame = {"B": bound_frame,
                 "U": unbound_frame,
                 "C": contact_frame,
                 "X": excl_vol_frame,
                 "O": other_frame}
    return paths.Trajectory([str2frame[char].copy()
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
    conditions = {
        'ext_stop': lambda result, subtraj, trajectory: \
            result == (len(subtraj) < len(trajectory)),
            # equiv to:
            #   if len(subtraj) < len(trajectory): return result == True
            #   else: return result == False
        'ext_nostop': lambda result, subtraj, trajectory: result == True,
        'check_acc': lambda result, subtraj, trajectory: \
            result == (len(subtraj) == len(trajectory)),
        'check_rej': lambda result, subtraj, trajectory: result == False
    }

    def setup(self):
        distance_pairs = list(itertools.product(range(4), range(4, 6)))
        dist = {
            pair: paths.MDTrajFunctionCV(
                name="dist" + str(pair),
                f=md.compute_distances,
                topology=topology,
                atom_pairs=[list(pair)]
            )
            for pair in distance_pairs
        }
        min_dist = paths.MDTrajFunctionCV(
            name="min_dist",
            topology=topology,
            f=lambda traj, pairs: \
                md.compute_distances(traj, atom_pairs=pairs).min(axis=1),
            pairs=list(dist.keys()),
            cv_scalarize_numpy_singletons=False
        )
        self.bound_state = paths.CVDefinedVolume(dist[(0, 4)], 0.0, 0.06) \
                and paths.CVDefinedVolume(dist[(1, 5)], 0.0, 0.06)
        self.unbound_state = paths.CVDefinedVolume(min_dist,
                                                   0.2, float("inf"))
        self.excluded_volume = (
            paths.CVDefinedVolume(dist[(0, 4)], 0.0, 0.075)
            and paths.CVDefinedVolume(dist[(0, 5)], 0.0, 0.075)
            and paths.CVDefinedVolume(dist[(1, 4)], 0.0, 0.075)
            and paths.CVDefinedVolume(dist[(1, 5)], 0.0, 0.075)
        )

        self.ensemble = MultipleBindingEnsemble(
            initial_state=self.bound_state,
            known_states=[self.bound_state, self.unbound_state],
            stable_contact_state=stable_contact_state,
            excluded_volume=self.excluded_volume
        )

    def test_setup_sanity(self):
        assert self.bound_state(bound_frame)
        assert self.unbound_state(unbound_frame)
        assert self.excluded_volume(excl_vol_frame)
        assert self.excluded_volume(bound_frame)
        assert not self.bound_state(excl_vol_frame)
        assert not self.excluded_volume(unbound_frame)
        assert not self.excluded_volume(contact_frame)

    @staticmethod
    def _generic_tester(traj_str, method, test_conditions, is_check,
                        direction):
        trajectory = make_trajectory(traj_str)
        for last_idx in range(len(trajectory)):
            logger.debug("RUNNING TEST ON TRAJ LEN %d", last_idx+1)
            subtraj_slice = {FWD: slice(0, last_idx+1),
                             BKWD: slice(-last_idx-1, None)}[direction]
            subtraj = trajectory[subtraj_slice]

            if is_check:
                result = method(subtraj)
            else:
                trusted = method(subtraj, trusted=True)
                untrusted = method(subtraj, trusted=False)
                assert trusted == untrusted
                result = trusted
            assert test_conditions(result, subtraj, trajectory)

    @pytest.mark.parametrize('traj', ['u', 'bocxu', 'bxcocc', 'xccxccco',
                                      'xcxcocococc'])
    def test_can_append_stop(self, traj):
        self._generic_tester(traj_str=traj,
                             method=self.ensemble.can_append,
                             test_conditions=self.conditions['ext_stop'],
                             is_check=False,
                             direction=FWD)

    @pytest.mark.parametrize('traj', ['bocxcc'])
    def test_can_append_no_stop(self, traj):
        self._generic_tester(traj_str=traj,
                             method=self.ensemble.can_append,
                             test_conditions=self.conditions['ext_nostop'],
                             is_check=False,
                             direction=FWD)

    @pytest.mark.parametrize('traj', ['bocxu', 'bxcocc', 'bocxcc', 'uoxu',
                                      'cccooooccc', 'coccoc'])
    def test_can_prepend_stop(self, traj):
        self._generic_tester(traj_str=traj,
                             method=self.ensemble.can_prepend,
                             test_conditions=self.conditions['ext_stop'],
                             is_check=False,
                             direction=BKWD)

    @pytest.mark.parametrize('traj', ['xccxccco', 'xcxcocococc', 'cccc'])
    def test_can_prepend_no_stop(self, traj):
        self._generic_tester(traj_str=traj,
                             method=self.ensemble.can_prepend,
                             test_conditions=self.conditions['ext_nostop'],
                             is_check=False,
                             direction=BKWD)

    @pytest.mark.parametrize('traj', ['bocxu', 'bxcocc', 'bxcxcocococc'])
    def test_check_forward_accept(self, traj):
        self._generic_tester(traj_str=traj,
                             method=self.ensemble,
                             test_conditions=self.conditions['check_acc'],
                             is_check=True,
                             direction=FWD)

    @pytest.mark.parametrize('traj', ['u', 'b', 'xxccco', 'bccc'])
    def test_check_forward_reject(self, traj):
        self._generic_tester(traj_str=traj,
                             method=self.ensemble,
                             test_conditions=self.conditions['check_rej'],
                             is_check=True,
                             direction=FWD)

    def test_check_reverse_accept(self):
        pytest.skip("Not implemented")

    def test_check_reverse_reject(self):
        pytest.skip("Not implemented")

    @pytest.mark.parametrize('traj', ['bocxu', 'bxcocc', 'u', 'x'])
    def test_strict_can_append_stop(self, traj):
        self._generic_tester(traj_str=traj,
                             method=self.ensemble.strict_can_append,
                             test_conditions=self.conditions['ext_stop'],
                             is_check=False,
                             direction=FWD)

    @pytest.mark.parametrize('traj', ['bccc', 'boxxcx'])
    def test_strict_can_append_no_stop(self, traj):
        self._generic_tester(traj_str=traj,
                             method=self.ensemble.strict_can_append,
                             test_conditions=self.conditions['ext_nostop'],
                             is_check=False,
                             direction=FWD)

    @pytest.mark.parametrize('traj', ['coccoc', 'cccooooccc', 'boccc',
                                      'uococc', 'bocxu', 'bxcocc', 'uoxu'])
    def test_strict_can_prepend_stop(self, traj):
        self._generic_tester(traj_str=traj,
                             method=self.ensemble.strict_can_prepend,
                             test_conditions=self.conditions['ext_stop'],
                             is_check=False,
                             direction=BKWD)

    @pytest.mark.parametrize('traj', ['xccxccco', 'xcxcocococc', 'cccc'])
    def test_strict_can_prepend_no_stop(self, traj):
        self._generic_tester(traj_str=traj,
                             method=self.ensemble.strict_can_prepend,
                             test_conditions=self.conditions['ext_nostop'],
                             is_check=False,
                             direction=BKWD)
