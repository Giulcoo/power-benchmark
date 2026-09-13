from pandapower.networks import case14, case24_ieee_rts, case30, case89pegase

from pandapower_env.action_space.action_space import (
    create_unitary_substation_action,
    add_actions_substation_line_switching,
    add_actions_substation_line_pst, verify_all_actions,
)
from pandapower_env.substation.create_double_busbar_substation import create_all_double_busbar_substations
from pandapower_env.toolbox.utils_profiles import (
    create_simbench_data_from_profiles,
    get_first_sb_profiles,
    get_orig_profiles,
    scale_profiles,
)
from pandapower_env.toolbox.utils_scaling import ensure_no_zero_values, find_scaling_recursive
from pandapower.auxiliary import pandapowerNet as ppNet
import pandapower as pp
from pandapower_env.substation.create_double_busbar_substation import create_all_double_busbar_substations, create_all_dbb_or_3bbwpst_substations
import logging

from power_benchmark.scenarios.BaseNetwork import BaseNetwork

logger = logging.getLogger("Benchmarker")

class IEEENetwork(BaseNetwork):
    @staticmethod
    def load_and_scale_profile(net: pp.pandapowerNet, sb_index: int, init_scaling: int, max_percent: int, overloaded_lines: int):
        get_first_sb_profiles(net, sb_index=sb_index)

        ensure_no_zero_values(net)
        for key, df in net.profiles.items():
            net.profiles[key] = df.replace(0.0, 1.0)
        orig_profiles = get_orig_profiles(net)

        find_scaling_recursive(net, init_scaling=init_scaling, orig_profiles=orig_profiles, max_percent=max_percent,
                               overloaded_lines=overloaded_lines)
        scale_profiles(net, orig_profiles)

        create_simbench_data_from_profiles(net, orig_profiles)
        create_all_double_busbar_substations(net)

        # delete net columns
        for eltype in ("gen", "sgen", "load"):
            if hasattr(net[eltype], "scenario_scaling"):
                del net[eltype]["scenario_scaling"]

class Case14Network(IEEENetwork):
    @property
    def net(self) -> pp.pandapowerNet:
        net: pp.pandapowerNet = case14()
        self.load_and_scale_profile(net, sb_index=2, init_scaling=1, max_percent=1, overloaded_lines=3)
        return net

class Case24Network(IEEENetwork):
    @property
    def net(self) -> pp.pandapowerNet:
        net: pp.pandapowerNet = case24_ieee_rts()
        self.load_and_scale_profile(net, sb_index=2, init_scaling=1, max_percent=25, overloaded_lines=3)
        return net

class Case30Network(IEEENetwork):
    @property
    def net(self) -> pp.pandapowerNet:
        net: pp.pandapowerNet = case30()
        self.load_and_scale_profile(net, sb_index=2, init_scaling=1, max_percent=40, overloaded_lines=3)
        return net

    @property
    def actions(self) -> list:
        return add_actions_substation_line_switching(self.net)

class Case89Network(IEEENetwork):
    @property
    def net(self) -> pp.pandapowerNet:
        net: pp.pandapowerNet = case89pegase()
        self.load_and_scale_profile(net, sb_index=2, init_scaling=100, max_percent=80, overloaded_lines=4)
        return net

    @property
    def actions(self) -> list:
        actions =  add_actions_substation_line_switching(self.net)
        actions = verify_all_actions(actions)
        return actions

# def config_case30pst() -> dict:
#     net: pp.Pandapowernet = case30()
#     get_first_sb_profiles(net, 2)
#     ensure_no_zero_values(net)
#     for key, df in net.profiles.items():
#         net.profiles[key] = df.replace(0.0, 1.0)
#     orig_profiles = get_orig_profiles(net)
#     find_scaling_recursive(net, init_scaling=1, orig_profiles=orig_profiles, max_percent=40, overloaded_lines=3)
#     create_simbench_data_from_profiles(net, orig_profiles)
#     create_all_dbb_or_3bbwpst_substations(net, [5])
#     actions = add_actions_substation_line_pst(net)
#     # delete net columns
#     for eltype in ("gen", "sgen", "load"):
#         if hasattr(net[eltype], "scenario_scaling"):
#             del net[eltype]["scenario_scaling"]
#
#     logger.info(net.trafo)
#     logger.info(net.trafo3w)
#
#
#     return {
#         "net": net,
#         "n_episodes": 366,
#         "episode_length": 96,
#         "action_space": actions,
#         "nminus1": False,
#     }
