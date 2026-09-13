import pandapower as pp
import simbench as sb
from typing import List

from power_benchmark.scenarios.NetworkModifier import NetworkModifier

from pandapower_env.substation.create_double_busbar_substation import create_all_double_busbar_substations
from power_benchmark.scenarios.BaseNetwork import BaseNetwork


class SimbenchNetwork(BaseNetwork):
    def __init__(self, simbench_net: str, modifier: List[NetworkModifier] | NetworkModifier = [], n_episodes: int = 366, episode_length: int = 96):
        super().__init__(modifier=modifier, n_episodes=n_episodes, episode_length=episode_length)
        self.simbench_net = simbench_net

    @property
    def net(self) -> pp.pandapowerNet:
        net = sb.get_simbench_net(self.simbench_net)
        net.profiles = sb.get_all_simbench_profiles(scenario=0) #TODO: Extract scenario from name?
        create_all_double_busbar_substations(net)
        return net