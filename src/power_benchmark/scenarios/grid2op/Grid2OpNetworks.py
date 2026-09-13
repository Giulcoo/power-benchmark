import pandapower as pp
import pickle
import os
from power_benchmark.scenarios.BaseNetwork import BaseNetwork

class Grid2opNetwork(BaseNetwork):
    class_path = os.path.dirname(__file__)

    @property
    def net(self) -> pp.pandapowerNet:
        # Load data paths
        net_path = os.path.join(self.class_path, "data", "grid2op_2022_done.p")

        return pp.from_pickle(net_path)

    @property
    def actions(self) -> list:
        actions_path = os.path.join(self.class_path, "data", "actions_grid2op_2022.pkl")

        # Load actions
        with open(actions_path, "rb") as f:
            actions = pickle.load(f)
        return actions