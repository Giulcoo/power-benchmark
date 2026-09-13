import pandapower as pp
from pandapower.auxiliary import pandapowerNet

from pandapower_env.action_space.action_space import (
    create_unitary_substation_action,
)
from pandapower_env.substation.create_double_busbar_substation import create_all_double_busbar_substations
from power_benchmark.scenarios.BaseNetwork import BaseNetwork

from power_benchmark.scenarios.profile.Profile import Profile
from power_benchmark.scenarios.profile.Multiplier import SinusoidalMultiplier
from typing import Optional
from power_benchmark.scenarios.NetworkModifier import NetworkModifier


class BottleneckNetwork(BaseNetwork):
    """
    Creates a pandapower network usable for PPTopoGym with a multi-layer bottleneck structure.
    """
    def __init__(self,
                 n_gen: int,
                 n_load: int,
                 n_bottleneck: list,
                 base_vk: float,
                 percent_sgen: float = 0.0,
                 deg_gen: int = None,
                 deg_load: int = None,
                 deg_bottleneck: list = None,
                 total_p: float = None,
                 dist_gen_p: list[float] = None,
                 dist_load_p: list[float] = None,
                 profile: Profile = None,
                 modifier: Optional[NetworkModifier] = None, n_episodes: int = 366, episode_length: int = 96
    ):
        """
        Initializes a bottleneck network scenarios with specified parameters.
        :param n_gen: Number of generator nodes (if percent_sgen > 0, some will be static generators).
        :param n_load: Number of load nodes.
        :param n_bottleneck: List specifying number of nodes in each bottleneck layer.
        :param base_vk: Base voltage level for all buses.
        :param percent_sgen: Percentage of generators that are static generators (SGEN). Default is 0.0 (all are GEN).
        :param deg_gen: Degree of connections from generators to first bottleneck layer.
        :param deg_load: Degree of connections from last bottleneck layer to loads.
        :param deg_bottleneck: List specifying degree of connections between bottleneck layers.
        :param total_p: Total power generation/load in the network. If set to None, it's set to n_load.
        :param dist_gen_p: Distribution in percent (from 0.0 to 1.0) of power generation among generators. If None, uniform distribution is used.
        :param dist_load_p: Distribution in percent (from 0.0 to 1.0) of power load among loads. If None, uniform distribution is used.
        """
        # Layer nodes
        self.n_gen = n_gen
        self.n_load = n_load
        self.n_bottleneck = n_bottleneck
        self.percent_sgen = percent_sgen

        # Base voltage
        self.base_vk = base_vk

        # Degrees
        self.deg_gen = deg_gen
        self.deg_load = deg_load
        self.deg_bottleneck = deg_bottleneck
        self.num_layers = len(n_bottleneck) if isinstance(n_bottleneck, list) else 1

        # Power distribution
        self.total_p = total_p if total_p is not None else n_load
        self.dist_gen_p = [val * self.total_p for val in dist_gen_p] if dist_gen_p is not None else [self.total_p/self.n_gen for _ in range(self.n_gen)]
        self.dist_load_p = [val * self.total_p for val in dist_load_p] if dist_load_p is not None else [self.total_p/self.n_load for _ in range(self.n_load)]

        # Validate and init network and environment
        self._check_params()

        # Create pandapower network with bottleneck structure and profile
        self.profile = profile if profile is not None else Profile(SinusoidalMultiplier(random_deviation=1.0))

    @property
    def net(self) -> pp.pandapowerNet:
        return self.profile.add_profile(self._create_net())

    def _check_params(self):
        # Set default degrees
        if self.deg_gen is None:
            self.deg_gen = self.n_bottleneck[0]
        if self.deg_load is None:
            self.deg_load = self.n_bottleneck[-1]
        if self.deg_bottleneck is None:
            # Default: connect to all nodes in next layer for each connection
            self.deg_bottleneck = [self.n_bottleneck[i + 1] for i in range(self.num_layers - 1)]

        # Validate deg_bottleneck
        if len(self.deg_bottleneck) != self.num_layers - 1:
            raise ValueError(f"deg_bottleneck must have length {self.num_layers - 1} "
                             f"(one degree per connection between layers), but got {len(self.deg_bottleneck)}.")

        # Set maximum degrees based on number of nodes
        self.deg_gen = min(self.deg_gen, self.n_bottleneck[0])
        self.deg_load = min(self.deg_load, self.n_bottleneck[-1])
        self.deg_bottleneck = [min(deg, self.n_bottleneck[i + 1]) for i, deg in enumerate(self.deg_bottleneck)]

        # Validate degrees
        if self.deg_gen <= 0 <= 0 or self.deg_load <= 0:
            raise ValueError("deg_gen and deg_load must be positive integers.")

        for i, deg in enumerate(self.deg_bottleneck):
            if deg <= 0:
                raise ValueError(f"deg_bottleneck[{i}] must be a positive integer, got {deg}.")

        # Check if every layer can be fully connected given the degrees
        for layer_idx in range(self.num_layers - 1):
            n_current = self.n_bottleneck[layer_idx]
            n_next = self.n_bottleneck[layer_idx + 1]
            degree = self.deg_bottleneck[layer_idx]
            if n_current * degree < n_next:
                raise ValueError(f"Cannot fully connect layer {layer_idx} to layer {layer_idx + 1} "
                                 f"with {n_current} nodes and degree {degree} "
                                 f"to cover {n_next} nodes in next layer.")

        if self.n_gen * self.deg_gen < self.n_bottleneck[0]:
            raise ValueError(f"Cannot fully connect generators to first bottleneck layer "
                             f"with {self.n_gen} generators and degree {self.deg_gen} "
                             f"to cover {self.n_bottleneck[0]} nodes.")

        if self.n_load * self.deg_load < self.n_bottleneck[-1]:
            raise ValueError(f"Cannot fully connect last bottleneck layer to loads "
                             f"with {self.n_load} loads and degree {self.deg_load} "
                             f"to cover {self.n_bottleneck[-1]} nodes.")

        # Validate power distributions
        if sum(self.dist_gen_p) != self.total_p:
            raise ValueError(f"Sum of dist_gen_p must equal total_p ({self.total_p}), but got {sum(self.dist_gen_p)}."
                             f" Set dist_gen_p values in percent (0-100) or ensure they sum 100.")
        if sum(self.dist_load_p) != self.total_p:
            raise ValueError(f"Sum of dist_load_p must equal total_p ({self.total_p}), but got {sum(self.dist_load_p)}."
                             f" Set dist_load_p values in percent (0-100) or ensure they sum 100.")

    def _create_net(self) -> pandapowerNet:
        """
        Creates a pandapower network with a multi-layer bottleneck structure.
        """
        net = pp.create_empty_network()

        # Create all bottleneck layers
        bottleneck_layers = []
        for layer_idx, n_nodes in enumerate(self.n_bottleneck):
            layer_buses = [pp.create_bus(net, vn_kv=self.base_vk, name=f"Layer_{layer_idx}_Node_{i}")
                           for i in range(n_nodes)]
            bottleneck_layers.append(layer_buses)

        # Left layer: GEN and SGEN buses
        sgen_counter = 0
        gen_buses = []
        for i in range(self.n_gen):
            sgen_counter += self.percent_sgen

            if sgen_counter >= 1.0:
                sgen_counter -= 1.0
                gen_buses.append(
                    pp.create_bus(net, vn_kv=self.base_vk, name=f"Static Generator {i}")
                )
            else:
                gen_buses.append(
                    pp.create_bus(net, vn_kv=self.base_vk, name=f"Generator {i}")
                )

        # Right layer: Load buses
        load_buses = [pp.create_bus(net, vn_kv=self.base_vk, name=f"Load {i}")
                      for i in range(self.n_load)]

        # Connect each GEN to deg_gen nodes in first bottleneck layer
        sgen_counter = 0
        for i, sbus in enumerate(gen_buses):
            sgen_counter += self.percent_sgen

            if sgen_counter >= 1.0:
                # Create static generator
                sgen_counter -= 1.0
                pp.create_sgen(net, bus=sbus, p_mw=self.dist_gen_p[i], q_mvar=0.0,
                               name=f"Static Generator {i}", profile= f"Generator {i}", controllable=True,
                               slack= i==0)
            else:
                # Create regular generator
                pp.create_gen(net, bus=sbus, p_mw=self.dist_gen_p[i], q_mvar=0.0,
                              name=f"Generator {i}", profile= f"Generator {i}", controllable=True,
                              slack= i==0)

            start_idx = (i * self.deg_gen) % self.n_bottleneck[0]
            for j in range(self.deg_gen):
                bottleneck_idx = (start_idx + j) % self.n_bottleneck[0]
                pp.create_line(net, sbus, bottleneck_layers[0][bottleneck_idx],
                               length_km=0.2, std_type="NAYY 4x50 SE")

        # Connect bottleneck layers to each other
        for layer_idx in range(self.num_layers - 1):
            current_layer = bottleneck_layers[layer_idx]
            next_layer = bottleneck_layers[layer_idx + 1]
            n_next = len(next_layer)

            # Get degree for this specific connection
            degree = min(self.deg_bottleneck[layer_idx], n_next)

            for i, current_bus in enumerate(current_layer):
                start_idx = (i * degree) % n_next
                for j in range(degree):
                    next_idx = (start_idx + j) % n_next
                    pp.create_line(net, current_bus, next_layer[next_idx],
                                   length_km=0.2, std_type="NAYY 4x50 SE")

        # Connect each Load to deg_load nodes in last bottleneck layer
        last_layer = bottleneck_layers[-1]
        for i, lbus in enumerate(load_buses):
            pp.create_load(net, bus=lbus, p_mw=self.dist_load_p[i], q_mvar=0.0,
                           name=f"Load {i}", profile= f"Load {i}", controllable=False,
                           slack=i == len(load_buses) - 1)

            start_idx = (i * self.deg_load) % self.n_bottleneck[-1]
            for j in range(self.deg_load):
                bottleneck_idx = (start_idx + j) % self.n_bottleneck[-1]
                pp.create_line(net, last_layer[bottleneck_idx], lbus,
                               length_km=0.2, std_type="NAYY 4x50 SE")

        return net