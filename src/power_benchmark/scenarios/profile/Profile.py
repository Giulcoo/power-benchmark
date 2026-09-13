import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandapower.auxiliary import pandapowerNet
from pandas import DatetimeIndex

from power_benchmark.scenarios.profile.Multiplier import Multiplier


class Profile:
    def __init__(self,
        multiplier: Multiplier,
        n_timesteps: int = None,
        use_loads: bool = True,
        use_gens: bool = True,
        use_sgens: bool = True,
    ):
        """
        Initializes a profile for adding time-series data to a pandapower network.
        :param multiplier: Multiplier object to generate load and generation multipliers.
        :param n_timesteps:  Number of time steps for the profile. If None, uses full year 2024 with 15-minute intervals.
        """
        self.multiplier = multiplier
        self.n_timesteps = n_timesteps
        self.use_loads = use_loads
        self.use_gens = use_gens
        self.use_sgens = use_sgens

    def add_profile(self, net: pandapowerNet) -> pandapowerNet:
        """
                Adds time-series profiles for loads and generators to the network.
                Creates profiles for the entire year 2024 with 15-minute intervals.

                Args:
                    net: pandapower network
                    n_timesteps: number of time steps (if None, uses full year 2024)

                Returns:
                    net: pandapower network with added profiles
                """
        # Create time index for entire year 2024 with 15-minute intervals
        if self.n_timesteps is None:
            time_index: DatetimeIndex = pd.date_range("2024-01-01 00:00", "2024-12-31 23:45", freq="15min")
        else:
            time_index: DatetimeIndex = pd.date_range("2024-01-01 00:00", periods=self.n_timesteps, freq="15min")

        # Format time column as "DD.MM.YYYY HH:MM"
        time_column = time_index.strftime("%d.%m.%Y %H:%M")

        # Create multipliers
        multipliers = self.multiplier.get_all_multiplier(time_index)
        load_multiplier = multipliers['load_multiplier']
        gen_multiplier = multipliers['gen_multiplier']
        sgen_multiplier = multipliers['sgen_multiplier']

        # Assign to net.profiles
        net.profiles = {
            'load': self.get_load_profile(net, time_column, load_multiplier),
            'powerplants': self.get_generator_profile(net, time_column, gen_multiplier),
            'renewables': self.get_renewable_profile(net, time_column, sgen_multiplier),
        }

        return net

    def get_load_profile(self, net: pandapowerNet, time_column: pd.Index, load_multiplier: NDArray[np.float64]) -> pd.DataFrame:
        columns = {'time': time_column}

        if self.use_loads:
            for load_idx in net.load.index:
                base_p = net.load.at[load_idx, 'p_mw']
                base_q = net.load.at[load_idx, 'q_mvar']

                columns[f'Load {load_idx}_pload'] = base_p * load_multiplier
                columns[f'Load {load_idx}_qload'] = base_q * load_multiplier

        return pd.DataFrame(columns)

    def get_generator_profile(self, net: pandapowerNet, time_column: pd.Index, gen_multiplier: NDArray[np.float64]) -> pd.DataFrame:
        columns = {'time': time_column}

        if self.use_gens:
            for gen_idx in net.gen.index:
                base_p = net.gen.at[gen_idx, 'p_mw']
                columns[f'Generator {gen_idx}'] = base_p * gen_multiplier

        return pd.DataFrame(columns)

    def get_renewable_profile(self, net: pandapowerNet, time_column: pd.Index, sgen_multiplier: NDArray[np.float64]) -> pd.DataFrame:
        columns = {'time': time_column}

        if self.use_sgens:
            for sgen_idx in net.sgen.index:
                base_p = net.sgen.at[sgen_idx, 'p_mw']
                columns[f'SGen {sgen_idx}'] = base_p * sgen_multiplier

        return pd.DataFrame(columns)