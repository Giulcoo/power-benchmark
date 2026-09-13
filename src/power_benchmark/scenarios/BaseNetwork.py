from typing import List, Optional, Dict, Any

from power_benchmark.configs.range_config import Index
from power_benchmark.scenarios.NetworkModifier import NetworkModifier
import pandapower as pp
import pandas as pd

from pandapower_env.action_space.action_space import create_unitary_substation_action, verify_all_actions


class BaseNetwork:
    def __init__(self, modifier: List[NetworkModifier] | NetworkModifier = [], n_episodes: int = 366, episode_length: int = 96):
        self.modifier = [modifier] if isinstance(modifier, NetworkModifier) else modifier
        self.n_episodes = n_episodes
        self.episode_length = episode_length

    @property
    def net(self) -> pp.pandapowerNet:
        raise NotImplementedError

    @property
    def actions(self) -> list:
        return create_unitary_substation_action(self.net)

    @property
    def verified_actions(self) -> list:
        return verify_all_actions(self.net, self.actions)

    def config(self, verify_actions: bool = True, indexes: Optional[List[Index]] = None, extra_config: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        net = self.net
        for mod in self.modifier:
            net = mod.modify(net, self.episode_length)

        if indexes is not None:
            self.filter_profile_days(net, indexes, self.episode_length)

        return {
            "net": net,
            "n_episodes": self.n_episodes,
            "episode_length": self.episode_length,
            "action_space": self.verified_actions if verify_actions else self.actions,
            "fix_obs_space": False,
            **extra_config
        }

    @staticmethod
    def filter_profile_days(net: pp.pandapowerNet, indexes: List[Index], timesteps_per_day) -> None:
        """
        Keep only the timesteps belonging to the provided day indexes in all profiles.
        All other timesteps are removed.

        Args:
            net:                pandapower network to modify.
            indexes:            List of indexes to keep
            timesteps_per_day:  Number of timesteps per day (default: 96).
        """
        keep_timesteps = [
            ts
            for index in indexes
            for ts in range(index.day * timesteps_per_day, (index.day + 1) * timesteps_per_day)
        ]

        for key, profile in net.profiles.items():
            if not isinstance(profile, pd.DataFrame):
                continue

            if len(valid_timesteps := [ts for ts in keep_timesteps if ts in profile.index]) > 0:
                net.profiles[key] = profile.loc[valid_timesteps].reset_index(drop=True)