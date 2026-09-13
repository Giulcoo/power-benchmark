from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from power_benchmark.configs.multiplier_config import AnyMultiplierConfig
from power_benchmark.scenarios.BottleneckNetwork import BottleneckNetwork
from power_benchmark.scenarios.profile.Profile import Profile

class ProfileConfig(BaseModel):
    multiplier_config: AnyMultiplierConfig
    n_timesteps: int = None, # Number of timesteps in the profile, if None use full year 2024 with 15-minute intervals
    use_loads: bool = True, # Whether to include load profiles
    use_gens: bool = True, # Whether to include generator profiles
    use_sgens: bool = True, # Whether to include stochastic generator profiles

    # Post init Fields
    profile: Any = Field(None, exclude=True)

    def model_post_init(self, context: Any, /) -> None:
        self.profile = Profile(
            multiplier=self.multiplier_config.multiplier,
            n_timesteps=self.n_timesteps,
            use_loads=self.use_loads,
            use_gens=self.use_gens,
            use_sgens=self.use_sgens,
        )

class BottleneckConfig(BaseModel):
    # Bottleneck init
    n_gen: int
    n_load: int
    n_bottleneck: list
    base_vk: float
    percent_sgen: float = 0.0
    deg_gen: int = None
    deg_load: int = None
    deg_bottleneck: list = None
    total_p: float = None
    dist_gen_p: list[float] = None
    dist_load_p: list[float] = None
    profile_config: ProfileConfig

    # Build configs param
    episode_length: int = 96

    # Post Init Fields
    config: Dict = Field(default_factory=dict, exclude=True)

    def model_post_init(self, context: Any, /) -> None:
        self.config = BottleneckNetwork(
            n_gen=self.n_gen,
            n_load=self.n_load,
            n_bottleneck=self.n_bottleneck,
            base_vk=self.base_vk,
            percent_sgen=self.percent_sgen,
            deg_gen=self.deg_gen,
            deg_load=self.deg_load,
            deg_bottleneck=self.deg_bottleneck,
            total_p=self.total_p,
            dist_gen_p=self.dist_gen_p,
            dist_load_p=self.dist_load_p,
            profile=None, # TODO
            episode_length=self.episode_length,
        ).config()

    def __str__(self) -> str:
        return f"BottleneckConfig(n_gen={self.n_gen}, n_load={self.n_load}, n_bottleneck={self.n_bottleneck})"