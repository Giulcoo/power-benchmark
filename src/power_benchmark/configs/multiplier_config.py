from typing import Optional, Union, Literal, Any, Tuple

from pydantic import BaseModel, Field, model_validator

from power_benchmark.scenarios.profile.Multiplier import Multiplier

class BaseMultiplierConfig(BaseModel):
    type : Literal["Base"] = "Base"
    random_deviation: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Random deviation factor for adding noise to multipliers"
    )

    # Post init Fields
    multiplier: Any = Field(None, exclude=True)

    def model_post_init(self, context: Any, /) -> None:
        from power_benchmark.scenarios.profile.Multiplier import Multiplier
        self.multiplier = Multiplier(random_deviation=self.random_deviation)

class SinusoidalMultiplierConfig(BaseMultiplierConfig):
    type : Literal["Sinusoidal"] = "Sinusoidal"
    starting_point: float = Field(default=1.0, description="Base value around which the sinusoidal variation occurs")
    amplitude: float = Field(default=0.1, ge=0.0, description="Amplitude of the sinusoidal variation")
    frequency: float = Field(default=1.0, gt=0.0, description="Frequency of the sinusoidal variation (cycles per day)")
    phase: float = Field(default=0.0, description="Phase shift of the sinusoidal variation (radians)")

    def model_post_init(self, context: Any, /) -> None:
        from power_benchmark.scenarios.profile.Multiplier import SinusoidalMultiplier
        self.multiplier = SinusoidalMultiplier(
            random_deviation=self.random_deviation,
            starting_point=self.starting_point,
            amplitude=self.amplitude,
            frequency=self.frequency,
            phase=self.phase,
        )


class DailyLoadProfileMultiplierConfig(BaseMultiplierConfig):
    type : Literal["DailyLoadProfile"] = "DailyLoadProfile"
    profile_type: Literal["household", "commercial", "industrial"] = Field(
        default="household",
        description="Type of load profile based on VDEW H0"
    )

    def model_post_init(self, context: Any, /) -> None:
        from power_benchmark.scenarios.profile.Multiplier import DailyLoadProfileMultiplier
        self.multiplier = DailyLoadProfileMultiplier(
            random_deviation=self.random_deviation,
            profile_type=self.profile_type,
        )


class SeasonalMultiplierConfig(BaseMultiplierConfig):
    type : Literal["Seasonal"] = "Seasonal"
    winter_factor: float = Field(
        default=1.3,
        gt=0.0,
        description="Multiplicative factor for winter peak"
    )
    summer_factor: float = Field(
        default=0.8,
        gt=0.0,
        description="Multiplicative factor for summer"
    )
    peak_winter_month: int = Field(
        default=1,
        ge=1,
        le=12,
        description="Month of peak winter (default 1=January)"
    )
    peak_summer_month: int = Field(
        default=7,
        ge=1,
        le=12,
        description="Month of peak summer (default 7=July)"
    )

    @model_validator(mode='after')
    def validate_months_different(self) -> 'SeasonalMultiplierConfig':
        if self.peak_winter_month == self.peak_summer_month:
            raise ValueError("peak_winter_month and peak_summer_month must be different")
        return self

    def model_post_init(self, context: Any, /) -> None:
        from power_benchmark.scenarios.profile.Multiplier import SeasonalMultiplier
        self.multiplier = SeasonalMultiplier(
            random_deviation=self.random_deviation,
            winter_factor=self.winter_factor,
            summer_factor=self.summer_factor,
            peak_winter_month=self.peak_winter_month,
            peak_summer_month=self.peak_summer_month,
        )


class SolarMultiplierConfig(BaseMultiplierConfig):
    type : Literal["Solar"] = "Solar"
    random_deviation: float = Field(
        default=0.1,
        ge=0.0,
        description="Random deviation (solar is less variable than wind)"
    )
    latitude: float = Field(
        default=51.0,
        ge=-90.0,
        le=90.0,
        description="Latitude for sun height estimation (e.g., 51° for Germany)"
    )
    peak_power_ratio: float = Field(
        default=1.0,
        ge=0.0,
        description="Scaling factor for peak solar power output"
    )

    def model_post_init(self, context: Any, /) -> None:
        from power_benchmark.scenarios.profile.Multiplier import SolarMultiplier
        self.multiplier = SolarMultiplier(
            random_deviation=self.random_deviation,
            latitude=self.latitude,
            peak_power_ratio=self.peak_power_ratio,
        )


class WindMultiplierConfig(BaseMultiplierConfig):
    type : Literal["Wind"] = "Wind"
    random_deviation: float = Field(
        default=0.3,
        ge=0.0,
        description="Random deviation (wind is highly variable)"
    )
    base_wind_speed: float = Field(
        default=7.0,
        ge=0.0,
        description="Base wind speed in m/s"
    )
    turbulence: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Turbulence factor for wind speed variation"
    )

    def model_post_init(self, context: Any, /) -> None:
        from power_benchmark.scenarios.profile.Multiplier import WindMultiplier
        self.multiplier = WindMultiplier(
            random_deviation=self.random_deviation,
            base_wind_speed=self.base_wind_speed,
            turbulence=self.turbulence,
        )


# ============================================================================

# Weighted Multiplier für CombinedMultiplier

# ============================================================================

class WeightedMultiplierConfig(BaseModel):
    """A multiplier configs with an associated weight for combining."""
    type: Literal["Base", "Sinusoidal", "DailyLoadProfile", "Seasonal", "Solar", "Wind"] = "Base"
    configs: dict = Field(default_factory=dict)
    weight: float = Field(default=1.0, gt=0.0, description="Weight for this multiplier")

    # Post init Fields
    multiplier: Optional[Any] = Field(default=None, exclude=True)

    def model_post_init(self, context: Any, /) -> None:
        config_classes = {
            "Base": BaseMultiplierConfig,
            "Sinusoidal": SinusoidalMultiplierConfig,
            "DailyLoadProfile": DailyLoadProfileMultiplierConfig,
            "Seasonal": SeasonalMultiplierConfig,
            "Solar": SolarMultiplierConfig,
            "Wind": WindMultiplierConfig,
            "Combined": CombinedMultiplierConfig,
        }
        self.multiplier = config_classes[self.type](**self.config).multiplier

class MultiplierInputConfig(BaseModel):
    """
    Input configuration that accepts either a single multiplier or a weighted list.
    Exactly one of 'single' or 'weighted_list' must be provided.
    """
    single: Optional[WeightedMultiplierConfig] = Field(
        default=None,
        description="Single multiplier configuration (weight will be ignored)"
    )
    weighted_list: Optional[list[WeightedMultiplierConfig]] = Field(
        default=None,
        description="List of weighted multiplier configurations"
    )

    @model_validator(mode='after')
    def validate_exactly_one(self) -> 'MultiplierInputConfig':
        if self.single is None and self.weighted_list is None:
            raise ValueError("Either 'single' or 'weighted_list' must be provided")
        if self.single is not None and self.weighted_list is not None:
            raise ValueError("Only one of 'single' or 'weighted_list' can be provided")
        return self

    def get_multiplier_tuples(self) -> list[Tuple[Any, float]]:
        """Returns list of (Multiplier, weight) tuples for CombinedMultiplier."""
        if self.single is not None:
            return [(self.single.multiplier, 1.0)]
        return [(wm.multiplier, wm.weight) for wm in self.weighted_list]


class CombinedMultiplierConfig(BaseModel):
    """Configuration for combining multiple multipliers for load, gen, and sgen."""
    type : Literal["Combined"] = "Combined"
    gen_multiplier: MultiplierInputConfig = Field(
        ...,
        description="Multiplier configuration for generation"
    )
    sgen_multiplier: MultiplierInputConfig = Field(
        ...,
        description="Multiplier configuration for static generation (renewables)"
    )
    load_multiplier: MultiplierInputConfig = Field(
        ...,
        description="Multiplier configuration for load"
    )

    # Post init Fields
    multiplier: Optional[Any] = Field(default=None, exclude=True)

    def model_post_init(self, context: Any, /) -> None:
        from power_benchmark.scenarios.profile.Multiplier import CombinedMultiplier
        self.multiplier = CombinedMultiplier(
            gen_multiplier=self.gen_multiplier.get_multiplier_tuples(),
            sgen_multiplier=self.sgen_multiplier.get_multiplier_tuples(),
            load_multiplier=self.load_multiplier.get_multiplier_tuples(),
        )

AnyMultiplierConfig = Union[BaseMultiplierConfig, SinusoidalMultiplierConfig, DailyLoadProfileMultiplierConfig, SeasonalMultiplierConfig, SolarMultiplierConfig, WindMultiplierConfig, CombinedMultiplierConfig]