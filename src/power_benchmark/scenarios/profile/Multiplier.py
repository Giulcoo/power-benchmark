import logging
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from pandas import DatetimeIndex
from typing_extensions import override

logger = logging.getLogger("Benchmarker")

class Multiplier:
    def __init__(self,
        random_deviation: float = None,
    ):
        if random_deviation is not None:
            from power_benchmark.configs.config import get_config, get_config_file

            if get_config_file() is not "":
                np.random.seed(get_config().seed)
            else:
                logger.warning("Random deviation specified for Multiplier, but no config file found.")
            self.random_deviation = random_deviation

        pass

    def get_all_multiplier(self, time_index: DatetimeIndex) -> dict[str, np.ndarray]:
        hours: NDArray[np.float64] = time_index.hour + time_index.minute / 60

        return {
            'load_multiplier': self.add_randomness(self.create_one_multiplier(hours)),
            'gen_multiplier': self.add_randomness(self.create_one_multiplier(hours)),
            'sgen_multiplier': self.add_randomness(self.create_one_multiplier(hours)),
        }

    def create_one_multiplier(self, hours: NDArray[np.float64]) -> np.ndarray:
        return np.array([])

    def add_randomness(self, base_multiplier: np.ndarray) -> np.ndarray:
        if hasattr(self, 'random_deviation'):
            random_variation = 1 + self.random_deviation * np.random.randn(len(base_multiplier))
            return base_multiplier * random_variation
        else:
            return base_multiplier

class SinusoidalMultiplier(Multiplier):
    def __init__(self,
        random_deviation: float = None,
        starting_point: float = 1.0,
        amplitude: float = 0.1,
        frequency: float = 1.0,
        phase: float = 0.0,
    ):
        """
        Initializes a sinusoidal multiplier for load and generation profiles.
        Args:
            starting_point: Base value around which the sinusoidal variation occurs.
            amplitude: Amplitude of the sinusoidal variation.
            frequency: Frequency of the sinusoidal variation (cycles per day).
            phase: Phase shift of the sinusoidal variation (radians).
        """
        super().__init__(random_deviation)
        self.starting_point = starting_point
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase

    @override
    def create_one_multiplier(self, hours: NDArray[np.float64]) -> np.ndarray:
        multiplier = self.starting_point + self.amplitude * np.sin(2 * np.pi * self.frequency * hours / 24 + self.phase)
        return multiplier

class DailyLoadProfileMultiplier(Multiplier):
    """Typical household profile based on VDEW H0"""

    def __init__(self, random_deviation: float = None, profile_type: str = "household"):
        super().__init__(random_deviation)
        self.profile_type = profile_type

        # Definition of typical daily profiles for different consumer types
        self.profiles = {
            "household": [0.65, 0.63, 0.61, 0.60, 0.62, 0.70, 0.85, 0.95, 0.90, 0.85,
                          0.80, 0.78, 0.76, 0.75, 0.77, 0.82, 0.95, 1.10, 1.15, 1.10,
                          1.00, 0.90, 0.80, 0.70],
            "commercial": [0.50, 0.48, 0.47, 0.48, 0.52, 0.65, 0.85, 1.00, 1.05, 1.08,
                           1.10, 1.08, 1.05, 1.02, 1.00, 0.98, 0.95, 0.90, 0.80, 0.70,
                           0.60, 0.55, 0.52, 0.51],
            "industrial": [0.85, 0.85, 0.85, 0.85, 0.90, 0.95, 1.00, 1.05, 1.08, 1.10,
                           1.10, 1.10, 1.10, 1.10, 1.08, 1.05, 1.00, 0.95, 0.90, 0.88,
                           0.87, 0.86, 0.85, 0.85],
        }

    @override
    def create_one_multiplier(self, hours: NDArray[np.float64]) -> np.ndarray:
        profile = np.array(self.profiles[self.profile_type])
        # Interpolate linearly between hourly values
        hour_indices = hours % 24
        floor_hours = np.floor(hour_indices).astype(int)
        ceil_hours = (floor_hours + 1) % 24
        fraction = hour_indices - floor_hours

        multiplier = profile[floor_hours] * (1 - fraction) + profile[ceil_hours] * fraction
        return multiplier

class SeasonalMultiplier(Multiplier):
    """ Seasonal multiplier with daily variation based on cosine function"""

    def __init__(self,
                 random_deviation: float = None,
                 winter_factor: float = 1.3,
                 summer_factor: float = 0.8,
                 peak_winter_month: int = 1,  # January
                 peak_summer_month: int = 7):  # July
        super().__init__(random_deviation)
        self.winter_factor = winter_factor
        self.summer_factor = summer_factor
        self.peak_winter_month = peak_winter_month
        self.peak_summer_month = peak_summer_month

    @override
    def get_all_multiplier(self, time_index: DatetimeIndex) -> dict[str, np.ndarray]:
        hours = time_index.hour + time_index.minute / 60

        # Calculate seasonal factor
        day_of_year = time_index.dayofyear
        seasonal_factor = 1.0 + 0.3 * np.cos(2 * np.pi * (day_of_year - 15) / 365)

        base_multiplier = self.create_one_multiplier(hours)
        load_mult = base_multiplier * seasonal_factor

        return {
            'load_multiplier': self.add_randomness(load_mult),
            'gen_multiplier': self.add_randomness(base_multiplier),
            'sgen_multiplier': self.add_randomness(base_multiplier),
        }

    @override
    def create_one_multiplier(self, hours: NDArray[np.float64]) -> np.ndarray:
        return 0.8 + 0.2 * np.sin(2 * np.pi * (hours - 6) / 24)

class SolarMultiplier(Multiplier):
    """ Realistic solar generation profile based on time of day and season """

    @override
    def __init__(self,
                 random_deviation: float = 0.1, # Solar is less variable than wind, but still some variation
                 latitude: float = 51.0,  # Latitude of testing region for sun height estimation (e.g. 51° for Germany)
                 peak_power_ratio: float = 1.0):
        super().__init__(random_deviation)
        self.latitude = latitude
        self.peak_power_ratio = peak_power_ratio

    @override
    def get_all_multiplier(self, time_index: DatetimeIndex) -> dict[str, np.ndarray]:
        hours = time_index.hour + time_index.minute / 60.0
        day_of_year = time_index.dayofyear

        # Solar elevation based on latitude, day of year, and time
        delta_deg = 23.45 * np.sin(2 * np.pi * (day_of_year + 284) / 365.0)
        delta = np.deg2rad(delta_deg)
        phi = np.deg2rad(self.latitude)

        # Hour angle (radians), 12:00 local solar time = 0
        H = np.deg2rad(15.0 * (hours - 12.0))

        # Elevation angle of the sun
        sin_elev = np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.cos(H)
        elev = np.arcsin(np.clip(sin_elev, -1.0, 1.0))

        # Solar factor (0 before sunrise, 1 at noon, 0 after sunset)
        solar_factor = np.maximum(0.0, np.sin(elev))

        # Renewable generation: solar, scaled by peak_power_ratio
        sgen_raw = solar_factor * self.peak_power_ratio
        sgen_multiplier = self.add_randomness(sgen_raw)

        # Fossil generation: remaining demand not covered by solar
        fossil_raw = np.maximum(0.0, 1.0 - solar_factor)
        gen_multiplier = self.add_randomness(fossil_raw)

        # Load profile (consumption): typical diurnal curve with morning/evening peaks
        morning = np.maximum(0.0, np.sin(np.pi * (hours - 6.0) / 12.0))
        evening = np.maximum(0.0, np.sin(np.pi * (hours - 18.0) / 12.0))
        load_shape = (morning + evening) / 2.0
        load_raw = 0.4 + 0.6 * load_shape
        load_multiplier = self.add_randomness(load_raw)

        return {
            'load_multiplier': load_multiplier,
            'gen_multiplier': gen_multiplier,
            'sgen_multiplier': sgen_multiplier,
        }

    @override
    def create_one_multiplier(self, hours: NDArray[np.float64]) -> np.ndarray:
        return np.ones_like(hours)

class WindMultiplier(Multiplier):
    """ Realistic wind generation profile based on time of day and season"""

    @override
    def __init__(self,
                 random_deviation: float = 0.3,  # Wind is very variable
                 base_wind_speed: float = 7.0,  # m/s
                 turbulence: float = 0.2):
        super().__init__(random_deviation)
        self.base_wind_speed = base_wind_speed
        self.turbulence = turbulence

    @override
    def get_all_multiplier(self, time_index: DatetimeIndex) -> dict[str, np.ndarray]:
        n = len(time_index)
        hours = time_index.hour + time_index.minute / 60

        # Slow variation over the days
        day_variation = 0.8 + 0.4 * np.sin(2 * np.pi * time_index.dayofyear / 365)

        # Turbulence/random variation
        if hasattr(self, 'random_deviation'):
            turbulence = 1 + self.turbulence * np.random.randn(n)
        else:
            turbulence = np.ones(n)

        # Wind speed to power (simplified power curve)
        wind_speed = self.base_wind_speed * day_variation * turbulence

        # Power curve: P = 0 for v < 3, P ~ v³ for 3 < v < 12, P = const for v > 12
        power = np.where(wind_speed < 3, 0,
                         np.where(wind_speed < 12, ((wind_speed - 3) / 9) ** 3,
                                  1.0))

        return {
            'load_multiplier': np.ones_like(hours),
            'gen_multiplier': np.ones_like(hours),
            'sgen_multiplier': power,
        }

    @override
    def create_one_multiplier(self, hours: NDArray[np.float64]) -> np.ndarray:
        return np.ones_like(hours)

class CombinedMultiplier(Multiplier):
    """ Combine different multipliers for load, generation, and sgen independently """
    @override
    def __init__(self, gen_multiplier: list[Tuple[Multiplier, float]] | Multiplier,
                 sgen_multiplier: list[Tuple[Multiplier, float]] | Multiplier,
                 load_multiplier: list[Tuple[Multiplier, float]] | Multiplier):
        """
        Initializes a combined multiplier that aggregates multiple multipliers for load, generation, and sgen.
        :param gen_multiplier: Either a single Multiplier or a list of (Multiplier, weight) tuples for generation.
        :param sgen_multiplier: Either a single Multiplier or a list of (Multiplier, weight) tuples for sgen.
        :param load_multiplier: Either a single Multiplier or a list of (Multiplier, weight) tuples for load.
        """

        super().__init__()

        if isinstance(gen_multiplier, Multiplier):
            gen_multiplier = [(gen_multiplier, 1.0)]
        if isinstance(sgen_multiplier, Multiplier):
            sgen_multiplier = [(sgen_multiplier, 1.0)]
        if isinstance(load_multiplier, Multiplier):
            load_multiplier = [(load_multiplier, 1.0)]

        self.gen_multipliers, self.gen_weights = zip(*gen_multiplier)
        self.sgen_multipliers, self.sgen_weights = zip(*sgen_multiplier)
        self.load_multipliers, self.load_weights = zip(*load_multiplier)

        self.gen_multipliers: list[Multiplier]
        self.sgen_multipliers: list[Multiplier]
        self.load_multipliers: list[Multiplier]
        self.gen_weights: list[float]
        self.sgen_weights: list[float]
        self.load_weights: list[float]

    @override
    def get_all_multiplier(self, time_index: DatetimeIndex) -> dict[str, np.ndarray]:
        return {
            'load_multiplier': self.create_one_multiplier(time_index, self.load_multipliers, self.load_weights, 'load_multiplier'),
            'gen_multiplier': self.create_one_multiplier(time_index, self.gen_multipliers, self.gen_weights, 'gen_multiplier'),
            'sgen_multiplier': self.create_one_multiplier(time_index, self.sgen_multipliers, self.sgen_weights, 'sgen_multiplier'),
        }

    @override
    def create_one_multiplier(self, time_index: DatetimeIndex, multipliers: list[Multiplier], weights: list[float], category: str) -> np.ndarray:
        combined = np.zeros(len(time_index))

        total_weight = sum(weights)

        for multiplier, weight in zip(multipliers, weights):
            mult_dict = multiplier.get_all_multiplier(time_index)
            combined += mult_dict[category] * (weight / total_weight)

        return combined