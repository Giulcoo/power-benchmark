import importlib
import logging
from typing import Literal, List, Any, Optional, Dict, Union

from pydantic import BaseModel, Field, model_validator, PrivateAttr

from power_benchmark.configs.env_config import EnvConfig
from power_benchmark.configs.range_config import RangeConfig, Index
from power_benchmark.configs.bottleneck_config import BottleneckConfig
from power_benchmark.scenarios.BaseNetwork import BaseNetwork
from power_benchmark.scenarios.NetworkModifier import NetworkModifier

logger = logging.getLogger("Benchmarker")

CATEGORIES = {
    "IEEE": ["14", "24", "30", "89"],
}

class NetModificationConfig(BaseModel):
    scale_loads: Dict[Literal["scale_p", "scale_q", "load_indices"], float | List[int]] = {}
    scale_generation: Dict[Literal["scale", "element_types", "element_indices"],
        float | List[Literal['gen', 'sgen', 'ext_grid']] | Dict[Literal['gen', 'sgen', 'ext_grid'], List[int]]] = {}
    scale_line_capacity: Dict[Literal["scale", "line_indices"], Union[float, List[int]]] = {}
    scale_trafo_capacity: Dict[Literal["scale", "trafo_indices", "trafo3w_indices"], Union[float, List[int]]] = {}

    line_maintenance: Optional[Dict[int, RangeConfig]] = None
    switch_maintenance: Optional[Dict[int, RangeConfig]] = None
    trafo_maintenance: Optional[Dict[int, RangeConfig]] = None
    trafo3w_maintenance: Optional[Dict[int, RangeConfig]] = None
    gen_maintenance: Optional[Dict[int, RangeConfig]] = None
    sgen_maintenance: Optional[Dict[int, RangeConfig]] = None

    @model_validator(mode="after")
    def validate_args(self) -> "NetModificationConfig":
        for key, value in self.scale_loads.items():
            if key in ["scale_p", "scale_q"]:
                if not isinstance(value, (float, int)):
                    raise ValueError(f"Value for '{key}' in scale_loads must be a number.")
            elif key == "load_indices":
                if not isinstance(value, list):
                    raise ValueError("Value for 'load_indices' in scale_loads must be a list of integers.")
                elif not all(isinstance(i, int) for i in value):
                    raise ValueError("All items in 'load_indices' must be integers.")
            else:
                raise ValueError(f"Invalid key '{key}' in scale_loads or invalid value type {type(value)}.")

        for key, value in self.scale_generation.items():
            if key == "scale":
                if not isinstance(value, (float, int)):
                    raise ValueError("Value for 'scale' in scale_generation must be a number.")
            elif key == "element_types":
                if not isinstance(value, list):
                    raise ValueError("Value for 'element_types' in scale_generation must be a list of strings.")
                elif not all(i in ['gen', 'sgen', 'ext_grid'] for i in value):
                    raise ValueError("All items in 'element_types' must be one of 'gen', 'sgen', or 'ext_grid'.")
            elif key == "element_indices":
                if not isinstance(value, dict):
                    raise ValueError("Value for 'element_indices' in scale_generation must be a dictionary.")
                elif not all(isinstance(k, str) and k in ['gen', 'sgen', 'ext_grid'] for k in value.keys()):
                    raise ValueError("All keys in 'element_indices' must be one of 'gen', 'sgen', or 'ext_grid'.")
                elif not all(isinstance(v, list) for v in value.values()):
                    raise ValueError("All values in 'element_indices' must be lists of integers.")
                elif not all(isinstance(i, int) for v in value.values() for i in v):
                    raise ValueError("All items in the lists of 'element_indices' must be integers.")
            else:
                raise ValueError(f"Invalid key '{key}' in scale_generation or invalid value type {type(value)}.")

        for key, value in self.scale_line_capacity.items():
            if key == "scale" and not isinstance(value, float):
                raise ValueError("Value for 'scale' in scale_line_capacity must be a float.")
            if key == "line_indices" and not isinstance(value, list):
                raise ValueError("Value for 'line_indices' in scale_line_capacity must be a list of integers.")
            if key not in ["scale", "line_indices"]:
                raise ValueError(f"Invalid key '{key}' in scale_line_capacity or invalid value type {type(value)}.")

        for key, value in self.scale_trafo_capacity.items():
            if key == "scale" and not isinstance(value, float):
                raise ValueError("Value for 'scale' in scale_line_capacity must be a float.")
            if key in ["trafo_indices", "trafo3w_indices"] and not isinstance(value, list):
                raise ValueError("Value for 'line_indices' in scale_line_capacity must be a list of integers.")
            if key not in ["scale", "trafo_indices", "trafo3w_indices"]:
                raise ValueError(f"Invalid key '{key}' in scale_line_capacity or invalid value type {type(value)}.")

        return self

    @staticmethod
    def maintenance_to_dict(maintenance: Optional[Dict[int, RangeConfig]]) -> Optional[Dict[int, Dict[str, Any]]]:
        if maintenance is None:
            return None
        return {k: v.to_dict() for k, v in maintenance.items()}

    def to_modifier(self) -> NetworkModifier:
        return NetworkModifier(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scale_loads": self.scale_loads,
            "scale_generation": self.scale_generation,
            "scale_line_capacity": self.scale_line_capacity,
            "scale_trafo_capacity": self.scale_trafo_capacity,
            "line_maintenance": self.maintenance_to_dict(self.line_maintenance),
            "switch_maintenance": self.maintenance_to_dict(self.switch_maintenance),
            "trafo_maintenance": self.maintenance_to_dict(self.trafo_maintenance),
            "trafo3w_maintenance": self.maintenance_to_dict(self.trafo3w_maintenance),
            "gen_maintenance": self.maintenance_to_dict(self.gen_maintenance),
            "sgen_maintenance": self.maintenance_to_dict(self.sgen_maintenance),
        }

class ScenarioConfig(BaseModel):
    name: Optional[str] = None
    category: Literal["Bottleneck", "IEEE", "Simbench"]
    case: Union[int, str, BottleneckConfig]
    indexes: Union[int, RangeConfig, List[int]]
    train_indexes: Optional[Union[int, RangeConfig, List[int]]] = None
    eval_indexes: Optional[Union[int, RangeConfig, List[int]]] = None
    number_of_runs: int = 1
    amount_of_days: int = 1
    timestep_hours: float = 0.25
    episode_length: int = 96
    verify_actions: bool = True
    handle_overloads: bool = False
    modification: Optional[NetModificationConfig | List[NetModificationConfig]] = None

    # Passed by main config
    env_config: EnvConfig

    # Post init Fields
    network: Optional[Any] = Field(default=None, exclude=True) # Contains BaseNetwork
    _eval_config: Optional[Dict[str, Any]] = PrivateAttr(default=None)
    _train_config: Optional[Dict[str, Any]] = PrivateAttr(default=None) # Eval Config will just take normal config for evaluation. For training another config is needed (with days filtered out)
    file_name: Optional[str] = Field(default=None, exclude=True)
    evaluation_tasks: int = 0

    @model_validator(mode='after')
    def validate_args(self):
        if isinstance(self.case, int):
            self.case = str(self.case)

        if self.category == "Bottleneck" and not isinstance(self.case, BottleneckConfig):
            raise ValueError("Bottleneck configs must be provided for Bottleneck category. Save the configs in 'case' argument.")
        if self.category != "Bottleneck" and isinstance(self.case, BottleneckConfig):
            raise ValueError("Bottleneck configs provided for non-Bottleneck category.")
        if self.category not in ["Bottleneck", "Simbench"] and self.case not in CATEGORIES[self.category]:
            raise ValueError(f"Unknown case '{self.case}' for category '{self.category}'. Available cases: {CATEGORIES[self.category]}")
        return self

    def model_post_init(self, context: Any, /) -> None:
        self.indexes: List[Index] = self.indexes_to_list(self.indexes, self.episode_length)

        if self.train_indexes is not None:
            self.train_indexes: List[Index] = self.indexes_to_list(self.train_indexes, self.episode_length)
        if self.eval_indexes is not None:
            self.eval_indexes: List[Index] = self.indexes_to_list(self.eval_indexes, self.episode_length)

        if self.train_indexes is not None and self.eval_indexes is None:
            self.eval_indexes: List[int] = [idx for idx in self.indexes if idx not in self.train_indexes]
        elif self.eval_indexes is not None and self.train_indexes is None:
            self.train_indexes: List[int] = [idx for idx in self.indexes if idx not in self.eval_indexes]
        elif self.train_indexes is not None and self.eval_indexes is not None:
            # Check if train and eval have shared indexes
            if set(self.train_indexes).intersection(set(self.eval_indexes)):
                logger.warning("Train and evaluation set share datapoints.")

        self.evaluation_tasks = len(self.eval_indexes) if self.eval_indexes is not None else len(self.indexes)

        modifier = []
        if isinstance(self.modification, NetModificationConfig):
            modifier.append(self.modification.to_modifier())
        elif isinstance(self.modification, list):
            for mod in self.modification:
                modifier.append(mod.to_modifier())

        # Load network class
        if self.category == "IEEE":
            module = importlib.import_module("power_benchmark.scenarios.IEEENetworks")
            class_name = f"Case{self.case}Network"

            self.network = getattr(module, class_name)(
                modifier=modifier,
                episode_length=self.episode_length,
            )
        elif self.category == "Simbench":
            from power_benchmark.scenarios.SimBenchNetworks import SimbenchNetwork
            self.network = SimbenchNetwork(self.case, modifier=modifier, episode_length=self.episode_length)

        logger.debug(f"Train/Eval config for scenario {self.scenario_name()}: Training indexes {self.train_indexes} | Eval indexes {self.eval_indexes}")

    @property
    def eval_config(self) -> Dict[str, Any]:
        if self._eval_config is not None:
            return self._eval_config

        if self.network is None:
            raise ValueError("No network configured.")

        extra_config = self.env_config.to_dict() if self.env_config is not None else {}
        extra_config["handle_overloads"] = self.handle_overloads
        self._eval_config = self.network.config(verify_actions=self.verify_actions, extra_config=extra_config)
        return self._eval_config

    @property
    def train_config(self) -> Dict[str, Any]:
        if self._train_config is not None:
            return self._train_config

        if self.network is None:
            raise ValueError("No network configured.")

        extra_config = self.env_config.to_dict() if self.env_config is not None else {}
        extra_config["handle_overloads"] = self.handle_overloads
        self._train_config = self.network.config(verify_actions=self.verify_actions, extra_config=extra_config, indexes=self.train_indexes)
        return self._train_config

    @staticmethod
    def indexes_to_list(indexes: Union[int, RangeConfig, List[int]], episode_length: int) -> List[Index]:
        if isinstance(indexes, int):
            return [Index.from_day(day=indexes, episode_length=episode_length)]
        elif isinstance(indexes, RangeConfig):
            return indexes.get_indexes(episode_length=episode_length)
        return [Index.from_day(day=i, episode_length=episode_length) for i in indexes]

    def scenario_name(self) -> str:
        """ Returns a unique name for the scenario based on its category and case. """
        return f"{self.category}_{self.case}_{self.amount_of_days}days" if self.name is None else self.name

    def run_data_path(self, run_output_dir: str) -> str:
        return run_output_dir + f"/{self.scenario_name()}"

    def __str__(self):
        indexes = self.eval_indexes if self.eval_indexes is not None else self.indexes

        if len(indexes) > 10:
            indexes_str = f"{indexes[:5]} ... {indexes[-5:]}"
        else:
            indexes_str = str(indexes)

        return f"Category: {self.category} | Case: {str(self.case)} | Name: {self.scenario_name()} | Indexes: {indexes_str}"

    def __eq__(self, other):
        if not isinstance(other, ScenarioConfig):
            return False
        return (
            self.category == other.category and
            self.case == other.case and
            self.indexes == other.indexes and
            self.number_of_runs == other.number_of_runs and
            self.amount_of_days == other.amount_of_days and
            self.timestep_hours == other.timestep_hours and
            self.file_name == other.file_name
        )

    def to_dict(self) -> dict:
        modification_config: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None

        if isinstance(self.modification, NetModificationConfig):
            modification_config = self.modification.to_dict()
        elif isinstance(self.modification, list):
            modification_config = [mod.to_dict() for mod in self.modification]

        return {
            "name": self.name,
            "category": self.category,
            "case": self.case if not isinstance(self.case, BottleneckConfig) else self.case.to_dict(),
            "indexes": Index.list_to_days(self.indexes),
            "train_indexes": Index.list_to_days(self.train_indexes) if self.train_indexes is not None else None,
            "eval_indexes": Index.list_to_days(self.eval_indexes) if self.eval_indexes is not None else None,
            "number_of_runs": self.number_of_runs,
            "amount_of_days": self.amount_of_days,
            "timestep_hours": self.timestep_hours,
            "episode_length": self.episode_length,
            "verify_actions": self.verify_actions,
            "modification": modification_config,
        }