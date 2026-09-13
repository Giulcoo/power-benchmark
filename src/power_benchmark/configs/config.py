import logging
import os
from functools import cache
from typing import Any, Literal, Dict, Optional, List, Union, Tuple

import yaml
from joblib import cpu_count
from pydantic import BaseModel, Field, model_validator

from power_benchmark.configs.PathResolver import PathResolver
from power_benchmark.configs.agent_config import CustomAgentConfig
from power_benchmark.configs.env_config import EnvConfig
from power_benchmark.configs.scenario_config import ScenarioConfig
from power_benchmark.configs.weight_config import WeightConfig
from power_benchmark.configs.task_config import Task
from power_benchmark.utils.path_utils import get_path_from_template, load_scenario, concat_path

logger = logging.getLogger("Benchmarker")

CONFIG_FILE_PATH = ""

class Config(BaseModel):
    name: str
    description: Optional[str] = None
    agent_type: Union[Literal["DoNothing", "Greedy"], CustomAgentConfig] # Either built-in agent type or custom agent
    env_config: Optional[EnvConfig] = None
    scenario_files: List[str] # Files with all scenarios configs
    output_dir: str = ""  # File to save benchmark results
    run_output_dir: str = "" # Directory to save/load run data # TODO: Remove this config
    task: List[str]
    logging_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "WARNING"
    seed: int = None # TODO: Either use this or remove it
    weights: Optional[WeightConfig] = None
    n_cpu: int | Literal["Slurm"]  # Number of CPUs, -1 for all CPUs, "Slurm" for amount Slurm allocated CPUs
    n_gpu: int = 0 # Number of GPUs, default no GPU usage
    save_run_data: bool = True # Whether to save run data (if task is only run the run data will be saved either way) # Todo: Remove this

    # Greedy
    num_workers: Optional[int] = Field(default=None, description="Number of workers for Greedy agent. Only used if agent_type is 'Greedy'.")

    # Plot or Webapp
    plotter: Optional[Literal["Matplotlib", "Plotly"] | Dict[Literal["Plot", "Webapp"], Literal["Matplotlib", "Plotly"]]] = None

    # Comparison App
    compare_with: Optional[str] = None  # Directory of results to compare with in the comparison webapp. Only used if task is Webapp.

    # Post Init Fields
    scenarios: List[ScenarioConfig] = Field(default_factory=list, exclude=True)
    evaluation_tasks: int = Field(default=0, exclude=True)
    eval_weights: Dict[str, Any] = Field(default={}, exclude=True)

    agent_cls: Any = Field(None, exclude=True)
    agent_init_params: Dict[str, Any] = Field(default_factory=dict, exclude=True)
    agent_checkpoint_dir: Optional[str] = Field(default=None, exclude=True)

    tuner_cls: Optional[Any] = Field(default=None, exclude=True)
    tuner_init_params: Dict[str, Any] = Field(default_factory=dict, exclude=True)

    @model_validator(mode='after')
    def case_checking(self):
        if Task.PLOT in self.task or Task.WEBAPP in self.task:
            if self.output_dir == "" or self.plotter is None:
                raise ValueError("output_dir and plotter must be specified for PLOT or WEBAPP tasks.")

        if isinstance(self.plotter, dict):
            if len(self.plotter) == 0:
                raise ValueError("Plotter dictionary cannot be empty.")
            if Task.PLOT in self.task and "Plot" not in self.plotter:
                raise ValueError("Plotter for 'Plot' must be specified in plotter dictionary.")
            if Task.WEBAPP in self.task and "Webapp" not in self.plotter:
                raise ValueError("Plotter for 'Webapp' must be specified in plotter dictionary.")

        return self

    def model_post_init(self, context: Any, /) -> None:
        logging.basicConfig(level=self.logging_level)

        if isinstance(self.logging_level, str):
            self.logging_level = logging.getLevelName(self.logging_level)

        if self.description is None:
            self.description = f"Experiment of {self.name}"

        if isinstance(self.n_cpu, str) and self.n_cpu == "Slurm":
            slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK') or os.environ.get('SLURM_NTASKS')
            if slurm_cpus is not None:
                logger.info(f"Setting n_cpu to SLURM allocated CPUs: {slurm_cpus}")
                self.n_cpu: int = int(slurm_cpus)
            else:
                logger.warning("SLURM_CPUS_ON_NODE not found in environment variables. Setting n_cpu to all available CPUs.")
                self.n_cpu = cpu_count()
        elif self.n_cpu > cpu_count():
            logger.warning("Number of jobs exceeds number of CPUs. Setting n_cpu to number of CPUs.")
            self.n_cpu = cpu_count()
        elif self.n_cpu == -1:
            self.n_cpu = cpu_count()
        self.num_workers = 1 # TODO: Give more workers or do better parallelization for Greedy
        if self.seed is None:
            import random
            self.seed = random.randint(0, 2**32 - 1)

        self.task: Task = Task.from_list(
            self.task,
            agent_type=self.agent_type if isinstance(self.agent_type, str) else "Custom",
            output_dir=self.output_dir
        )

        if self.task & Task.COMPUTE_TASKS:
            if self.weights is None:
                self.weights = WeightConfig()
            self.eval_weights = self.weights.to_dict()

            for file in self.scenario_files:
                scenarios_in_file = self.create_scenario(file, self.env_config)
                if isinstance(scenarios_in_file, list):
                    self.scenarios.extend(scenarios_in_file)
                else:
                    self.scenarios.append(scenarios_in_file)

            self.evaluation_tasks = sum([s.evaluation_tasks for s in self.scenarios])

            scenario_names = []
            for scenario in self.scenarios:
                if scenario.name in scenario_names:
                    raise ValueError(f"Duplicate scenario name found: {scenario.name}. Scenario names must be unique.")
                scenario_names.append(scenario.name)

        if isinstance(self.agent_type, CustomAgentConfig):
            self.agent_cls = self.agent_type.agent_cls
            self.agent_init_params = self.agent_type.init_params
            self.tuner_cls = self.agent_type.tuner_cls
            self.tuner_init_params = self.agent_type.tuner_init_params

        self.output_dir = get_path_from_template(
            self.output_dir,
            self.agent_type.class_name if isinstance(self.agent_type, CustomAgentConfig) else self.agent_type,
            self.seed
        )

        if self.compare_with is not None:
            self.compare_with = get_path_from_template(
                self.compare_with,
                self.agent_type.class_name if isinstance(self.agent_type, CustomAgentConfig) else self.agent_type,
                self.seed
            )

        if self.run_output_dir == "":
            self.run_output_dir = concat_path(self.output_dir, "run")
        else:
            self.run_output_dir = get_path_from_template(
                self.run_output_dir,
                self.agent_type.class_name if isinstance(self.agent_type, CustomAgentConfig) else self.agent_type,
                self.seed
            )

        # Create output folder if needed
        os.makedirs(self.output_dir, exist_ok=True)

        logger.info(self.__str__())

    @staticmethod
    def create_scenario(file_path: str, env_config: Optional[EnvConfig]) -> ScenarioConfig | List[ScenarioConfig]:
        def create_one(config: Dict[str, Any]) -> ScenarioConfig:
            return ScenarioConfig(**config, file_name=file_path, env_config=env_config)

        scenario_configs = load_scenario(file_path)
        if isinstance(scenario_configs, list):
            return [create_one(scenario_config) for scenario_config in scenario_configs]
        return create_one(scenario_configs)

    def __str__(self) -> str:
        tab = "    "
        scenarios_str = "\n".join(f"{tab}Scenario {i}: {scenario}"for i, scenario in enumerate(self.scenarios))
        return (
            f"PowerBenchmark Config {len(self.scenarios)} scenarios with {self.evaluation_tasks} evaluation tasks | "
            f"tasks: {str(self.task)} | "
            f"n_cpu: {self.n_cpu} | "
            f"n_gpu: {self.n_gpu} | "
            f"output directory: {self.output_dir}, "
            f"\n{scenarios_str}"
        )

def set_config_file(file_name: str) -> None:
    global CONFIG_FILE_PATH
    CONFIG_FILE_PATH = file_name

def get_config_file() -> str:
    return CONFIG_FILE_PATH

@cache
def get_config() -> Config:
    with open(CONFIG_FILE_PATH) as f:
        data = yaml.safe_load(f)

    data = PathResolver(data).resolve()
    return Config(**data)