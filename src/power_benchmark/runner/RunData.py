import json
import logging
import os
from abc import abstractmethod, ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional

from pandapower_env.environments.simulation_env import LoggedArray
from power_benchmark.configs.env_config import EnvConfig
from power_benchmark.configs.range_config import Index
from power_benchmark.configs.scenario_config import ScenarioConfig

logger = logging.getLogger("Benchmarker")

@dataclass
class BaseRunData(ABC):
    index: Index
    amount_of_days: Optional[int]

    @abstractmethod
    def save_data(self, output_dir: Path | str) -> None:
        ...

    @classmethod
    @abstractmethod
    def load_data(cls, output_dir: Path | str, index: Index, amount_of_days: Optional[int]) -> "RunData":
        ...

    @staticmethod
    def create_dir(output_dir: Path | str) -> Path:
        """Create a directory for the run data based on index and amount of days."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def save_run_data(self, output_dir: Path, run_data: Dict[str, Any]):
        with open(output_dir / self.scenario_filename(self.index, self.amount_of_days), "w") as f:
            json.dump(run_data, f, indent=2)

    @staticmethod
    def load_run_data(output_dir: Path, index: Index, amount_of_days: Optional[int]) -> Dict[str, Any]:
        with open(output_dir / BaseRunData.scenario_filename(index, amount_of_days), "r") as f:
            if f.seek(0, 2) == 0:
                raise FileNotFoundError(
                    f"Run data file not found for index {index} and amount of days {amount_of_days} in directory {output_dir}"
                )
            f.seek(0)
            run_data = json.load(f)
        return run_data

    @staticmethod
    def scenario_filename(index: Index, amount_of_days: Optional[int]) -> str:
        """Generate a filename for the scenario data based on index and amount of days."""
        index_str = f"index{index.index}_day{index.day}"
        index_str += f"_amountdays{amount_of_days}" if amount_of_days is not None and amount_of_days > 1 else ""
        return f"{index_str}.json"

@dataclass
class RunDataLight(BaseRunData):
    run_metrics: Dict[str, float | List[float]]
    index: Index
    amount_of_days: Optional[int]

    def save_data(self, output_dir: Path | str) -> None:
        """Save all run data to JSON files."""
        output_dir = self.create_dir(output_dir)

        # Save run data for this specific scenario
        self.save_run_data(output_dir, { "run_metrics": self.run_metrics })

    @classmethod
    def load_data(cls, output_dir: Path | str, index: Index, amount_of_days: Optional[int]) -> "RunDataLight":
        """Load all run data from JSON files."""
        output_dir = Path(output_dir)

        return cls(
            run_metrics=cls.load_run_data(output_dir, index, amount_of_days)["run_metrics"],
            index=index,
            amount_of_days=amount_of_days,
        )

@dataclass
class RunData(BaseRunData):
    """Container for all data passed from run to eval."""
    run_metrics: Dict[str, float | List[float]]
    run_actions: LoggedArray
    scenario_config: ScenarioConfig
    index: Index
    amount_of_days: Optional[int]

    def save_data(self, output_dir: Path | str) -> None:
        """Save all run data to JSON files."""
        output_dir = self.create_dir(output_dir)

        # Save run data for this specific scenario
        self.save_run_data(output_dir, {
            "run_metrics": self.run_metrics,
            "run_actions": list(self.run_actions.__iter__()),
        })

        self.write_scenario_config_once(output_dir, self.scenario_config)

    @staticmethod
    def write_scenario_config_once(output_dir: Path, scenario_config: ScenarioConfig) -> None:
        """ Thread-safe write of scenario config to file. """
        path = output_dir / "scenario_config.json"
        tmp_path = output_dir / f".{os.getpid()}.scenario_config.json.tmp"

        if path.exists():
            return

        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(scenario_config.to_dict(), f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_path, path)

    @classmethod
    def load_data(cls, output_dir: Path | str, index: Index, amount_of_days: Optional[int], env_config: Optional[EnvConfig] = None) -> "RunData":
        """Load all run data from JSON files."""
        if env_config is None:
            raise ValueError("env_config must be provided when loading RunData")

        output_dir = Path(output_dir)

        run_data = cls.load_run_data(output_dir, index, amount_of_days)

        with open(output_dir / f"scenario_config.json", "r") as f:
            scenario_config = json.load(f)

        run_actions = LoggedArray(len(run_data["run_actions"]))
        for run_action in run_data["run_actions"]:
            run_actions.append(run_action)

        return cls(
            run_metrics=run_data["run_metrics"],
            run_actions=run_actions,
            scenario_config=ScenarioConfig(**scenario_config, env_config=env_config),
            index=index,
            amount_of_days=amount_of_days,
        )

    def __str__(self):
        return self.scenario_config.name