import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging

class BaseTuner(ABC):
    """ Base of a hyperparameter tuner class that defines needed inputs and function for a tuner class inside the benchmark """
    def __init__(
            self,
            scenario_name: str,
            env_config: Dict[str, Any],
            results_dir: str,
            cpu_count: int,
            gpu_count: int = 0,
    ) -> None:
        self.scenario_name = scenario_name
        self.agent_config = env_config
        self.results_file = results_dir + "/hyperparams_" + scenario_name + ".json"
        self.cpu_count = cpu_count
        self.gpu_count = gpu_count

        self.logger = logging.getLogger("Benchmarker")

    @abstractmethod
    def tune(self) -> Dict[str, Any]:
        """ main function that implements the hyperparameter tuning and returns the best found hyperparameters """
        ...

    def _load_existing(self) -> Optional[Dict[str, Any]]:
        """Load cached results from file."""
        if not os.path.isfile(self.results_file):
            return None
        try:
            with open(self.results_file, "r") as f:
                data = json.load(f)
            if self._validate_loaded(data):
                return data
        except (json.JSONDecodeError, IOError) as e:
            self.logger.warning(f"Failed to load {self.results_file}: {e}")

        return None

    def _validate_loaded(self, data: Dict[str, Any]) -> bool:
        """
        Validate loaded JSON data. Override to add custom validation.

        Args:
            data: Parsed JSON dict.

        Returns:
            True if data is valid and can be used directly.
        """
        return True

    def _save_results(self, payload: Dict[str, Any]) -> None:
        """Save best configs and metadata."""
        os.makedirs(os.path.dirname(self.results_file) or ".", exist_ok=True)

        with open(self.results_file, "w") as f:
            json.dump(payload, f, indent=2)

        self.logger.info(f"Saved best hyperparameters to {self.results_file}")