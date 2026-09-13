import importlib
import os
from typing import Any, Optional, Dict

from pydantic import BaseModel, Field

from power_benchmark.agent.BenchAgent import BenchAgent
from power_benchmark.agent.tuning.BaseTuner import BaseTuner
from power_benchmark.utils.path_utils import get_path_from_template


class TunerConfig(BaseModel):
    results_dir: str
    module_name: str
    class_name: str
    init_params: Dict[str, Any] = {}

    def model_post_init(self, context: Any, /) -> None:
        self.results_dir = get_path_from_template(self.results_dir)
        self.init_params["results_dir"] = self.results_dir
        os.makedirs(self.results_dir, exist_ok=True)

class CustomAgentConfig(BaseModel):
    checkpoint_dir: str
    module_name: str
    class_name: str
    init_params: Dict[str, Any] = {}

    tuner_config: Optional[TunerConfig] = None

    # Post Init Fields
    agent_cls: Any = Field(None, exclude=True)
    tuner_cls: Optional[Any] = Field(None, exclude=True)
    tuner_init_params: Optional[Dict[str, Any]] = Field(default_factory=dict, exclude=True)

    def model_post_init(self, context: Any, /) -> None:
        self.checkpoint_dir = get_path_from_template(self.checkpoint_dir, agent_type=self.class_name, seed=-1)

        self.agent_cls = self.load_class(self.module_name, self.class_name)
        self.tuner_cls = self.load_class(self.tuner_config.module_name, self.tuner_config.class_name) if self.tuner_config is not None else None
        self.tuner_init_params = self.tuner_config.init_params if self.tuner_config is not None else None

        if not issubclass(self.agent_cls, BenchAgent):
            raise ValueError(f"Class type set in module_path and class_name must be a subclass of BenchAgent. Got {self.agent_cls}.")

        if self.tuner_cls is not None and not issubclass(self.tuner_cls, BaseTuner):
            raise ValueError(f"Class type set in tuner_config module_path and class_name must be a subclass of BaseTuner. Got {self.tuner_cls}.")

        self.init_params["checkpoint_dir"] = self.checkpoint_dir

        os.makedirs(self.checkpoint_dir, exist_ok=True)

    @staticmethod
    def load_class(module_name: str, class_name: str) -> Any:
        module = importlib.import_module(module_name)
        return getattr(module, class_name)