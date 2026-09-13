from abc import abstractmethod, ABC
from typing import Any, Dict, Optional
from gymnasium import spaces
import os

from pandapower_env.environments.simulation_env import PPTopoGym
import logging

class BenchAgent(ABC):
    """ Standard class needed for the input agents of the PowerBenchmarker. Any agent that
     should be used in the benchmark has to implement this class. Or wrap an existing agent with this class. """
    def __init__(self,
                 scenario_name: str,
                 env_config: Dict[str, Any],
                 checkpoint_dir: str,
                 cpu_count: int = 1,
                 gpu_count: int = 0,
                 patience: Optional[int] = None,  # After how many iterations of no improvement the agent should stop the training
                 ):
        self.logger = logging.getLogger("Benchmarker")
        self.scenario_name = scenario_name
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.agent_config = env_config
        self.env = PPTopoGym(self.agent_config)
        self.action_space = spaces.Discrete(len(self.agent_config["action_space"]))

        self.cpu_count = cpu_count
        self.gpu_count = gpu_count

        self.patience = patience

    @abstractmethod
    def build_agent(self) -> Any:
        """ Initialize the agent for training. """
        pass

    @abstractmethod
    def load_agent(self):
        """ Load the agent from checkpoint for testing/evaluation. """
        pass

    @abstractmethod
    def train(self) -> Any:
        """Train the agent.
        Returns:
            Path of last/best/any checkpoint.
        """
        pass

    @abstractmethod
    def act(self, observation, info) -> int:
        """
        Chooses one action for a single observation.

        Args:
            observation: Single observation
            info: Additional info from environment
        Returns:
            Selected action as integer
        """
        pass