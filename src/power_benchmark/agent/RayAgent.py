import hashlib
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import time
import ray
from ray.rllib.algorithms import Algorithm

from power_benchmark.agent.BenchAgent import BenchAgent


class RayAgent(BenchAgent):
    """ Standard class needed for the input agents of the PowerBenchmarker. Any agent that
     should be used in the benchmark has to implement this class. Or wrap an existing agent with this class. """
    def __init__(self,
                 scenario_name: str,
                 env_config: Dict[str, Any],
                 checkpoint_dir: str,
                 cpu_count: int = 1,
                 gpu_count: int = 0,
                 patience: Optional[int] = None,
                 resume_from_checkpoint: bool = True,
                 training_iters: int = 1000,
                 checkpoint_every_iters: int = 100,
                 explore_during_act:  bool = False,
                 ):
        super().__init__(
            scenario_name=scenario_name,
            env_config=env_config,
            checkpoint_dir=checkpoint_dir,
            cpu_count=cpu_count,
            gpu_count=gpu_count,
            patience=patience
        )
        self.explore_during_act = explore_during_act

        self.training_iters = training_iters
        self.checkpoint_every_iters = checkpoint_every_iters

        ray.shutdown()

        short_id = hashlib.md5(scenario_name.encode()).hexdigest()[:8] # Short id because scenario name can be too long and crashes init
        ray.init(
            ignore_reinit_error=True,
            num_cpus=self.cpu_count,
            num_gpus=self.gpu_count,
            namespace=self.scenario_name,
            _temp_dir = f"/tmp/ray_{short_id}",
        )

        self.algo = self.build_agent()

        if resume_from_checkpoint and self.checkpoint_exists(checkpoint_dir):
            self.algo.restore(checkpoint_dir)

    @abstractmethod
    def build_agent(self) -> Algorithm:
        """ Initialize the agent for training. """
        pass

    def load_agent(self):
        """ Load the agent from checkpoint for testing/evaluation. """
        self.algo.restore(self.checkpoint_dir)

    def train(self) -> str:
        """Train the agent.
        Returns:
            Path of last/best/any checkpoint.
        """
        checkpoint_path = ""

        # Needed for early stopping
        no_improve_count = 0
        best_reward = float("-inf")

        start_iter = self.algo.iteration
        max_iter = start_iter + self.training_iters

        start_time = time.time()
        for i in range(start_iter, max_iter):
            iter_time = time.time()
            result = self.algo.train()

            total_elapsed = time.time() - start_time
            iter_elapsed = time.time() - iter_time
            iters_done = i - start_iter
            self.log_train_results(i, max_iter, iters_done, total_elapsed, iter_elapsed, result)

            if i % self.checkpoint_every_iters == 0:
                checkpoint_path = self.algo.save(self.checkpoint_dir)

            if self.patience is not None:
                env_stats = result.get("env_runners", {})
                reward = env_stats.get("episode_reward_mean", env_stats.get("episode_return_mean", 0))

                if reward > best_reward:
                    best_reward = reward
                    no_improve_count = 0
                    best_checkpoint = self.algo.save(self.checkpoint_dir)
                else:
                    no_improve_count += 1

                if no_improve_count >= self.patience:
                    self.logger.info(f"Early stopping at iter {i}")
                    break

        return checkpoint_path if checkpoint_path != "" else self.algo.save(self.checkpoint_dir)

    def log_train_results(self, iteration: int, max_iter: int, iter_done: int, total_elapsed_time: float, iteration_elapsed_time: float, result: Dict):
        """ Logs result of training iteration. """
        if iter_done > 0:
            avg_time_per_iter = total_elapsed_time / iter_done
            remaining_iters = max_iter - iteration
            remaining_seconds = avg_time_per_iter * remaining_iters

            days, remainder = divmod(int(remaining_seconds), 86400)
            hours, remainder = divmod(remainder, 3600)
            minutes, _ = divmod(remainder, 60)

            if days > 0:
                eta_str = f"{days}d {hours}h {minutes}m"
            elif hours > 0:
                eta_str = f"{hours}h {minutes}m"
            else:
                eta_str = f"{minutes}m"
        else:
            eta_str = "calculating..."

        self.logger.info(
            f"Training Iteration {iteration}/{max_iter} "
            f"[Iteration Elapsed Time: {iteration_elapsed_time:.1f}s |"
            f"Estimated Time Left: {eta_str}]"
        )

    def act(self, observation, info) -> int:
        """
        Chooses one action for a single observation.

        Args:
            observation: Single observation
            info: Additional info from environment
        Returns:
            Selected action as integer
        """
        action = self.algo.compute_single_action(
            observation,
            info=info,
            explore=self.explore_during_act
        )
        return int(action)

    @staticmethod
    def checkpoint_exists(checkpoint_dir: str) -> bool:
        path = Path(checkpoint_dir)
        return path.exists() and path.is_dir() and any(path.iterdir())