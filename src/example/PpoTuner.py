import tempfile
from typing import Any, Dict, Optional
from typing_extensions import override

import optuna
import ray

from example.PpoAgent import PpoAgent
from power_benchmark.agent.tuning.OptunaTuner import OptunaTuner


class PpoTuner(OptunaTuner):
    """Optuna hyperparameter tuner for the GNN-based PPO agent."""

    def __init__(
        self,
        scenario_name: str,
        env_config: Dict[str, Any],
        results_dir: str,
        cpu_count: int,
        gpu_count: int = 0,
        n_trials: int = 50,
        training_iters_per_trial: int = 200,
        patience_per_trial: int = 30,
        study_name_prefix: str = "gnn_ppo_tuning",
        storage: Optional[str] = None,
        seed: int = 42,
        n_startup_trials: int = 10,
        pruner_warmup_steps: int = 20,
    ):
        """
        Args:
            env_config: Environment config passed to PPTopoGym.
            training_iters_per_trial: Max training iterations per trial.
            cpu_count: CPUs available per trial.
            gpu_count: GPUs available per trial.
            patience_per_trial: Early stopping patience within each trial.
        """
        super().__init__(
            scenario_name=scenario_name,
            env_config=env_config,
            results_dir=results_dir,
            cpu_count=cpu_count,
            gpu_count=gpu_count,
            n_trials=n_trials,
            study_name_prefix=study_name_prefix,
            direction="maximize",
            storage=storage,
            seed=seed,
            n_startup_trials=n_startup_trials,
            pruner_warmup_steps=pruner_warmup_steps,
        )
        self.training_iters_per_trial = training_iters_per_trial
        self.patience_per_trial = patience_per_trial

    @override
    def _suggest_hyperparams(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest GNN model and PPO training hyperparameters."""
        return {
            "model_config": self._suggest_model_config(trial),
            "ppo_config": self._suggest_ppo_config(trial),
        }

    @override
    def _objective(self, trial: optuna.Trial) -> float:
        """Train a PpoAgentTunable and return best mean reward."""
        hparams = self._suggest_hyperparams(trial)
        checkpoint_dir = tempfile.mkdtemp(prefix=f"optuna_trial_{trial.number}_")
        agent = None

        try:
            agent = PpoAgent(
                scenario_name=self.scenario_name,
                env_config=self.agent_config,
                checkpoint_dir=checkpoint_dir,
                training_iters=self.training_iters_per_trial,
                checkpoint_every_iters=max(1, self.training_iters_per_trial // 4),
                explore_during_act=False,
                cpu_count=self.cpu_count,
                gpu_count=self.gpu_count,
                patience=self.patience_per_trial,
                model_config=hparams["model_config"],
                ppo_config=hparams["ppo_config"],
            )

            best_reward = float("-inf")

            for i in range(self.training_iters_per_trial):
                result = agent.algo.train()

                reward_mean = self._extract_reward(result)
                best_reward = max(best_reward, reward_mean)

                trial.report(reward_mean, i)
                if trial.should_prune():
                    self.logger.info(f"Trial {trial.number} pruned at iter {i}")
                    raise optuna.TrialPruned()

            return best_reward

        except optuna.TrialPruned:
            raise
        except Exception as e:
            self.logger.error(f"Trial {trial.number} failed: {e}")
            raise

        finally:
            if agent is not None and hasattr(agent, "algo"):
                agent.algo.stop()
            ray.shutdown()

    @override
    def _extract_configs(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Convert flat Optuna params into structured model_config and ppo_config."""
        model_config = {
            "gine_hidden_dims": [params["gine_hidden_dim"]] * params["num_gine_layers"],
            "gine_mlp_layers": params["gine_mlp_layers"],
            "gine_mlp_activation": params["gine_mlp_activation"],
            "pooling": params["pooling"],
            "fc_hidden_dims": [params["fc_hidden_dim"]] * params["num_fc_layers"],
            "fc_activation": params["fc_activation"],
            "norm": params["norm"],
            "dropout": params["dropout"],
        }

        ppo_config = {
            "train_batch_size": params["train_batch_size"],
            "minibatch_size": params["minibatch_size"],
            "num_epochs": params["num_epochs"],
            "gamma": params["gamma"],
            "lr": params["lr"],
            "entropy_coeff": params["entropy_coeff"],
            "clip_param": params["clip_param"],
            "vf_clip_param": params["vf_clip_param"],
            "lambda_": params["lambda_"],
            "num_envs_per_env_runner": params["num_envs_per_env_runner"],
        }

        return {"model_config": model_config, "ppo_config": ppo_config}

    @override
    def _validate_loaded(self, data: Dict[str, Any]) -> bool:
        """Check that loaded JSON has both config sections."""
        return "model_config" in data and "ppo_config" in data

    @staticmethod
    def _suggest_model_config(trial: optuna.Trial) -> Dict[str, Any]:
        """Search space for GNN architecture."""
        num_gine_layers = trial.suggest_int("num_gine_layers", 2, 5)
        gine_hidden_dim = trial.suggest_categorical("gine_hidden_dim", [128, 256, 512])

        num_fc_layers = trial.suggest_int("num_fc_layers", 1, 3)
        fc_hidden_dim = trial.suggest_categorical("fc_hidden_dim", [128, 256, 512])

        return {
            "gine_hidden_dims": [gine_hidden_dim] * num_gine_layers,
            "gine_mlp_layers": trial.suggest_int("gine_mlp_layers", 2, 4),
            "gine_mlp_activation": trial.suggest_categorical("gine_mlp_activation", ["relu", "elu"]),
            "pooling": trial.suggest_categorical("pooling", ["mean", "max", "add"]),
            "fc_hidden_dims": [fc_hidden_dim] * num_fc_layers,
            "fc_activation": trial.suggest_categorical("fc_activation", ["relu", "elu"]),
            "norm": trial.suggest_categorical("norm", ["graph", "layer", "none"]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.2),
        }

    @staticmethod
    def _suggest_ppo_config(trial: optuna.Trial) -> Dict[str, Any]:
        """Search space for PPO training."""
        train_batch_size = trial.suggest_categorical("train_batch_size", [2048, 4096]) # alternative [2048, 4096, 8192]
        minibatch_choices = [s for s in [256, 512, 1024, 2048] if s <= train_batch_size]
        minibatch_size = trial.suggest_categorical("minibatch_size", minibatch_choices)

        return {
            "train_batch_size": train_batch_size,
            "minibatch_size": minibatch_size,
            "num_epochs": trial.suggest_int("num_epochs", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.95, 0.999, log=True),
            "lr": trial.suggest_float("lr", 1e-5, 5e-4, log=True),
            "entropy_coeff": trial.suggest_float("entropy_coeff", 0.001, 0.05, log=True),
            "clip_param": trial.suggest_float("clip_param", 0.1, 0.4),
            "vf_clip_param": trial.suggest_float("vf_clip_param", 5.0, 50.0),
            "lambda_": trial.suggest_float("lambda_", 0.9, 1.0),
            "num_envs_per_env_runner": trial.suggest_categorical("num_envs_per_env_runner", [4, 8, 16]),
        }

    @staticmethod
    def _extract_reward(result: Dict[str, Any]) -> float:
        """Extract mean episode reward from Ray training result."""
        env_stats = result.get("env_runners", {})
        return env_stats.get(
            "episode_reward_mean",
            env_stats.get("episode_return_mean", 0.0),
        )