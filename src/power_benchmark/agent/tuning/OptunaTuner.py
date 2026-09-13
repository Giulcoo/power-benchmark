import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Literal
from typing_extensions import override

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from power_benchmark.agent.tuning.BaseTuner import BaseTuner


class OptunaTuner(BaseTuner, ABC):
    """
    Abstract base class for Optuna-based hyperparameter tuning of RL agents.

    Subclasses must implement:
        - `_suggest_hyperparams`: Define the search space.
        - `_objective`: Train an agent and return the objective value.

    Optionally override:
        - `_extract_configs`: Convert flat Optuna params to structured configs.
        - `_create_sampler` / `_create_pruner`: Customize Optuna components.
    """

    def __init__(
        self,
        scenario_name: str,
        env_config: Dict[str, Any],
        results_dir: str,
        cpu_count: int,
        gpu_count: int = 0,
        n_trials: int = 50,
        study_name_prefix: str = "rl_hparam_tuning",
        direction: Literal["maximize", "minimize"] = "maximize",
        storage: Optional[str] = None,
        seed: int = 42,
        n_startup_trials: int = 10,
        pruner_warmup_steps: int = 20,
    ):
        """
        Args:
            scenario_name: Name of the scenario being tuned.
            results_dir: Path to save/load best hyperparameters (JSON).
            n_trials: Number of Optuna trials.
            study_name_prefix: Prefix of optuna study name (full name: prefix_scenario_name).
            direction: 'maximize' or 'minimize'.
            storage: Optuna storage URL (e.g. 'sqlite:///study.db'). None = in-memory.
            seed: Random seed for the sampler.
            n_startup_trials: Random trials before TPE kicks in.
            pruner_warmup_steps: Steps before pruning is allowed.
        """
        super().__init__(scenario_name, env_config, results_dir, cpu_count, gpu_count)
        self.n_trials = n_trials
        self.study_name = study_name_prefix + "_" + scenario_name
        self.direction = direction
        self.storage = storage
        self.seed = seed
        self.n_startup_trials = n_startup_trials
        self.pruner_warmup_steps = pruner_warmup_steps

    @override
    def tune(self) -> Dict[str, Any]:
        """
        Main entry point.

        Returns cached hyperparameters if `results_path` exists and is valid.
        Otherwise runs the Optuna study, saves, and returns the best config.

        Returns:
            Dict of best hyperparameter configs.
        """
        existing = self._load_existing()
        if existing is not None:
            self.logger.info(f"Loaded existing hyperparameters from {self.results_file}")
            return existing

        self.logger.info("No existing hyperparameters found. Starting Optuna tuning...")
        study = self._run_study()

        best_configs = self._extract_configs(study.best_params)
        self._save_study(best_configs, study)
        self.logger.info(
            f"Tuning complete. Best value: {study.best_value:.4f} "
            f"(trial {study.best_trial.number})"
        )
        return best_configs

    @abstractmethod
    def _suggest_hyperparams(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Define the search space and return suggested hyperparameters.

        Args:
            trial: Optuna trial object.

        Returns:
            Dict of hyperparameters for this trial.
        """
        ...

    @abstractmethod
    def _objective(self, trial: optuna.Trial) -> float:
        """
        Train an agent with the suggested hyperparameters and return the objective value.

        Implementations should:
            1. Call `self._suggest_hyperparams(trial)` to get params.
            2. Build and train the agent.
            3. Call `trial.report(value, step)` for intermediate results.
            4. Check `trial.should_prune()` and raise `optuna.TrialPruned()`.
            5. Return the final objective value (e.g. best mean reward).

        Args:
            trial: Optuna trial object.

        Returns:
            Objective value (reward to maximize, or loss to minimize).
        """
        ...

    def _extract_configs(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert flat Optuna best_params into a structured config dict.

        Override for custom structure. Default returns params as-is.

        Args:
            params: Flat dict from `study.best_params`.

        Returns:
            Structured config dict.
        """
        return {"hyperparams": params}

    def _create_sampler(self) -> optuna.samplers.BaseSampler:
        """Create the Optuna sampler. Override for custom samplers."""
        return TPESampler(seed=self.seed, n_startup_trials=self.n_startup_trials)

    def _create_pruner(self) -> optuna.pruners.BasePruner:
        """Create the Optuna pruner. Override for custom pruners."""
        return MedianPruner(
            n_startup_trials=max(3, self.n_startup_trials // 2),
            n_warmup_steps=self.pruner_warmup_steps,
            interval_steps=5,
        )

    def _run_study(self) -> optuna.Study:
        """Create and execute the Optuna study."""
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            direction=self.direction,
            sampler=self._create_sampler(),
            pruner=self._create_pruner(),
            load_if_exists=True,
        )

        study.optimize(
            self._objective,
            n_trials=self.n_trials,
            show_progress_bar=True,
        )

        return study

    def _save_study(self, configs: Dict[str, Any], study: optuna.Study) -> None:
        """Save best configs and metadata."""
        super()._save_results({
            **configs,
            "_meta": {
                "best_value": study.best_value,
                "best_trial_number": study.best_trial.number,
                "n_trials_completed": len(study.trials),
                "study_name": self.study_name,
            },
        })