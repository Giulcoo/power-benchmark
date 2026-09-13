import logging
from collections.abc import Sequence
from typing import Dict, List, Union, Optional, Callable

import numpy as np
import pandas as pd

from pandapower_env.metrics.evaluation_metrics import MetricRegistry
from pandapower_env.metrics.metric_utils import EvaluateMetrics
import power_benchmark.eval.BenchMetrics # noqa: F401 – Needed for loading metrics inside MetricRegistry
from power_benchmark.constants.types import VALUE_TYPE, EVAL_CRITERION, VALUE_LITERAL, WEIGHT_TYPE
from power_benchmark.eval.RunMetrics import RunMetrics
from power_benchmark.eval.Scoring import ScoreEntry, MinMaxFunction
from power_benchmark.results.data.MetricData import MetricResult, MetricCategory
from power_benchmark.results.data.ScenarioData import SingleResult
from power_benchmark.utils.net_utils import max_possible_regions_lower_bound
from power_benchmark.configs.range_config import Index

logger = logging.getLogger("Benchmarker")


class EvaluateBench:
    """"
    Evaluates all the metrics gathered by BenchMetrics, AllMetrics, and other metric containers.
    Applies weights to different metrics and computes overall scores for various categories such as operational performance, security, costs, and performance.
    """
    def __init__(self,
        index: Index,
        amount_of_days: int,
        run_metrics: Dict[str, List[float]],
        weights: Dict[str, Dict[str, float]],
        env_config: dict,
        actions: Sequence[int | np.integer],
        dn_metrics: Dict[str, List[float]] = {}
    ):
        # TODO: Evaluation from run data fix no actions in DN
        self.index = index
        self.amount_of_days = amount_of_days
        self.run_metrics = run_metrics
        self.weights = weights # Configuration for weighting different metrics in evaluation
        self.episode_length = env_config["episode_length"]
        self.net = env_config["net"]
        self.nminus1 = env_config.get("nminus1", False)

        self.eval = EvaluateMetrics(env_config, MetricRegistry.METRICS)
        self.df_steps, self.df_stats = pd.DataFrame(), pd.DataFrame()
        self.actions = actions

        # Do Nothing metrics
        if len(dn_metrics) > 0:
            statistics = RunMetrics.calc_statistics(dn_metrics)
            self.dn_average_runtime = statistics["runtimes"]["mean"]
            self.dn_max_runtime = statistics["runtimes"]["max"]
            self.dn_average_cpu_usage = statistics["cpu_usages"]["mean"]
            self.dn_max_cpu_usage = statistics["cpu_usages"]["max"]
            self.dn_average_memory_usage = statistics["memory_usages"]["mean"]
            self.dn_max_memory_usage = statistics["memory_usages"]["max"]

            statistics = RunMetrics.calc_statistics(run_metrics)
            self.agent_mean_runtime = statistics["runtimes"]["mean"]
            self.agent_max_runtime = statistics["runtimes"]["max"]
            self.agent_mean_cpu_usage = statistics["cpu_usages"]["mean"]
            self.agent_max_cpu_usage = statistics["cpu_usages"]["max"]
            self.agent_mean_memory_usage = statistics["memory_usages"]["mean"]
            self.agent_max_memory_usage = statistics["memory_usages"]["max"]

            self.check_performance = True
        else:
            self.check_performance = False

        # Calculate evaluations
        self.scores: Dict[str, List[ScoreEntry]] = {}

        # Net info
        self.num_lines = len(self.net.line)
        self.num_switches = len(self.net.switch)
        self.num_trafo = len(self.net.trafo)
        self.maximum_regions = max_possible_regions_lower_bound(self.net)

        # Needed fields
        self.total_iterations = 0
        self.total_score = 0
        self.category_scores: Dict[str, float] = {}

    def run(self) -> SingleResult:
        self.df_steps, self.df_stats = self.eval.evaluate(self.actions, start_index=self.index.index)

        logger.debug("Evaluation Results:\n", self.df_steps, "\n", self.df_stats)
        self.total_iterations = len(self.actions)

        if self.total_iterations == 0:
            logger.warning("No actions were taken during the benchmark run. Security evaluation cannot be performed.")

        return SingleResult(
            index= self.index.index,
            amount_of_days= self.amount_of_days,
            metric_categories= [
                self._overload_mitigation_eval(),
                self._voltage_violation_mitigation_eval(),
                self._survival_eval(),
                self._cost_eval(),
                self._computational_performance_eval()
            ]
        )

    def _overload_mitigation_eval(self) -> MetricCategory:
        cat = "overload_mitigation"

        def build_metrics() -> list:
            metrics = [
                self._gather_entry(cat, "timesteps_without_overload", 0, self.total_iterations, last_value=True),
                self._gather_entry(cat, "total_lines_overloaded", self.num_lines * self.total_iterations, 0, last_value=True),
                # self._gather_entry(cat, "total_regions_overloaded", self.maximum_regions * self.total_iterations, 0, last_value=True), # Future work
                self._gather_entry(cat, "max_line_loading_nminus0", 100, 0, eval_criterion=["mean", "worst"], rename="max_line_loading_timestep_nminus0"),
                self._gather_entry(cat, "max_line_loading_mean_nminus0", 100, 0, eval_criterion=["mean", "worst"], rename="mean_line_loading_timestep_nminus0"),
                self._gather_entry(cat, "max_lines_overloaded_nminus0", self.num_lines, 0, eval_criterion=["mean", "worst"], rename="total_lines_overloaded_timestep_nminus0"),
                # self._gather_entry(cat, "regions_timestep_overloaded", self.maximum_regions, 0, eval_criterion=["mean", "worst"]),
            ]

            if self.nminus1:
                metrics += [
                    self._gather_entry(cat, "max_line_loading_nminus1", 100, 0, eval_criterion=["mean", "worst"]),
                    self._gather_entry(cat, "max_lines_overloaded_nminus1", self.num_lines, 0, eval_criterion=["mean", "worst"], rename="total_lines_overloaded_timestep_nminus1"),
                ]
            return metrics

        return self._build_category(cat, build_metrics)

    def _voltage_violation_mitigation_eval(self) -> MetricCategory:
        cat = "voltage_violation_mitigation"
        return self._build_category(cat, lambda: [
            self._gather_entry(cat, "total_timesteps_with_voltage_violation", self.total_iterations, 0,
                               last_value=True),
            self._gather_entry(cat, "max_voltage_timestep_deviation", 0.05, 0, eval_criterion=["mean", "worst"]),
            self._gather_entry(cat, "mean_voltage_timestep_deviation", 0.05, 0, eval_criterion=["mean", "worst"]),
            self._gather_entry(cat, "percent_bus_voltage_violation", 100, 0, eval_criterion=["mean", "worst"]),
        ])

    def _survival_eval(self) -> MetricCategory:
        cat = "survival"
        return self._build_category(cat, lambda: [
            self._gather_entry(cat, "survived_iterations", 0, self.episode_length, values=self.total_iterations),
        ])


    def _cost_eval(self) -> MetricCategory:
        cat = "costs"

        def build_metrics() -> list:
            metrics = [
                # self._gather_entry(cat, "timesteps_in_start_topology", 0, self.total_iterations, last_value=True),

                self._gather_entry(cat, "timesteps_without_actions", 0, self.total_iterations, last_value=True),
                self._gather_entry(cat, "substation_changes", self.total_iterations, 0, last_value=True),
                # self._gather_entry(cat, "lines_in_service_changes", self.total_iterations * self.num_lines, 0, last_value=True), # Future work, when agents do these actions
                # self._gather_entry(cat, "closed_switches_changes", self.total_iterations, 0, last_value=True),
                # self._gather_entry(cat, "total_max_used_line", self.total_iterations, 0, last_value=True),
                # self._gather_entry(cat, "total_max_used_switch", self.total_iterations, 0, last_value=True),

                self._gather_entry(cat, "active_power_loss_timestep_percent", 100, 0, eval_criterion=["mean", "worst"]),
                self._gather_entry(cat, "timesteps_with_load_shedding", self.total_iterations, 0, last_value=True),
                self._gather_entry(cat, "load_shedding_timestep_percent", 100, 0, eval_criterion=["mean", "worst"]),
            ]

            # if self.num_trafo > 0: # Future work, when agents do these actions
            #     metrics.extend([
            #         self._gather_entry(cat, "trafo_tap_changes", self.total_iterations, 0, last_value=True),
            #         self._gather_entry(cat, "total_max_used_trafo", self.total_iterations, 0, last_value=True),
            #     ])

            return metrics

        return self._build_category(cat, build_metrics)

    def _computational_performance_eval(self) -> MetricCategory:
        cat = "computational_performance"
        return self._build_category(cat, lambda: [
            self._gather_entry(
                cat, "runtime_timestep",
                {"mean": self.dn_average_runtime * 2, "worst": self.dn_max_runtime * 2},
                {"mean": self.dn_average_runtime, "worst": self.dn_max_runtime},
                values={"mean": self.agent_mean_runtime, "worst": self.agent_max_runtime},
            ),
            self._gather_entry(
                cat, "cpu_usage_timestep",
                {"mean": min(self.dn_average_cpu_usage + 25, 100), "worst": min(self.dn_max_cpu_usage + 25, 100)},
                {"mean": self.dn_average_cpu_usage, "worst": self.dn_max_cpu_usage},
                values={"mean": self.agent_mean_cpu_usage, "worst": self.agent_max_cpu_usage},
            ),
            self._gather_entry(
                cat, "memory_usage_timestep",
                {"mean": min(self.dn_average_memory_usage + 25, 100), "worst": min(self.dn_max_memory_usage + 25, 100)},
                {"mean": self.dn_average_memory_usage, "worst": self.dn_max_memory_usage},
                values={"mean": self.agent_mean_memory_usage, "worst": self.agent_max_memory_usage},
            ),
        ], extra_condition=self.check_performance)

    def _build_category(self, cat: str, metrics_factory: Callable[[], List[Optional[MetricResult]]], extra_condition: bool = True) -> MetricCategory:
        """Builds a MetricCategory with guard logic"""
        enabled = self.total_iterations > 0 and self.weights["total"][cat] > 0 and extra_condition
        return MetricCategory(name=cat, weight=self.weights["total"][cat], metrics=metrics_factory() if enabled else [])

    def _gather_entry(self,
            category: str,
            metric: str,
            wp_val: VALUE_TYPE,
            bp_val: VALUE_TYPE,
            values: Union[VALUE_TYPE, List[float], None] = None,
            eval_criterion: EVAL_CRITERION = "mean",
            ascending_groups: bool = False,
            last_value: bool = False,
            rename: Optional[str] = None) -> Optional[MetricResult]:
        """
          Gathers metric data to create a MetricResult

          :param metric: Name of the metric to gather
          :param wp_val: Worst possible value for this metric (the value that would yield the lowest score)
          :param bp_val: Best possible value for this metric (the value that would yield the highest score)
          :param values: Optional value or list of values to use for this metric
          :param eval_criterion: Criterion or criteria to use for evaluation (e.g., "min", "max", "mean" or a list of these).
          :param ascending_groups: For data that goes like 1,2,3,4,1,2,3,1,2,3,4,5 to make it to a list like [4,3,5]
          :param last_value: If True, only the last value of the metric will be considered for evaluation. Otherwise, all values will be used.
          :return: A MetricResult containing the gathered data and scores for this metric
          """
        weight: WEIGHT_TYPE = self.weights.get(category, {}).get(metric, 1.0)

        if isinstance(weight, float) and weight == 0.0:
            return None
        if isinstance(weight, dict) and all(v == 0.0 for v in weight.values()):
            return None

        if values is None:
            # Gather values if none provided
            values = self.df_steps[metric].tolist()

            if last_value and len(values) > 0:
                values = [values[-1]]

        if ascending_groups:
            if not isinstance(values, list):
                logger.warning(f"Values {values} for metric {category}:{metric} are not a list.")
                return None

            # For data that goes like 1,2,3,4,1,2,3,1,2,3,4,5 to make it to a list like [4,3,5]
            groups = []
            current_group = None
            for i in range(0, len(values)):
                if current_group is None:
                    current_group = values[i]
                elif values[i] < current_group:
                    groups.append(current_group)
                    current_group = values[i]
                else:
                    current_group = values[i]

            values = groups

        score_function: MinMaxFunction | Dict[VALUE_LITERAL, MinMaxFunction]
        if isinstance(wp_val, dict) and isinstance(bp_val, dict):
            score_function = {key: MinMaxFunction(wp_val[key], bp_val[key]) for key in wp_val.keys()}
        elif isinstance(wp_val, dict) or isinstance(bp_val, dict):
            raise ValueError("Both wp_val and bp_val must be of the same type (either both dicts or both floats).")
        else:
            score_function = MinMaxFunction(wp_val, bp_val)

        entry = ScoreEntry(metric, values, score_function)

        return MetricResult(
            name=metric if rename is None else rename,
            values=entry.zip_results(),
            weight=weight,
            eval_criterion=eval_criterion,
            wp_val=wp_val,
            bp_val=bp_val
        )