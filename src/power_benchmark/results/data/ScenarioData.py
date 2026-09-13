import json
import re
from collections import defaultdict
from functools import cached_property
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field

from power_benchmark.results.data.MetricData import MetricCategory
from power_benchmark.utils.json_utils import save_json, load_json

import logging
logger = logging.getLogger("Benchmarker")

class SingleResult(BaseModel):
    index: int
    amount_of_days: int
    failed: bool = Field(default=False)
    metric_categories: List[MetricCategory]

    @property
    def total_score(self) -> float:
        if self.failed:
            return 0.0

        return float(
            np.sum([category.category_score * category.weight for category in self.metric_categories])
            / np.sum([category.weight for category in self.metric_categories])
        )

    @property
    def category_scores(self) -> Dict[str, float]:
        return {
            category.name: category.category_score
            for category in self.metric_categories
        }

    @property
    def weights(self) -> Dict[str, Any]:
        weights = {}
        for category in self.metric_categories:
            weights[category.name] = category.get_weights()
        return weights

    @classmethod
    def from_dict(cls, data: Dict[str, Any], weight: Dict[str, Any]) -> 'SingleResult':
        index = data["index"]
        amount_of_days = data["amount_of_days"]
        del data["total_score"]
        del data["index"]
        del data["amount_of_days"]

        metric_categories = [
            MetricCategory.from_dict(category_name, category_data, weight)
            for category_name, category_data in data.items()
        ]

        return cls(
            index=index,
            amount_of_days=amount_of_days,
            metric_categories=metric_categories
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            **{category.name: category.to_dict() for category in self.metric_categories},
            "index": self.index,
            "amount_of_days": self.amount_of_days,
            "total_score": self.total_score
        }


class ScenarioResult(BaseModel):
    network_category: str
    name: str
    scenario_config: str
    file: Path
    results: List[SingleResult]

    @cached_property
    def non_failed_results(self) -> List[SingleResult]:
        return [result for result in self.results if not result.failed]

    @cached_property
    def failed_results(self) -> Dict[str, Union[str, List[int]]]:
        return {
            "name": self.name,
            "indexes": [result.index for result in self.results if result.failed]
        }

    @cached_property
    def metrics_results(self) -> Dict[str, List[Tuple[float, float]]]:
        metrics_results: Dict[str, List[Tuple[float, float]]] = {}
        for result in self.non_failed_results:
            for category in result.metric_categories:
                for metric in category.metrics:
                    value_dict = metric.value_dict()
                    score_dict = metric.score_dict()

                    if len(score_dict) == 1:
                        if metric.name not in metrics_results:
                            metrics_results[metric.name] = []

                        value = next(iter(value_dict.values()))
                        score = next(iter(score_dict.values()))

                        if value is not None and score is not None:
                            metrics_results[metric.name].append((value, score))
                    else:
                        for criterion, score in score_dict.items():
                            metric_name = f"{metric.name}[{criterion}]"
                            if metric_name not in metrics_results:
                                metrics_results[metric_name] = []

                            value = value_dict[criterion]

                            if value is not None and score is not None:
                                metrics_results[metric_name].append((value, score))
        return metrics_results

    @property
    def metrics_average(self) -> Dict[str, Dict[str, float]]:
        return {
            name: {
                "score": float(np.mean([score for _, score in results])),
                "value": float(np.mean([value for value, _ in results])),
            }
            for name, results in self.metrics_results.items()
        }

    @property
    def metrics_gathered(self) -> Dict[str, int]:
        gathered = {
            name: len(values)
            for name, values in self.metrics_results.items()
        }

        # Check if all counts are the same
        if len(set(gathered.values())) == 1:
            return {
                "amount_of_metrics": len(set(gathered.keys())),
                "total_amount_metric_gathered": sum(gathered.values()),
            }
        return gathered

    @property
    def average_score(self) -> float:
        return float(
            np.sum([result.total_score for result in self.non_failed_results]) / len(self.non_failed_results)
        )

    @property
    def average_category_scores(self) -> Dict[str, float]:
        category_scores_gathered: Dict[str, List[float]] = defaultdict(list)

        for result in self.non_failed_results:
            for key, score in result.category_scores.items():
                category_scores_gathered[key].append(score)

        return {
            key: float(np.mean(scores))
            for key, scores in category_scores_gathered.items()
        }

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "scenario_score": self.average_score,
            "category_scores": self.average_category_scores,
            "metrics_average": self.metrics_average,
            "metrics_gathered": self.metrics_gathered,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], path: Path, weights: Dict[str, Any]) -> 'ScenarioResult':
        results = [
            SingleResult.from_dict(result_data, weights)
            for result_data in data["results"]
        ]

        return cls(
            scenario_config=data["scenario_config"],
            file=path,
            network_category=data["network_category"],
            name=path.stem,
            results=results
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "network_category": self.network_category,
            "scenario_config": self.scenario_config,
            "results": [result.to_dict() for result in self.non_failed_results]
        }

    @classmethod
    def load_json(cls, path: str | Path, weights: Dict[str, Any]) -> 'ScenarioResult':
        return cls.from_dict(load_json(path), path, weights)

    def save_json(self) -> None:
        save_json(self.to_dict(), self.file, max_fields=6, max_items=-1, max_outer_items=-1)