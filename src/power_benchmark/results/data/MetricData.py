from typing import List, Dict, Any, Tuple, Union, Literal, Optional

import numpy as np
from pydantic import BaseModel, Field
import logging
from power_benchmark.constants.types import VALUE_TYPE, EVAL_CRITERION, VALUE_SCORE_TYPES, VALUE_LITERAL

logger = logging.getLogger("Benchmarker")

class MetricResult(BaseModel):
    name: str
    values: VALUE_SCORE_TYPES
    weight: VALUE_TYPE
    eval_criterion: EVAL_CRITERION = Field(default="mean")
    wp_val: VALUE_TYPE # Worst possible value (the value that would yield the lowest score)
    bp_val: VALUE_TYPE # Best possible value (the value that would yield the highest score)

    def model_post_init(self, context: Any, /) -> None:
        # Remove any NaN values from the values list
        if isinstance(self.values, list):
            self.values = [v for v in self.values if not np.isnan(v[0]) and not np.isnan(v[1])]
        elif isinstance(self.values, dict):
            self.values = {key: value for key, value in self.values.items() if not np.isnan(value[0]) and not np.isnan(value[1])}
        elif isinstance(self.values, tuple):
            if np.isnan(self.values[0]) or np.isnan(self.values[1]):
                raise ValueError("Values cannot contain NaN. Got NaN in values tuple.")
        else:
            raise ValueError(f"Invalid type in {self.name} for values. Expected list, dict or tuple. Got {type(self.values)} instead.")

    @property
    def worst_score(self) -> Tuple[float, float] | None:
        """ Return the (value, score) tuple corresponding to the minimum score."""
        if isinstance(self.values, dict):
            if "worst" not in self.values:
                raise ValueError(f"Values for {self.name} must have 'max' key. Instead got {self.values.keys()} instead.")

            return self.values["worst"]
        elif isinstance(self.values, tuple):
            if len(self.values) != 2:
                raise ValueError(f"Values for {self.name} should have two values as a tuple. Instead got {len(self.values)} instead.")

            return self.values
        elif isinstance(self.values, list):
            if len(self.values) == 0:
                logger.warning(f"No values for {self.name}")
                return None
            return min(self.values, key=lambda x: x[1])
        else:
            raise ValueError(f"Invalid type in {self.name} for values. Expected list, dict or tuple. Got {type(self.values)} instead.")

    @property
    def mean_score(self) -> Tuple[float, float] | None:
        """ Return the mean of the scores."""
        if isinstance(self.values, dict):
            if "mean" not in self.values:
                raise ValueError(f"Values for {self.name} must have 'mean' key. Instead got {self.values.keys()} instead.")

            return self.values["mean"]
        elif isinstance(self.values, tuple):
            if len(self.values) != 2:
                raise ValueError(f"Values for {self.name} should have two values as a tuple. Instead got {len(self.values)} instead.")

            return self.values
        elif isinstance(self.values, list):
            if len(self.values) == 0:
                logger.warning(f"No values for {self.name}")
                return None

            return float(np.mean([v[0] for v in self.values])), float(np.mean([v[1] for v in self.values]))
        else:
            raise ValueError(f"Invalid type in {self.name} for values. Expected list, dict or tuple. Got {type(self.values)} instead.")

    def str_to_score(self, criterion: VALUE_LITERAL) -> float | None:
        if criterion == "worst":
            return self.worst_score[1] if self.worst_score is not None else None
        elif criterion == "mean":
            return self.mean_score[1] if self.mean_score is not None else None
        else:
            raise ValueError(f"Invalid eval_criterion: {criterion} (only 'worst' or 'mean' allowed).")

    def str_to_value(self, criterion: VALUE_LITERAL) -> float | None:
        if criterion == "worst":
            return self.worst_score[0] if self.worst_score is not None else None
        elif criterion == "mean":
            return self.mean_score[0] if self.mean_score is not None else None
        else:
            raise ValueError(f"Invalid eval_criterion: {criterion} (only 'worst' or 'mean' allowed)")

    @property
    def score(self) -> List[float]:
        if isinstance(self.eval_criterion, str):
            score = self.str_to_score(self.eval_criterion)
            return score if score is not None else []
        else:
            score = [self.str_to_score(criterion) for criterion in self.eval_criterion]
            return [s for s in score if s is not None]

    def score_dict(self) -> Dict[VALUE_LITERAL, Optional[float]]:
        if isinstance(self.eval_criterion, str):
            return {self.eval_criterion: self.str_to_score(self.eval_criterion)}
        else:
            return {criterion: self.str_to_score(criterion) for criterion in self.eval_criterion}

    def score_weight(self) -> List[Tuple[Optional[float], float]]:
        result = []
        for criteria, score in self.score_dict().items():
            result.append((score, self.weight if isinstance(self.weight, float) else self.weight[criteria]))
        return result

    def value_dict(self) -> Dict[VALUE_LITERAL, Optional[float]]:
        if isinstance(self.eval_criterion, str):
            return {self.eval_criterion: self.str_to_value(self.eval_criterion)}
        else:
            return {criterion: self.str_to_value(criterion) for criterion in self.eval_criterion}

    def to_dict(self) -> Dict[str, Any] | None:
        score_dict = self.score_dict()
        first_score = next(iter(score_dict.values()))

        value_dict = self.value_dict()
        first_value = next(iter(value_dict.values()))

        if all(v is None for v in score_dict.values()) or all(v is None for v in value_dict.values()):
            return None

        return {
            "values": self.values,
            "resulting_scores": first_score if len(score_dict) == 1 else score_dict,
            "resulting_values": first_value if len(value_dict) == 1 else value_dict,
            "scoring_range": {"wp_val": self.wp_val, "bp_val": self.bp_val},
        }

class MetricCategory(BaseModel):
    name: str
    metrics: List[Optional[MetricResult]]
    weight: float

    def model_post_init(self, context: Any, /) -> None:
        self.metrics: List[MetricResult] = [m for m in self.metrics if m is not None]

    @property
    def category_score(self) -> float:
        if self.weight == 0:
            return 0.0

        scores: List[Tuple[float, float]] = [] # List of (score, weight) tuples
        for metric in self.metrics:
            scores += [(score, weight) for (score, weight) in metric.score_weight() if score is not None]

        weight_sum =  np.sum([weight for _, weight in scores])

        if weight_sum == 0:
            return 0.0

        result = float(
            np.sum([score * weight for score, weight in scores]) / weight_sum
        )

        return 0.0 if np.isnan(result) else result

    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any], weights: Dict[str, Any]) -> 'MetricCategory':
        del data["category_score"]

        metrics = [
            MetricResult(
                name=metric_name,
                values=metric_data['values'],
                weight=weights.get(name, {}).get(metric_name, 1.0),
                wp_val=metric_data["scoring_range"]["wp_val"],
                bp_val=metric_data["scoring_range"]["bp_val"],
                eval_criterion=metric_data["resulting_scores"].keys()
                    if isinstance(metric_data["resulting_scores"], dict) else "mean",
            )
            for metric_name, metric_data in data.items()
        ]

        return cls(
            name=name,
            metrics=metrics,
            weight=weights.get("total", {}).get(name, 1.0)
        )

    def get_weights(self) -> Dict[str, Any]:
        return {
            metric.name: metric.weight for metric in self.metrics
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            **{ metric.name: metric.to_dict() for metric in self.metrics if metric.to_dict() is not None},
            "category_score": self.category_score
        }