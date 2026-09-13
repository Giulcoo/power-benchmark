from pydantic import BaseModel, model_validator
from typing import Dict, Any, List, Union, Literal
from power_benchmark.constants.types import ALLOWED_METRICS, WEIGHT_TYPE, METRICS_ENTRY_WITH_WEIGHTS

class MetricWeightConfig(BaseModel):
    metric: str
    weight: WEIGHT_TYPE

    def is_valid(self, category: str) -> bool:
        if category not in ALLOWED_METRICS:
            raise ValueError(f"Invalid category '{category}'. Allowed categories are: {list(ALLOWED_METRICS.keys())}")

        for allowed_metric in ALLOWED_METRICS[category]:
            if isinstance(allowed_metric, str) and self.metric == allowed_metric and isinstance(self.weight, float):
                return True
            elif isinstance(allowed_metric, dict) and self.metric in allowed_metric:
                allowed_metric_value = allowed_metric[self.metric]

                for key in self.weight.keys():
                    if key not in allowed_metric_value:
                        raise ValueError(f"Invalid weight key '{key}' for metric '{self.metric}' in category '{category}'. Allowed keys for this metric are: {allowed_metric_value}")

                return True
        return False

    @classmethod
    def from_data(cls, metric_name: str, weight: WEIGHT_TYPE) -> 'MetricWeightConfig':
        if isinstance(weight, float):
            return cls(metric=metric_name, weight=weight)
        elif isinstance(weight, dict):
            return cls(metric=metric_name, weight=weight)
        else:
            raise ValueError(f"Invalid weight type for metric '{metric_name}'. Weight must be either a float or a dict with keys 'min', 'max', 'mean' and float values.")

class CategoryWeightConfig(BaseModel):
    category: str
    weights: List[MetricWeightConfig] = []

    @model_validator(mode="after")
    def validate_category(self) -> 'CategoryWeightConfig':
        if self.category not in ALLOWED_METRICS:
            raise ValueError(f"Invalid category '{self.category}'. Allowed categories are: {list(ALLOWED_METRICS.keys())}")

        for metric in self.weights:
            if not metric.is_valid(self.category):
                raise ValueError(f"Invalid metric '{metric.metric}' in category '{self.category}'. Allowed metrics for this category are: {ALLOWED_METRICS[self.category]}")

        return self

    def cumulative_weight(self) -> float:
        total_weight = 0.0
        for metric in self.weights:
            if isinstance(metric.weight, float):
                total_weight += metric.weight
            elif isinstance(metric.weight, dict):
                total_weight += sum(metric.weight.values())
        return total_weight

    def fill_defaults(self, default: float = 1.0, other_categories: List['CategoryWeightConfig'] = []):
        # Fill in default weights for missing metrics in this category.
        if self.category != "total":
            existing_metrics = [metric.metric for metric in self.weights]
            allowed_metrics = ALLOWED_METRICS[self.category]

            for metric in allowed_metrics:
                if isinstance(metric, str) and metric not in existing_metrics:
                    self.weights.append(MetricWeightConfig(metric=metric, weight=default))
                elif isinstance(metric, dict):
                    metric_name = list(metric.keys())[0]
                    if metric_name not in existing_metrics:
                        weights = {key: default for key in metric[metric_name]}
                        self.weights.append(MetricWeightConfig(metric=metric_name, weight=weights))
        else:
            # Cumulative weights
            allowed_metrics = ALLOWED_METRICS["total"]
            existing_metrics = [metric.metric for metric in self.weights]

            for metric in allowed_metrics:
                if metric not in existing_metrics:
                    category = [c for c in other_categories if c.category == metric]

                    if len(category) != 1:
                        raise ValueError(f"Category '{metric}' appeared {len(category)} times instead of just one time in other_categories")

                    self.weights.append(MetricWeightConfig(metric=metric, weight=category[0].cumulative_weight()))

    @classmethod
    def from_data(cls, category: str, data: METRICS_ENTRY_WITH_WEIGHTS) -> 'CategoryWeightConfig':
        return cls(category=category, weights=[MetricWeightConfig.from_data(name, weight) for name, weight in data.items()])

    def to_dict(self) -> METRICS_ENTRY_WITH_WEIGHTS:
        return {metric.metric: metric.weight for metric in self.weights}


class WeightConfig(BaseModel):
    overload_mitigation: Union[METRICS_ENTRY_WITH_WEIGHTS, CategoryWeightConfig] = {}
    voltage_violation_mitigation: Union[METRICS_ENTRY_WITH_WEIGHTS, CategoryWeightConfig] = {}
    survival: Union[METRICS_ENTRY_WITH_WEIGHTS, CategoryWeightConfig] = {}
    costs: Union[METRICS_ENTRY_WITH_WEIGHTS, CategoryWeightConfig] = {}
    computational_performance: Union[METRICS_ENTRY_WITH_WEIGHTS, CategoryWeightConfig] = {}
    total: Union[METRICS_ENTRY_WITH_WEIGHTS, CategoryWeightConfig] = {}

    def model_post_init(self, context: Any, /) -> None:
        self.overload_mitigation: CategoryWeightConfig = CategoryWeightConfig.from_data("overload_mitigation", self.overload_mitigation)
        self.voltage_violation_mitigation: CategoryWeightConfig = CategoryWeightConfig.from_data("voltage_violation_mitigation", self.voltage_violation_mitigation)
        self.survival: CategoryWeightConfig = CategoryWeightConfig.from_data("survival", self.survival)
        self.costs: CategoryWeightConfig = CategoryWeightConfig.from_data("costs", self.costs)
        self.computational_performance: CategoryWeightConfig = CategoryWeightConfig.from_data("computational_performance", self.computational_performance)

        self.overload_mitigation.fill_defaults(0.0 if self.total["overload_mitigation"] == 0.0 else 1.0)
        self.voltage_violation_mitigation.fill_defaults(0.0 if self.total["voltage_violation_mitigation"] == 0.0 else 1.0)
        self.survival.fill_defaults(0.0 if self.total["survival"] == 0.0 else 1.0)
        self.costs.fill_defaults(0.0 if self.total["costs"] == 0.0 else 1.0)
        self.computational_performance.fill_defaults(0.0 if self.total["computational_performance"] == 0.0 else 1.0)

        self.total: CategoryWeightConfig = CategoryWeightConfig.from_data("total", self.total)
        self.total.fill_defaults(other_categories=[
            self.overload_mitigation,
            self.voltage_violation_mitigation,
            self.survival,
            self.costs,
            self.computational_performance
        ])

    def to_dict(self) -> Dict[str, METRICS_ENTRY_WITH_WEIGHTS]:
        return {
            "overload_mitigation": self.overload_mitigation.to_dict(),
            "voltage_violation_mitigation": self.voltage_violation_mitigation.to_dict(),
            "survival": self.survival.to_dict(),
            "costs": self.costs.to_dict(),
            "computational_performance": self.computational_performance.to_dict(),
            "total": self.total.to_dict(),
        }