import json
import textwrap
from collections import defaultdict
from functools import cached_property
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel

from power_benchmark.results.data.ScenarioData import ScenarioResult
from power_benchmark.utils.json_utils import save_json, load_json, json_compact_dump

import logging
logger = logging.getLogger("Benchmarker")

class ResultData(BaseModel):
    name: str
    agent: str
    description: Optional[str] = None
    config_file: str
    scenarios: List[ScenarioResult]
    result_dir: Path
    weights: Dict

    @property
    def average_score(self) -> float:
        return sum(scenario.average_score for scenario in self.scenarios) / len(self.scenarios) \
            if self.scenarios and len(self.scenarios) > 0 else 0

    @property
    def average_category_scores(self) -> Dict[str, float]:
        category_scores_gathered: Dict[str, List[float]] = defaultdict(list)

        for scenario in self.scenarios:
            for key, score in scenario.average_category_scores.items():
                category_scores_gathered[key].append(score)

        return {
            key: float(np.mean(scores))
            for key, scores in category_scores_gathered.items()
        }


    @property
    def metrics_results(self) -> Dict[str, List[Tuple[float, float]]]:
        metrics_results: Dict[str, List[Tuple[float, float]]] = {}
        for scenario in self.scenarios:
            for name, avg in scenario.metrics_average.items():
                avg_score = avg["score"]
                avg_value = avg["value"]

                if name not in metrics_results:
                    metrics_results[name] = []

                metrics_results[name].append((avg_value, avg_score))
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
        gathered: Dict[str, int] = {}

        for scenario in self.scenarios:
            for name, values in scenario.metrics_results.items():
                if name not in gathered:
                    gathered[name] = len(values)
                else:
                    gathered[name] += len(values)

        # Check if all counts are the same
        gathered_values = set(gathered.values())
        if len(gathered_values) == 1:
            amount_of_metrics = set(gathered.keys())
            return {
                "amount_of_metrics": len(amount_of_metrics),
                "amount_of_scenarios": len(self.scenarios),
                "total_amount_metric_gathered": sum(gathered.values())
            }
        return gathered

    @property
    def category_metrics_average(self) -> Dict[str, Dict[str, float]]:
        """category_name -> {metric_name (+[criterion]) -> mean score}"""
        gathered: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        for scenario in self.scenarios:
            for result in scenario.non_failed_results:
                for category in result.metric_categories:
                    for metric in category.metrics:
                        score_dict = metric.score_dict()
                        for criterion, score in score_dict.items():
                            if score is None:
                                continue
                            key = metric.name if len(score_dict) == 1 else f"{metric.name}[{criterion}]"
                            gathered[category.name][key].append(score)
        return {
            cat: {name: float(np.mean(scores)) for name, scores in metrics.items()}
            for cat, metrics in gathered.items()
        }

    @classmethod
    def load_from_dir(cls, result_dir: str | Path, metadata: Optional[Dict] = None) -> 'ResultData':
        result_dir: Path = Path(result_dir)
        metadata_path = result_dir / Path("meta_data.json")

        if metadata_path.exists():
            logger.info(f"Loading metadata from {metadata_path}")
            metadata = load_json(metadata_path)
        elif metadata is None:
            raise ValueError("Metadata must be provided if meta_data.json does not exist in the result directory")

        return cls(
            result_dir=result_dir,
            name=metadata["name"],
            agent=metadata["agent"],
            description=metadata.get("description", None),
            config_file=metadata["config_file"],
            scenarios= [
                ScenarioResult.load_json(json_file, metadata["weights"])
                for json_file in (result_dir / Path("results")).rglob("*.json")
            ],
            weights=metadata.get("weights", {})
        )

    def save_meta_data(self, weights: Optional[Dict] = None) -> None:
        if weights is None:
            # Load original weights from meta_data file
            with open(self.result_dir / Path("meta_data.json")) as f:
                metadata = json.load(f)
                weights = metadata.get("weights", {})

        meta_data = {
            "name": self.name,
            "agent": self.agent,
            "description": self.description,
            "config_file": self.config_file,
            "weights": weights
        }

        if self.description is None or self.description.strip() == "":
            del meta_data["description"]

        save_json(meta_data, self.result_dir / Path("meta_data.json"), indent=4)

    def save_statistics(self):
        data = {
            "total_score": self.average_score,
            "category_scores": self.average_category_scores,
            "metrics_average": self.metrics_average,
            "metrics_gathered": self.metrics_gathered,
            "scenarios": {
                scenario.name: scenario.get_statistics()
                for scenario in self.scenarios
            }
        }

        save_json(data, self.result_dir / Path("statistics.json"), max_fields=6, max_items=-1, max_outer_items=-1)

    @cached_property
    def failed_scenarios(self) -> List[Dict[str, Union[str, List[int]]]]:
        failed_results: List[Dict[str, Union[str, List[int]]]] = []
        for scenario in self.scenarios:
            failed = scenario.failed_results

            if len(failed["indexes"]) > 0:
                failed_results.append(failed)
        return failed_results

    def save_failed_scenarios(self):
        failed_scenarios_path = self.result_dir / Path("failed_scenarios.json")

        if len(self.failed_scenarios) == 0:
            logger.info("No errors happened")
            failed_scenarios_path.unlink(missing_ok=True)
            return

        logger.info(f"Writing failed scenarios to {failed_scenarios_path}")
        save_json(self.failed_scenarios, self.result_dir / Path("failed_scenarios.json"), indent=4)

    def save_scenarios(self) -> None:
        for scenario in self.scenarios:
            scenario.save_json()

    def summary_str(self):
        metrics_average_str = json_compact_dump(self.metrics_average, indent=2, max_items=-1, max_fields=-1, max_outer_items=-1, prefix_tabs=2)
        metrics_gathered_str = json_compact_dump(self.metrics_gathered, indent=2, max_items=-1, max_fields=-1, max_outer_items=-1, prefix_tabs=2)

        return (
            f"Name: {self.name} | Agent: {self.agent} | Description: {self.description}\n"
            f"   Number of scenarios: {len(self.scenarios)}\n"
            f"   Average score: {self.average_score:.2f}\n"
            f"   Average category scores: {self.average_category_scores}\n"
            f"   Metrics average:\n{metrics_average_str}\n"
            f"   Metrics gathered:\n{metrics_gathered_str}"
        )