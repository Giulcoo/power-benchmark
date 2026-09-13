import json
import copy
from pathlib import Path
from typing import Dict, Any, Optional, List, Literal


class WeightModifier:
    """ Class to modify weights and recalculating scores accordingly for a given benchmark JSON result file. """

    def __init__(self, json_file_path: str):
        """
        Initialize the WeightModifier.

        Args:
            json_file_path: Path to the JSON file containing benchmark results.
        """
        self.json_file_path = json_file_path

        # Constant list of score categories
        self.score_categories = ["overload_mitigation", "voltage_violation_mitigation", "security", "costs", "computational_performance"]

        # Load the JSON file
        with open(self.json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.updated_data = copy.deepcopy(data)

    def update_weights(
            self,
            weights: Dict[str, Any],
            total_weight_calculation: Literal["mean", "sum"] = "mean"
    ) -> Dict[str, Any]:
        """
        Updates weights and recalculates category scores and total scores.

        Args:
            weights: Dictionary containing the new weights
            total_weight_calculation: Method to calculate total weights ("mean" or "sum")

        Returns:
            The updated data dictionary
        """
        # Iterate through the nested structure
        for category_content in self.updated_data.values():
            for config_name, config_list in category_content.items():
                for entry in config_list:
                    self._process_entry(entry, weights, total_weight_calculation)

        return self.updated_data

    def _process_entry(
            self,
            entry: Dict[str, Any],
            weights: Dict[str, Any],
            total_weight_calculation: Literal["mean", "sum"] = "mean"
    ) -> None:
        """
        Updates weights and recalculates scores for a single entry.

        Args:
            entry: A single configuration entry containing categories and metrics
            weights: Dictionary containing the new weights
        """
        # Update weights and recalculate category_score for each category
        for cat in self.score_categories:
            if cat not in entry:
                continue

            category_data = entry[cat]
            category_weights = weights.get(cat, {})

            total_weighted_score = 0.0
            total_weight = 0.0

            for metric_name, metric_data in category_data.items():
                if metric_name == "category_score":
                    continue

                if isinstance(metric_data, dict) and "weight" in metric_data and "score" in metric_data:
                    # Update weight if specified in weights dict
                    if metric_name in category_weights:
                        metric_data["weight"] = category_weights[metric_name]

                    # Accumulate for weighted average
                    weight = metric_data["weight"]
                    score = metric_data["score"]
                    total_weighted_score += score * weight
                    total_weight += weight

            # Recalculate category_score (weighted average)
            if total_weight > 0:
                category_data["category_score"] = total_weighted_score / total_weight
            else:
                category_data["category_score"] = 0.0

        # Recalculate total_score using category weights from "total"
        total_config = weights.get("total", {})
        total_weighted_score = 0.0
        total_weight = 0.0

        for cat in self.score_categories:
            if cat in entry:
                category_score = entry[cat].get("category_score", 0.0)
                cat_weight = total_config.get(cat, 1.0)  # Default to 1.0 if not specified
                total_weighted_score += category_score * cat_weight
                total_weight += cat_weight

        if total_weight_calculation == "mean":
            if total_weight > 0:
                entry["total_score"] = total_weighted_score / total_weight
            else:
                entry["total_score"] = 0.0
        elif total_weight_calculation == "sum":
            entry["total_score"] = total_weighted_score

    def pretty_print_changes(self):
        """ Pretty print the updated data with recalculated scores. """
        print(json.dumps(self.updated_data, indent=4))

    def safe_changes(self, output_file_path: Optional[str] = None, override: bool = False):
        if override:
            # Overwrite the original file
            with open(self.json_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.updated_data, f, indent=4)

            print(f"Original file overwritten: {self.json_file_path}")
        else:
            # Determine output file path
            if output_file_path is None:
                path = Path(self.json_file_path)
                output_file_path = str(path.parent / f"{path.stem}_updated{path.suffix}")

            # Save the updated data
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.updated_data, f, indent=4)

            print(f"Updated file saved to: {output_file_path}")