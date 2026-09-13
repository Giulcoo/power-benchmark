import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from power_benchmark.results.data.ResultData import ResultData, ScenarioResult

logger = logging.getLogger("Benchmarker")

class ResultPlotter(ABC):
    """Abstract base class for benchmark result plotters."""

    def __init__(self, results: ResultData, folder: Path = None):
        if folder is not None:
            os.makedirs(folder, exist_ok=True)
            self.output_dir = folder
        else:
            self.output_dir = None
        self.results = results

    def plot(self, save_plots: bool = True) -> List[Any]:
        """Generate plots for the benchmark results. Returns a list of plot objects."""
        plots = []

        for scenario in self.results.scenarios:
            scenario_plots = self.plot_scenario(scenario, save_plot=save_plots)
            plots.extend(scenario_plots)

        summary_plots = self._create_summary_plots(save_plots=save_plots)
        plots.extend(summary_plots)

        return plots

    @abstractmethod
    def plot_scenario(self, scenario: ScenarioResult, save_plot: bool = True) -> List[Any]:
        """Generate plots for one scenario. Returns a list of plot objects."""
        pass

    @abstractmethod
    def _create_summary_plots(self, save_plots: bool = True) -> List[Any]:
        """Create individual summary plots."""
        pass

    def _save_figure(self, fig: Any, filename: str, folder: Optional[Path] = None) -> None:
        """Save figure to file."""
        if self.output_dir is None:
            logger.warning("Output directory is not set. Cannot save figure.")
            return

    @staticmethod
    def _extract_category_scores(results: List, category_name: str) -> List[float]:
        """Extract scores for a specific category from all results."""
        return [
            next((cat.category_score for cat in r.metric_categories if cat.name == category_name), 0)
            for r in results
        ]

    @staticmethod
    def _calculate_average_category_scores(results: List, category_names: List[str]) -> List[float]:
        """Calculate average scores for each category across all results."""
        avg_scores = []
        for cat_name in category_names:
            scores = [
                cat.category_score
                for result in results
                for cat in result.metric_categories
                if cat.name == cat_name
            ]
            avg_scores.append(np.mean(scores) if scores else 0)
        return avg_scores

    def _get_all_category_names(self) -> List[str]:
        """Get all unique category names across all scenarios."""
        return list({
            cat.name
            for scenario in self.results.scenarios
            for result in scenario.results
            for cat in result.metric_categories
            if cat.weight != 0
        })

    def _build_heatmap_data(self, category_names: List[str]) -> np.ndarray:
        """Build the data matrix for the heatmap."""
        heatmap_data = []
        for scenario in self.results.scenarios:
            row = self._calculate_average_category_scores(scenario.results, category_names)
            heatmap_data.append(row)
        return np.array(heatmap_data)

    def _collect_category_score_distribution(self) -> Dict[str, List[float]]:
        """Collect category scores across all runs and scenarios."""
        category_scores: Dict[str, List[float]] = {}
        for scenario in self.results.scenarios:
            for result in scenario.results:
                if result.failed:
                    continue

                for category in result.metric_categories:
                    if category.weight == 0:
                        continue
                    category_scores.setdefault(category.name, []).append(category.category_score)
        return category_scores

    def _clean_filename(self, filename: str) -> str:
        """Clean filename for safe file system usage."""
        return filename.replace(' ', '_').replace('/', '_')

    def _ensure_folder_exists(self, folder: Optional[Path]) -> Path:
        """Ensure output folder exists and return the full path."""
        if self.output_dir is None:
            logger.warning("Output directory is not set. Cannot save figure.")
            return Path("")

        if folder:
            full_path = self.output_dir / folder
            os.makedirs(full_path, exist_ok=True)
            return full_path
        return self.output_dir

    @staticmethod
    def _get_scenario_category_names(results: List) -> List[str]:
        """Get category names (weight != 0) from the first result."""
        if not results:
            return []
        return [cat.name for cat in results[0].metric_categories if cat.weight != 0]