import re
import textwrap
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from power_benchmark.constants.plotting import WRAP_WIDTH, WRAP_WIDTH_Y, MAX_XTICKS, BASE_FONT_SIZE, MIN_FONT_SIZE
from power_benchmark.results.data.ResultData import ScenarioResult
from power_benchmark.results.plotting.ResultPlotter import ResultPlotter


def _natural_key(value: Any) -> tuple:
    """Sort key: 'run2' < 'run10', case-insensitive, numbers compared as numbers."""
    s = str(value)
    parts = re.split(r'(\d+)', s)
    return tuple(
        (1, int(p)) if p.isdigit() else (0, p.casefold())
        for p in parts if p != ''
    )


def _natural_sorted(values: Iterable[Any]) -> List[Any]:
    """Return values sorted alphabetically / numerically (natural order)."""
    return sorted(values, key=_natural_key)


class MatplotlibPlotter(ResultPlotter):
    """Handles all plotting logic for benchmark results using Matplotlib."""

    def plot_scenario(self, scenario: ScenarioResult, save_plot: bool = True) -> List[plt.Figure]:
        """Generate plots for one scenario. Returns a list of plot objects."""
        # print(plt.style.available) To print all available styles use this
        # plt.style.use('seaborn-v0_8')

        plots = []
        base_folder = Path(scenario.name)
        n_results = len(scenario.results)
        n_categories = len(self._get_scenario_category_names(scenario.results))

        # Plot 1: Score over index
        fig1, ax1 = plt.subplots(
            figsize=self._dynamic_figsize(n_results, height=6.0, per_item=0.35, min_w=8.0),
            constrained_layout=True,
        )
        self._plot_score_over_index(ax1, scenario)
        self._set_figure_title(fig1, scenario)
        if save_plot:
            self._save_figure(fig1, 'score_over_index', folder=base_folder)
        plots.append(fig1)

        # Plot 2: Category scores bar
        fig3, ax3 = plt.subplots(
            figsize=self._dynamic_figsize(n_results, height=6.0, per_item=0.45, min_w=9.0),
            constrained_layout=True,
        )
        self._plot_category_scores_bar(ax3, scenario)
        self._set_figure_title(fig3, scenario)
        if save_plot:
            self._save_figure(fig3, 'category_scores_bar', folder=base_folder)
        plots.append(fig3)

        # Plot 3: Category radar
        radar_size = min(12.0, max(8.0, 6.0 + 0.25 * n_categories))
        fig4, ax4 = plt.subplots(
            figsize=(radar_size, radar_size),
            subplot_kw=dict(projection='polar'),
            constrained_layout=True,
        )
        self._plot_category_radar(ax4, scenario)
        self._set_figure_title(fig4, scenario)
        if save_plot:
            self._save_figure(fig4, 'category_radar', folder=base_folder)
        plots.append(fig4)

        return plots

    # -------------------------------------------------------------------------
    # Sorting helpers
    # -------------------------------------------------------------------------

    def _sorted_scenarios(self) -> List[ScenarioResult]:
        """Scenarios sorted alphabetically / naturally by name."""
        return sorted(self.results.scenarios, key=lambda s: _natural_key(s.name))

    @staticmethod
    def _sorted_results(results: List[Any]) -> List[Any]:
        """Results sorted by their index (numeric or natural for string indices)."""
        return sorted(results, key=lambda r: _natural_key(r.index))

    def _get_scenario_category_names(self, results: List[Any]) -> List[str]:
        """Category names of a scenario, always naturally sorted."""
        return _natural_sorted(super()._get_scenario_category_names(results))

    def _get_all_category_names(self) -> List[str]:
        """All category names across scenarios, always naturally sorted."""
        return _natural_sorted(super()._get_all_category_names())

    # -------------------------------------------------------------------------
    # Summary plots
    # -------------------------------------------------------------------------

    def _create_summary_plots(self, save_plots: bool = True) -> List[plt.Figure]:
        """Create individual summary plots."""
        plots = []
        base_filename = f'{self.results.name}'
        n_scenarios = len(self.results.scenarios)
        category_names = self._get_all_category_names()

        # Plot 1: Scenario comparison bar
        fig1, ax1 = plt.subplots(
            figsize=self._dynamic_figsize(n_scenarios, height=6.5),
            constrained_layout=True,
        )
        self._plot_scenario_comparison_bar(ax1)
        self._set_summary_figure_title(fig1)
        if save_plots:
            self._save_figure(fig1, f'{base_filename}_scenario_comparison')
        plots.append(fig1)

        # Plot 2: Scenario distribution box
        fig2, ax2 = plt.subplots(
            figsize=self._dynamic_figsize(n_scenarios, height=6.5),
            constrained_layout=True,
        )
        self._plot_scenario_distribution_box(ax2)
        self._set_summary_figure_title(fig2)
        if save_plots:
            self._save_figure(fig2, f'{base_filename}_score_distribution')
        plots.append(fig2)

        # Plot 3: Category distribution box
        fig3, ax3 = plt.subplots(
            figsize=self._dynamic_figsize(len(category_names), height=6.5),
            constrained_layout=True,
        )
        self._plot_category_distribution_box(ax3)
        self._set_summary_figure_title(fig3)
        if save_plots:
            self._save_figure(fig3, f'{base_filename}_category_score_distribution')
        plots.append(fig3)

        # Plot 4: Category heatmap
        CELL_W, CELL_H = 0.55, 0.32  # inches per cell (was 0.9 / 0.5)
        fig4, ax4 = plt.subplots(
            figsize=(
                min(18.0, max(6.0, CELL_W * len(category_names) + 3.5)),
                min(14.0, max(4.0, CELL_H * n_scenarios + 2.5)),
            ),
            constrained_layout=True,
        )
        self._plot_category_heatmap(fig4, ax4)
        self._set_summary_figure_title(fig4)
        if save_plots:
            self._save_figure(fig4, f'{base_filename}_category_heatmap')
        plots.append(fig4)

        # Plot 5: Score progression
        fig5, ax5 = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
        self._plot_score_progression(ax5)
        self._set_summary_figure_title(fig5)
        if save_plots:
            self._save_figure(fig5, f'{base_filename}_score_progression')
        plots.append(fig5)

        return plots

    def _save_figure(self, fig: plt.Figure, filename: str, folder: Optional[Path] = None) -> None:
        """Save figure to file."""
        super()._save_figure(fig, filename, folder)
        if self.output_dir is None:
            return
        clean_filename = self._clean_filename(filename) + '.png'
        base_path = self._ensure_folder_exists(folder)
        fig.savefig(base_path / clean_filename, dpi=150, bbox_inches='tight')

    # -------------------------------------------------------------------------
    # Title helpers
    # -------------------------------------------------------------------------

    def _set_figure_title(self, fig: plt.Figure, scenario: ScenarioResult) -> None:
        """Set the title for a scenario figure."""
        fig.suptitle(f'Scenario {scenario.name}', fontsize=12, fontweight='bold')

    def _set_summary_figure_title(self, fig: plt.Figure) -> None:
        """Set the title for a summary figure."""
        fig.suptitle(f'Summary {self.results.name}', fontsize=14, fontweight='bold')

    # -------------------------------------------------------------------------
    # Individual plots
    # -------------------------------------------------------------------------

    def _plot_score_over_index(self, ax: plt.Axes, scenario: ScenarioResult) -> None:
        """Plot total score over result index with average line."""
        sorted_results = self._sorted_results(scenario.results)
        indices = [r.index for r in sorted_results]
        scores = [r.total_score for r in sorted_results]

        ax.plot(indices, scores, marker='o', linewidth=2, color='steelblue', markersize=6)
        ax.fill_between(indices, scores, alpha=0.3)
        ax.axhline(
            y=scenario.average_score,
            color='red', linestyle='--', linewidth=2,
            label=f'Average: {scenario.average_score:.4f}'
        )

        self._thin_numeric_xticks(ax, indices)
        self._format_axis(ax, 'Total Score', 'Total Score by Index')
        ax.set_xlabel('Result Index')
        ax.legend(loc='best', fontsize=9)

    def _plot_category_scores_bar(self, ax: plt.Axes, scenario: ScenarioResult) -> None:
        """Plot grouped bar chart of category scores per result."""
        results = self._sorted_results(scenario.results)
        if not results:
            return

        category_names = self._get_scenario_category_names(results)
        x = np.arange(len(results))
        n_categories = len(category_names)
        width = 0.8 / max(n_categories, 1)
        colors = plt.cm.tab10(np.linspace(0, 1, max(n_categories, 1)))

        for i, cat_name in enumerate(category_names):
            cat_scores = self._extract_category_scores(results, cat_name)
            offset = i * width - (n_categories - 1) * width / 2
            ax.bar(x + offset, cat_scores, width, label=cat_name, color=colors[i])

        # thin out index ticks so they never overlap
        step = max(1, int(np.ceil(len(results) / MAX_XTICKS)))
        ax.set_xticks(x[::step])
        ax.set_xticklabels([r.index for r in results][::step],
                           fontsize=self._auto_fontsize(len(results[::step]), base=9))

        self._format_axis(ax, 'Category Score', 'Category Scores by Result')
        ax.set_xlabel('Result Index')
        # legend outside the axes -> never covers the bars
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0),
                  fontsize=self._auto_fontsize(n_categories, base=9), frameon=False)

    def _plot_category_radar(self, ax: plt.Axes, scenario: ScenarioResult) -> None:
        """Plot radar chart for average category scores."""
        results = scenario.results
        if not results:
            return

        category_names = self._get_scenario_category_names(results)
        if not category_names:
            return

        avg_scores = self._calculate_average_category_scores(results, category_names)

        self._draw_radar_chart(ax, category_names, avg_scores)
        ax.set_title('Average Category Scores (Radar)', pad=25)

    def _plot_scenario_comparison_bar(self, ax: plt.Axes) -> None:
        """Plot bar chart comparing average scores across scenarios."""
        scenarios = self._sorted_scenarios()
        if not scenarios:
            return

        labels = self._wrap_labels([f'{s.name}' for s in scenarios])
        scores = [s.average_score for s in scenarios]
        colors = self._get_color_palette(len(scenarios))

        x = np.arange(len(scenarios))
        bars = ax.bar(x, scores, color=colors, edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)

        ax.axhline(
            y=self.results.average_score,
            color='red', linestyle='--', linewidth=2,
            label=f'Overall Avg: {self.results.average_score:.4f}'
        )

        self._add_bar_labels(ax, bars, scores)
        self._format_axis(ax, 'Average Score', 'Average Score by Scenario',
                          rotation=45, tick_fontsize=self._auto_fontsize(len(labels)))
        ax.legend(loc='best', fontsize=9)

    def _plot_scenario_distribution_box(self, ax: plt.Axes) -> None:
        """Plot box plot showing score distribution per scenario."""
        scenarios = self._sorted_scenarios()
        if not scenarios:
            return

        labels = self._wrap_labels([f'{s.name}' for s in scenarios])
        all_scores = [[r.total_score for r in s.results] for s in scenarios]
        colors = self._get_color_palette(len(scenarios))

        bp = self._boxplot(ax, all_scores, labels)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        self._format_axis(ax, 'Total Score', 'Score Distribution by Scenario',
                          rotation=45, tick_fontsize=self._auto_fontsize(len(labels)))

    def _plot_category_distribution_box(self, ax: plt.Axes) -> None:
        """Plot box plot showing score distribution per metric category."""
        category_scores = self._collect_category_score_distribution()
        # keep category order consistent with all other plots
        raw_labels = _natural_sorted(category_scores.keys())
        all_scores = [category_scores[label] for label in raw_labels]

        if not all_scores:
            self._format_axis(ax, 'Category Score', 'Score Distribution by Category')
            return

        labels = self._wrap_labels(raw_labels)
        colors = self._get_color_palette(len(labels))

        bp = self._boxplot(ax, all_scores, labels)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        self._format_axis(ax, 'Category Score', 'Score Distribution by Category',
                          rotation=45, tick_fontsize=self._auto_fontsize(len(labels)))

    def _plot_category_heatmap(self, fig: plt.Figure, ax: plt.Axes) -> None:
        """Plot heatmap of category scores across scenarios."""
        scenarios = self._sorted_scenarios()
        if not scenarios or not scenarios[0].results:
            return

        category_names = self._get_all_category_names()
        if not category_names:
            return

        heatmap_data = self._build_sorted_heatmap_data(category_names, scenarios)

        im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0.0, vmax=100.0)

        self._format_heatmap_axes(ax, category_names, scenarios)
        self._add_heatmap_annotations(ax, heatmap_data)

        ax.set_title('Category Scores Heatmap')
        fig.colorbar(im, ax=ax, shrink=0.8)

    def _build_sorted_heatmap_data(self, category_names: List[str],
                                   scenarios: List[ScenarioResult]) -> np.ndarray:
        """Build heatmap data in the given (sorted) scenario order.

        Falls back to reordering the rows produced by ``_build_heatmap_data``
        if that method does not accept an explicit scenario list.
        """
        try:
            return self._build_heatmap_data(category_names, scenarios)
        except TypeError:
            data = np.asarray(self._build_heatmap_data(category_names))
            original = list(self.results.scenarios)
            order = [original.index(s) for s in scenarios]
            return data[order, :]

    def _plot_score_progression(self, ax: plt.Axes) -> None:
        """Plot line chart showing score progression across result indices."""
        scenarios = self._sorted_scenarios()
        colors = self._get_color_palette(len(scenarios))

        for scenario, color in zip(scenarios, colors):
            sorted_results = self._sorted_results(scenario.results)
            indices = [r.index for r in sorted_results]
            scores = [r.total_score for r in sorted_results]
            ax.plot(
                indices, scores,
                marker='o', markersize=5, linewidth=2, color=color,
                label=f'{scenario.name}'
            )

        self._format_axis(ax, 'Total Score', 'Score Progression Across Results')
        ax.set_xlabel('Result Index')
        # legend outside so long scenario names don't overlap the lines
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0),
                  fontsize=self._auto_fontsize(len(scenarios), base=9), frameon=False)

    # -------------------------------------------------------------------------
    # Label / layout helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _wrap_label(label: Any, width: int = WRAP_WIDTH, max_lines: int = 2) -> str:
        """Wrap (and if needed truncate) a tick label."""
        lines = textwrap.wrap(str(label), width=width) or ['']
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            lines[-1] = lines[-1][:max(1, width - 1)] + '…'
        return '\n'.join(lines)

    @classmethod
    def _wrap_labels(cls, labels: List[Any], width: int = WRAP_WIDTH,
                     max_lines: int = 2) -> List[str]:
        """Wrap a list of tick labels."""
        return [cls._wrap_label(label, width, max_lines) for label in labels]

    @staticmethod
    def _auto_fontsize(n: int, base: int = BASE_FONT_SIZE, minimum: int = MIN_FONT_SIZE) -> int:
        """Shrink font size when there are many labels."""
        return max(minimum, int(base - 0.35 * max(0, n - 6)))

    @staticmethod
    def _dynamic_figsize(n: int, height: float = 6.0, per_item: float = 0.75,
                         min_w: float = 8.0, max_w: float = 24.0) -> Tuple[float, float]:
        """Figure width scales with the number of x categories."""
        return (float(min(max(min_w, per_item * max(n, 1) + 2.0), max_w)), float(height))

    @classmethod
    def _thin_numeric_xticks(cls, ax: plt.Axes, values: List[Any]) -> None:
        """Reduce the number of numeric x ticks to avoid overlap."""
        if not values:
            return
        step = max(1, int(np.ceil(len(values) / MAX_XTICKS)))
        ax.set_xticks(values[::step])
        ax.tick_params(axis='x', labelsize=cls._auto_fontsize(len(values[::step]), base=9))

    @staticmethod
    def _boxplot(ax: plt.Axes, data: List[List[float]], labels: List[str]):
        """Matplotlib-version-safe boxplot with tick labels."""
        try:  # matplotlib >= 3.9
            return ax.boxplot(data, tick_labels=labels, patch_artist=True)
        except TypeError:  # matplotlib < 3.9
            return ax.boxplot(data, labels=labels, patch_artist=True)

    # -------------------------------------------------------------------------
    # Matplotlib-specific helper methods
    # -------------------------------------------------------------------------

    @staticmethod
    def _get_color_palette(n: int) -> np.ndarray:
        """Generate a color palette with n colors."""
        return plt.cm.viridis(np.linspace(0.2, 0.8, max(n, 1)))

    @staticmethod
    def _format_axis(ax: plt.Axes, ylabel: str, title: str, rotation: int = 0,
                     tick_fontsize: Optional[int] = None) -> None:
        """Apply common formatting to an axis."""
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')
        if rotation:
            # rotation_mode='anchor' + ha='right' keeps labels from running into each other
            plt.setp(ax.get_xticklabels(), rotation=rotation, ha='right',
                     rotation_mode='anchor')
        if tick_fontsize:
            ax.tick_params(axis='x', labelsize=tick_fontsize)

    @staticmethod
    def _add_bar_labels(ax: plt.Axes, bars, values: List[float]) -> None:
        """Add value labels on top of bars."""
        fontsize = max(6, int(9 - 0.2 * max(0, len(values) - 8)))
        for bar, value in zip(bars, values):
            ax.annotate(
                f'{value:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 2), textcoords='offset points',
                ha='center', va='bottom', fontsize=fontsize
            )

    @staticmethod
    def _add_trend_line(ax: plt.Axes, x: List, y: List) -> None:
        """Add a trend line if enough data points exist."""
        if len(x) > 1:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_sorted = sorted(x)
            ax.plot(x_sorted, p(x_sorted), 'r--', alpha=0.7, label='Trend')
            ax.legend()

    @classmethod
    def _draw_radar_chart(cls, ax: plt.Axes, labels: List[str], values: List[float]) -> None:
        """Draw a radar chart on a polar axis."""
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        values_closed = list(values) + [values[0]]
        angles_closed = angles + [angles[0]]

        ax.plot(angles_closed, values_closed, 'o-', linewidth=2, color='steelblue')
        ax.fill(angles_closed, values_closed, alpha=0.25, color='steelblue')
        ax.set_xticks(angles)
        ax.set_xticklabels(
            cls._wrap_labels(labels, width=10),
            fontsize=max(6, 9 - len(labels) // 6)
        )
        # push labels away from the plot area
        ax.tick_params(axis='x', pad=12)
        ax.tick_params(axis='y', labelsize=7)

    def _format_heatmap_axes(self, ax: plt.Axes, category_names: List[str],
                             scenarios: List) -> None:
        """Format heatmap axis labels."""
        ax.set_xticks(np.arange(len(category_names)))
        ax.set_yticks(np.arange(len(scenarios)))
        ax.set_xticklabels(
            self._wrap_labels(category_names, width=WRAP_WIDTH),
            rotation=45, ha='right', rotation_mode='anchor',
            fontsize=max(8, self._auto_fontsize(len(category_names), base=10))
        )
        ax.set_yticklabels(
            self._wrap_labels([f'{s.name}' for s in scenarios],
                              width=WRAP_WIDTH_Y),
            fontsize=max(8, self._auto_fontsize(len(scenarios), base=10))
        )

    @staticmethod
    def _add_heatmap_annotations(ax: plt.Axes, data: np.ndarray) -> None:
        """Add value annotations to heatmap cells."""
        # fixed, readable size - cells are compact, so keep the numbers large
        fontsize = 11 if data.shape[1] <= 12 else 10
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                color = 'white' if data[i, j] < 0.5 else 'black'
                ax.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center',
                        fontsize=fontsize, fontweight='bold', color=color)