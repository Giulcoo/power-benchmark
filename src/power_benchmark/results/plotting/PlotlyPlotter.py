import plotly.graph_objects as go
import numpy as np
from typing import List, Optional
from pathlib import Path

from power_benchmark.results.data.ResultData import ScenarioResult
from power_benchmark.results.plotting.ResultPlotter import ResultPlotter


class PlotlyPlotter(ResultPlotter):
    """Handles all plotting logic for benchmark results using Plotly."""

    def __init__(self, results, folder: Path = None):
        super().__init__(results, folder)
        self.color_palette = [
            '#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
            '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52'
        ]

    def plot_scenario(self, scenario: ScenarioResult, save_plot: bool = True) -> List[go.Figure]:
        """Generate plots for one scenario. Returns a list of Plotly figures."""
        plots = []
        base_folder = Path(f'{scenario.scenario_config}')
        title_suffix = (
            f"<br><sub>{scenario.network_category} - {scenario.name} | "
            f"Config: {scenario.scenario_config} | Avg: {scenario.average_score:.4f}</sub>"
        )

        plot_configs = [
            (self._plot_score_over_index, "Total Score by Index", 'score_over_index'),
            (self._plot_category_scores_bar, "Category Scores by Result", 'category_scores_bar'),
            (self._plot_category_radar, "Average Category Scores (Radar)", 'category_radar'),
        ]

        for plot_func, title, filename in plot_configs:
            fig = plot_func(scenario)
            fig.update_layout(title=f"{title}{title_suffix}")
            if save_plot:
                self._save_figure(fig, filename, folder=base_folder)
            plots.append(fig)

        return plots

    def _create_summary_plots(self, save_plots: bool = True) -> List[go.Figure]:
        """Create individual summary plots."""
        plots = []
        base_filename = f'{self.results.name}'
        title_suffix = (
            f"<br><sub>Benchmark: {self.results.name} | "
            f"Overall Avg: {self.results.average_score:.4f}</sub>"
        )

        plot_configs = [
            (self._plot_scenario_comparison_bar, "Average Score by Scenario", f'{base_filename}_scenario_comparison'),
            (self._plot_scenario_distribution_box, "Score Distribution by Scenario", f'{base_filename}_score_distribution'),
            (self._plot_category_distribution_box, "Score Distribution by Category", f'{base_filename}_category_score_distribution'),
            (self._plot_category_heatmap, "Category Scores Heatmap", f'{base_filename}_category_heatmap'),
            (self._plot_score_progression, "Score Progression Across Results", f'{base_filename}_score_progression'),
        ]

        for plot_func, title, filename in plot_configs:
            fig = plot_func()
            fig.update_layout(title=f"{title}{title_suffix}")
            if save_plots:
                self._save_figure(fig, filename)
            plots.append(fig)

        return plots

    def _save_figure(self, fig: go.Figure, filename: str, folder: Optional[Path] = None) -> None:
        """Save figure to file (HTML and PNG)."""
        super()._save_figure(fig, filename, folder)
        clean_filename = self._clean_filename(filename)
        base_path = self._ensure_folder_exists(folder) / clean_filename

        fig.write_html(str(base_path) + '.html')

        try:
            fig.write_image(str(base_path) + '.png', scale=2)
        except Exception:
            pass  # kaleido not installed

    def _plot_score_over_index(self, scenario: ScenarioResult) -> go.Figure:
        """Plot total score over result index with average line."""
        sorted_results = sorted(scenario.results, key=lambda r: r.index)
        indices = [r.index for r in sorted_results]
        scores = [r.total_score for r in sorted_results]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=indices, y=scores,
            mode='lines+markers',
            name='Total Score',
            line=dict(color='steelblue', width=2),
            marker=dict(size=10),
            fill='tozeroy',
            fillcolor='rgba(70, 130, 180, 0.3)',
            hovertemplate='Index: %{x}<br>Score: %{y:.4f}<extra></extra>'
        ))

        fig.add_hline(
            y=scenario.average_score,
            line_dash="dash", line_color="red",
            annotation_text=f"Avg: {scenario.average_score:.4f}",
            annotation_position="top right"
        )

        fig.update_layout(
            xaxis_title="Result Index",
            yaxis_title="Total Score",
            hovermode='x unified',
            template='plotly_white'
        )

        return fig

    def _plot_category_scores_bar(self, scenario: ScenarioResult) -> go.Figure:
        """Plot grouped bar chart of category scores per result."""
        results = scenario.results

        if not results:
            return go.Figure()

        category_names = [cat.name for cat in results[0].metric_categories if cat.weight != 0]
        result_indices = [r.index for r in results]

        fig = go.Figure()

        for i, cat_name in enumerate(category_names):
            cat_scores = self._extract_category_scores(results, cat_name)
            fig.add_trace(go.Bar(
                x=result_indices, y=cat_scores,
                name=cat_name,
                marker_color=self.color_palette[i % len(self.color_palette)],
                hovertemplate=f'{cat_name}<br>Index: %{{x}}<br>Score: %{{y:.4f}}<extra></extra>'
            ))

        fig.update_layout(
            barmode='group',
            xaxis_title="Result Index",
            yaxis_title="Category Score",
            legend_title="Categories",
            template='plotly_white',
            xaxis=dict(tickmode='array', tickvals=result_indices)
        )

        return fig

    def _plot_category_radar(self, scenario: ScenarioResult) -> go.Figure:
        """Plot radar chart for average category scores."""
        results = scenario.results

        if not results:
            return go.Figure()

        category_names = [cat.name for cat in results[0].metric_categories if cat.weight != 0]
        avg_scores = self._calculate_average_category_scores(results, category_names)

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=avg_scores + [avg_scores[0]],
            theta=category_names + [category_names[0]],
            fill='toself',
            fillcolor='rgba(70, 130, 180, 0.3)',
            line=dict(color='steelblue', width=2),
            marker=dict(size=8),
            hovertemplate='%{theta}<br>Score: %{r:.4f}<extra></extra>'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(avg_scores) * 1.1] if avg_scores else [0, 1]
                )
            ),
            template='plotly_white'
        )

        return fig

    def _plot_scenario_comparison_bar(self) -> go.Figure:
        """Plot bar chart comparing average scores across scenarios."""
        scenarios = self.results.scenarios
        labels = [f'{s.network_category}<br>{s.name}' for s in scenarios]
        scores = [s.average_score for s in scenarios]
        colors = [self.color_palette[i % len(self.color_palette)] for i in range(len(scenarios))]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=labels, y=scores,
            marker_color=colors,
            marker_line_color='black', marker_line_width=1,
            text=[f'{s:.3f}' for s in scores],
            textposition='outside',
            hovertemplate='%{x}<br>Score: %{y:.4f}<extra></extra>'
        ))

        fig.add_hline(
            y=self.results.average_score,
            line_dash="dash", line_color="red",
            annotation_text=f"Overall Avg: {self.results.average_score:.4f}",
            annotation_position="top right"
        )

        fig.update_layout(yaxis_title="Average Score", template='plotly_white')

        return fig

    def _plot_scenario_distribution_box(self) -> go.Figure:
        """Plot box plot showing score distribution per scenario."""
        scenarios = self.results.scenarios

        fig = go.Figure()

        for i, scenario in enumerate(scenarios):
            scores = [r.total_score for r in scenario.results]
            fig.add_trace(go.Box(
                y=scores,
                name=f'{scenario.network_category}<br>{scenario.name}',
                marker_color=self.color_palette[i % len(self.color_palette)],
                boxmean=True,
                hovertemplate='%{y:.4f}<extra></extra>'
            ))

        fig.update_layout(
            yaxis_title="Total Score",
            showlegend=False,
            template='plotly_white'
        )

        return fig

    def _plot_category_distribution_box(self) -> go.Figure:
        """Plot box plot showing score distribution per metric category."""
        category_scores = self._collect_category_score_distribution()

        fig = go.Figure()

        for i, (category_name, scores) in enumerate(category_scores.items()):
            fig.add_trace(go.Box(
                y=scores,
                name=category_name,
                marker_color=self.color_palette[i % len(self.color_palette)],
                boxmean=True,
                hovertemplate=f'{category_name}<br>Score: %{{y:.4f}}<extra></extra>'
            ))

        fig.update_layout(
            yaxis_title="Category Score",
            showlegend=False,
            template='plotly_white'
        )

        return fig

    def _plot_category_heatmap(self) -> go.Figure:
        """Plot heatmap of category scores across scenarios."""
        scenarios = self.results.scenarios

        if not scenarios or not scenarios[0].results:
            return go.Figure()

        category_names = self._get_all_category_names()
        heatmap_data = self._build_heatmap_data(category_names)
        scenario_labels = [f'{s.network_category}-{s.name}' for s in scenarios]

        fig = go.Figure()

        fig.add_trace(go.Heatmap(
            z=heatmap_data,
            x=category_names,
            y=scenario_labels,
            colorscale='RdYlGn',
            text=np.round(heatmap_data, 2),
            texttemplate='%{text}',
            textfont=dict(size=10),
            hovertemplate='Scenario: %{y}<br>Category: %{x}<br>Score: %{z:.4f}<extra></extra>'
        ))

        fig.update_layout(
            xaxis_title="Category",
            yaxis_title="Scenario",
            template='plotly_white'
        )

        return fig

    def _plot_score_progression(self) -> go.Figure:
        """Plot line chart showing score progression across result indices."""
        scenarios = self.results.scenarios

        fig = go.Figure()

        for i, scenario in enumerate(scenarios):
            sorted_results = sorted(scenario.results, key=lambda r: r.index)
            indices = [r.index for r in sorted_results]
            scores = [r.total_score for r in sorted_results]
            color = self.color_palette[i % len(self.color_palette)]

            fig.add_trace(go.Scatter(
                x=indices, y=scores,
                mode='lines+markers',
                name=f'{scenario.network_category}-{scenario.name}',
                line=dict(color=color, width=2),
                marker=dict(size=8),
                hovertemplate='Index: %{x}<br>Score: %{y:.4f}<extra></extra>'
            ))

        fig.update_layout(
            xaxis_title="Result Index",
            yaxis_title="Total Score",
            legend_title="Scenarios",
            hovermode='x unified',
            template='plotly_white'
        )

        return fig
