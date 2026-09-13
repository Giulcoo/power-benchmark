from abc import ABC, abstractmethod
import json
from typing import Any, Dict, Literal

import pandas as pd
import streamlit as st

from power_benchmark.results.data.ResultData import ResultData
from power_benchmark.results.data.ScenarioData import ScenarioResult, SingleResult


class ResultApp(ABC):
    """Abstract base class for Streamlit web applications visualizing benchmark results."""

    def __init__(self, results: ResultData, plotter: Literal["Matplotlib", "Plotly"]) -> None:
        self.results = results
        self.statistics = self._load_statistics()

        if plotter == "Matplotlib":
            from power_benchmark.results.plotting.MatplotlibPlotter import MatplotlibPlotter
            self.plotter = MatplotlibPlotter(self.results)
        elif plotter == "Plotly":
            from power_benchmark.results.plotting.PlotlyPlotter import PlotlyPlotter
            self.plotter = PlotlyPlotter(self.results)
        else:
            raise ValueError(f"Unknown plotter type: {self.plotter}")

    def _load_statistics(self) -> Dict[str, Any]:
        statistics_path = self.results.result_dir / "statistics.json"
        if not statistics_path.exists():
            return {}

        try:
            with open(statistics_path, "r", encoding="utf-8") as file:
                return json.load(file)
        except (OSError, json.JSONDecodeError):
            return {}

    @staticmethod
    def _coerce_category_scores(scores: Any) -> Dict[str, float]:
        if not isinstance(scores, dict):
            return {}

        category_scores: Dict[str, float] = {}
        for category, score in scores.items():
            try:
                category_scores[str(category)] = float(score)
            except (TypeError, ValueError):
                continue
        return category_scores

    def _zero_weight_category_names(self) -> set[str]:
        """Collect names of categories that have weight 0 across all results."""
        zero_weight: set[str] = set()
        weighted: set[str] = set()
        for scenario in self.results.scenarios:
            for result in scenario.results:
                for category in result.metric_categories:
                    if category.weight == 0:
                        zero_weight.add(category.name)
                    else:
                        weighted.add(category.name)
        # Only treat as zero-weight if never weighted anywhere
        return zero_weight - weighted

    @staticmethod
    def _average_category_scores_from_results(results: list[SingleResult]) -> Dict[str, float]:
        gathered_scores: Dict[str, list[float]] = {}

        for result in results:
            if result.failed:
                continue

            for category in result.metric_categories:
                if category.weight == 0:
                    continue
                gathered_scores.setdefault(category.name, []).append(category.category_score)

        return {
            category: sum(scores) / len(scores)
            for category, scores in gathered_scores.items()
            if scores
        }

    def _experiment_category_scores(self) -> Dict[str, float]:
        statistics_scores = self._coerce_category_scores(self.statistics.get("category_scores"))
        if statistics_scores:
            return statistics_scores

        gathered_scores: Dict[str, list[float]] = {}
        for scenario in self.results.scenarios:
            for category, score in self._scenario_category_scores(scenario).items():
                gathered_scores.setdefault(category, []).append(score)

        return {
            category: sum(scores) / len(scores)
            for category, scores in gathered_scores.items()
            if scores
        }

    def _scenario_category_scores(self, scenario: ScenarioResult) -> Dict[str, float]:
        scenario_statistics = self.statistics.get("scenarios", {}).get(scenario.name, {})
        statistics_scores = self._coerce_category_scores(scenario_statistics.get("category_scores"))
        if statistics_scores:
            return statistics_scores

        return self._average_category_scores_from_results(scenario.results)

    @staticmethod
    def _category_scores_df(scores: Dict[str, float]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"Metric Category": category, "Average Score": score}
                for category, score in scores.items()
            ]
        )

    def render_category_scores(self, title: str, scores: Dict[str, float]) -> None:
        st.subheader(title)

        zero_weight = self._zero_weight_category_names()
        scores = {
            category: score
            for category, score in scores.items()
            if category not in zero_weight
        }

        if not scores:
            st.info("No category scores available.")
            return

        category_df = self._category_scores_df(scores)
        table_col, chart_col = st.columns([2, 3])

        with table_col:
            st.dataframe(
                category_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Average Score": st.column_config.NumberColumn(format="%.4f"),
                },
            )

        with chart_col:
            st.bar_chart(category_df.set_index("Metric Category"))

    def configure_page(self) -> None:
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title=f"Results: {self.results.name}",
            page_icon="📊",
            layout="wide"
        )

    def render_header(self) -> None:
        """Render the page header with title and description."""
        st.title(f"📊 {self.results.name}")
        if self.results.description:
            st.markdown(f"*{self.results.description}*")

    def render_overview_metrics(self) -> None:
        """Render overview metrics cards."""
        st.header("Overview")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Average Score", f"{self.results.average_score:.4f}")
        with col2:
            st.metric("Total Scenarios", len(self.results.scenarios))
        with col3:
            total_results = sum(len(s.results) for s in self.results.scenarios)
            st.metric("Total Results", total_results)

        self.render_category_scores("Average Metric Category Scores", self._experiment_category_scores())

    def render_scenario_table(self) -> None:
        """Render the scenario overview table."""
        st.header("Scenario Overview")

        scenario_data = []
        for scenario in self.results.scenarios:
            scenario_row = {
                "Category": scenario.network_category,
                "Scenario": scenario.name,
                "Config": scenario.scenario_config,
                "Avg Score": scenario.average_score,
                "# Results": len(scenario.results)
            }
            for category, score in self._scenario_category_scores(scenario).items():
                scenario_row[f"{category} Avg"] = score
            scenario_data.append(scenario_row)

        scenario_df = pd.DataFrame(scenario_data)
        score_columns = [
            column
            for column in scenario_df.columns
            if column == "Avg Score" or column.endswith(" Avg")
        ]
        st.dataframe(
            scenario_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                column: st.column_config.NumberColumn(format="%.4f")
                for column in score_columns
            },
        )

    def render_scenario_selector(self) -> 'ScenarioResult':
        """Render scenario selector and return selected scenario."""
        scenario_options = [
            f"{s.network_category} / {s.name} / {s.scenario_config}"
            for s in self.results.scenarios
        ]
        selected_name = st.selectbox("Select Scenario", scenario_options)
        selected_idx = scenario_options.index(selected_name)
        return self.results.scenarios[selected_idx]

    def render_scenario_info(self, scenario: 'ScenarioResult') -> None:
        """Render scenario information panel."""
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader(f"Scenario: {scenario.name}")
            st.write(f"**Category:** {scenario.network_category}")
            st.write(f"**Config:** {scenario.scenario_config}")

        with col2:
            st.metric("Scenario Avg Score", f"{scenario.average_score:.4f}")

        self.render_category_scores("Scenario Metric Category Scores", self._scenario_category_scores(scenario))

    @abstractmethod
    def render_scenario_plot(self, scenario: 'ScenarioResult', **kwargs) -> None:
        """Render plots for a scenario. Must be implemented by subclasses."""
        pass

    def render_result_selector(self, scenario: 'ScenarioResult') -> 'SingleResult':
        """Render result selector and return selected result."""
        st.subheader("Individual Results")

        result_options = [
            f"Result {r.index} ({r.amount_of_days} days)"
            for r in scenario.results
        ]
        selected_name = st.selectbox("Select Result", result_options)
        result_idx = result_options.index(selected_name)
        return scenario.results[result_idx]

    def render_result_summary(self, result: 'SingleResult') -> None:
        """Render result summary information."""
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Score", f"{result.total_score:.4f}")
        with col2:
            st.metric("Days", result.amount_of_days)

    def render_metric_categories(self, result: 'SingleResult') -> None:
        """Render metric categories as tabs."""
        categories = [cat for cat in result.metric_categories if cat.weight != 0]

        if not categories:
            st.info("No metric categories available.")
            return

        tabs = st.tabs([cat.name for cat in categories])

        for tab, category in zip(tabs, categories):
            with tab:
                self._render_category_content(category)

    def _render_category_content(self, category) -> None:
        """Render content for a single metric category."""
        col1, col2 = st.columns([1, 3])

        with col1:
            st.metric("Category Score", f"{category.category_score:.4f}")
            st.metric("Weight", f"{category.weight:.2f}")

        with col2:
            metrics_data = []
            for metric in category.metrics:
                if metric.weight == 0:
                    continue
                scores = metric.score_dict()
                values = metric.value_dict()

                for eval_crit, score in scores.items():
                    value = values.get(eval_crit, "N/A")
                    metrics_data.append({
                        "Metric": metric.name,
                        "Eval Criterion": eval_crit,
                        "Value": value,
                        "Score": score,
                        "Weight": metric.weight
                    })

            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    @abstractmethod
    def render_all_plots(self, **kwargs) -> None:
        """Render all plots. Must be implemented by subclasses."""
        pass

    def render_scenario_details(self) -> None:
        """Render the scenario details section."""
        st.header("Scenario Details")

        selected_scenario = self.render_scenario_selector()
        self.render_scenario_info(selected_scenario)
        self.render_scenario_plot(selected_scenario)

        selected_result = self.render_result_selector(selected_scenario)
        self.render_result_summary(selected_result)
        self.render_metric_categories(selected_result)

    def run(self) -> None:
        """Run the Streamlit web application."""
        self.configure_page()
        self.render_header()
        self.render_overview_metrics()

        st.divider()
        self.render_scenario_table()

        st.divider()
        self.render_scenario_details()

        st.divider()
        self.render_all_plots()
