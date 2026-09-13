from abc import abstractmethod
import json
from typing import Any, Dict, Literal

import pandas as pd
import streamlit as st

from power_benchmark.results.data.ResultData import ResultData
from power_benchmark.results.data.ScenarioData import ScenarioResult, SingleResult


class ComparisonApp:
    """Streamlit web application for comparing two benchmark results side by side."""

    def __init__(
        self,
        results_a: ResultData,
        results_b: ResultData,
        plotter: Literal["Matplotlib", "Plotly"] = "Plotly",
    ) -> None:
        self.results_a = results_a
        self.results_b = results_b
        self.statistics_a = self._load_statistics(results_a)
        self.statistics_b = self._load_statistics(results_b)
        self.plotter_type = plotter
        self.plotter_a = self._build_plotter(results_a)
        self.plotter_b = self._build_plotter(results_b)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_plotter(self, results: ResultData):
        if self.plotter_type == "Matplotlib":
            from power_benchmark.results.plotting.MatplotlibPlotter import MatplotlibPlotter
            return MatplotlibPlotter(results)
        elif self.plotter_type == "Plotly":
            from power_benchmark.results.plotting.PlotlyPlotter import PlotlyPlotter
            return PlotlyPlotter(results)
        else:
            raise ValueError(f"Unknown plotter type: {self.plotter_type}")

    def _scenario_label(self, scenario: ScenarioResult) -> str:
        return f"{scenario.network_category} / {scenario.name}"

    @staticmethod
    def _load_statistics(results: ResultData) -> Dict[str, Any]:
        statistics_path = results.result_dir / "statistics.json"
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

    @staticmethod
    def _average_category_scores_from_results(results: list[SingleResult]) -> Dict[str, float]:
        gathered_scores: Dict[str, list[float]] = {}

        for result in results:
            if result.failed:
                continue

            for category in result.metric_categories:
                gathered_scores.setdefault(category.name, []).append(category.category_score)

        return {
            category: sum(scores) / len(scores)
            for category, scores in gathered_scores.items()
            if scores
        }

    @staticmethod
    def _zero_weight_categories(results: ResultData) -> set[str]:
        total_weights = results.weights.get("total", {}) if results.weights else {}
        return {
            str(category)
            for category, weight in total_weights.items()
            if weight == 0
        }

    def _filter_zero_weight(
            self,
            results: ResultData,
            scores: Dict[str, float],
    ) -> Dict[str, float]:
        zero_categories = self._zero_weight_categories(results)
        return {
            category: score
            for category, score in scores.items()
            if category not in zero_categories
        }

    def _experiment_category_scores(
            self,
            results: ResultData,
            statistics: Dict[str, Any],
    ) -> Dict[str, float]:
        statistics_scores = self._coerce_category_scores(statistics.get("category_scores"))
        if statistics_scores:
            return self._filter_zero_weight(results, statistics_scores)

        gathered_scores: Dict[str, list[float]] = {}
        for scenario in results.scenarios:
            for category, score in self._scenario_category_scores(results, scenario, statistics).items():
                gathered_scores.setdefault(category, []).append(score)

        averaged = {
            category: sum(scores) / len(scores)
            for category, scores in gathered_scores.items()
            if scores
        }
        return self._filter_zero_weight(results, averaged)

    def _scenario_category_scores(
            self,
            results: ResultData,
            scenario: ScenarioResult,
            statistics: Dict[str, Any],
    ) -> Dict[str, float]:
        scenario_statistics = statistics.get("scenarios", {}).get(scenario.name, {})
        statistics_scores = self._coerce_category_scores(scenario_statistics.get("category_scores"))
        if statistics_scores:
            return self._filter_zero_weight(results, statistics_scores)

        averaged = self._average_category_scores_from_results(scenario.results)
        return self._filter_zero_weight(results, averaged)

    def _category_score_comparison_df(
        self,
        scores_a: Dict[str, float],
        scores_b: Dict[str, float],
    ) -> pd.DataFrame:
        rows = []
        for category in sorted(set(scores_a) | set(scores_b)):
            score_a = scores_a.get(category)
            score_b = scores_b.get(category)
            rows.append({
                "Metric Category": category,
                self.results_a.name: score_a,
                self.results_b.name: score_b,
                "Delta (A-B)": score_a - score_b
                    if score_a is not None and score_b is not None else None,
            })
        return pd.DataFrame(rows)

    def _render_category_score_comparison(
        self,
        title: str,
        scores_a: Dict[str, float],
        scores_b: Dict[str, float],
    ) -> None:
        st.subheader(title)

        if not scores_a and not scores_b:
            st.info("No category scores available.")
            return

        comparison_df = self._category_score_comparison_df(scores_a, scores_b)
        chart_df = comparison_df.set_index("Metric Category")[
            [self.results_a.name, self.results_b.name]
        ]

        table_col, chart_col = st.columns([2, 3])
        with table_col:
            st.dataframe(
                comparison_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    self.results_a.name: st.column_config.NumberColumn(format="%.4f"),
                    self.results_b.name: st.column_config.NumberColumn(format="%.4f"),
                    "Delta (A-B)": st.column_config.NumberColumn(format="%.4f"),
                },
            )

        with chart_col:
            st.bar_chart(chart_df, stack=False)

    # ------------------------------------------------------------------
    # Page setup
    # ------------------------------------------------------------------

    def configure_page(self) -> None:
        st.set_page_config(
            page_title=f"Comparison: {self.results_a.name} vs {self.results_b.name}",
            page_icon="⚖️",
            layout="wide",
        )

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------

    def render_header(self) -> None:
        st.title("⚖️ Benchmark Comparison")
        col_a, col_sep, col_b = st.columns([5, 1, 5])
        with col_a:
            st.subheader(f"🅰️ {self.results_a.name}")
            if self.results_a.description:
                st.markdown(f"*{self.results_a.description}*")
        with col_sep:
            st.markdown("<div style='text-align:center;font-size:2rem;padding-top:0.5rem'>vs</div>",
                        unsafe_allow_html=True)
        with col_b:
            st.subheader(f"🅱️ {self.results_b.name}")
            if self.results_b.description:
                st.markdown(f"*{self.results_b.description}*")

    # ------------------------------------------------------------------
    # Overview metrics
    # ------------------------------------------------------------------

    def render_overview_metrics(self) -> None:
        st.header("Overview")

        headers = st.columns([3, 3, 3])
        headers[0].markdown("**Metric**")
        headers[1].markdown(f"**🅰️ {self.results_a.name}**")
        headers[2].markdown(f"**🅱️ {self.results_b.name}**")

        def _delta(val_a: float, val_b: float) -> str:
            diff = val_a - val_b
            return f"{diff:+.4f}"

        col_label, col_a, col_b = st.columns([3, 3, 3])

        with col_label:
            st.metric("Average Score", "")
        with col_a:
            st.metric(
                label="🅰️",
                value=f"{self.results_a.average_score:.2f}",
                delta=_delta(self.results_a.average_score, self.results_b.average_score),
            )
        with col_b:
            st.metric(label="🅱️", value=f"{self.results_b.average_score:.4f}",
                delta=_delta(self.results_b.average_score, self.results_a.average_score))

        col_label2, col_a2, col_b2 = st.columns([3, 3, 3])
        with col_label2:
            st.metric("Total Scenarios", "")
        with col_a2:
            st.metric("🅰️", len(self.results_a.scenarios))
        with col_b2:
            st.metric("🅱️", len(self.results_b.scenarios))

        col_label3, col_a3, col_b3 = st.columns([3, 3, 3])
        total_a = sum(len(s.results) for s in self.results_a.scenarios)
        total_b = sum(len(s.results) for s in self.results_b.scenarios)
        with col_label3:
            st.metric("Total Results", "")
        with col_a3:
            st.metric("🅰️", total_a)
        with col_b3:
            st.metric("🅱️", total_b)

        self._render_category_score_comparison(
            "Average Metric Category Scores",
            self._experiment_category_scores(self.results_a, self.statistics_a),
            self._experiment_category_scores(self.results_b, self.statistics_b),
        )

    # ------------------------------------------------------------------
    # Scenario table comparison
    # ------------------------------------------------------------------

    def render_scenario_table(self) -> None:
        st.header("Scenario Overview")

        def _build_df(results: ResultData, suffix: str) -> pd.DataFrame:
            rows = [
                {
                    "Category": s.network_category,
                    "Scenario": s.name,
                    f"Avg Score {suffix}": round(s.average_score, 4),
                    f"# Results {suffix}": len(s.results),
                }
                for s in results.scenarios
            ]
            return pd.DataFrame(rows)

        df_a = _build_df(self.results_a, "🅰️")
        df_b = _build_df(self.results_b, "🅱️")

        merge_keys = ["Category", "Scenario"]
        merged = pd.merge(df_a, df_b, on=merge_keys, how="outer")
        cols = list(merged.columns)
        cols[4], cols[5] = cols[5], cols[4]
        merged = merged[cols]

        # Score delta column
        score_col_a = "Avg Score 🅰️"
        score_col_b = "Avg Score 🅱️"
        if score_col_a in merged.columns and score_col_b in merged.columns:
            merged["Δ Score (🅰️−🅱️)"] = (
                merged[score_col_a] - merged[score_col_b]
            ).round(4)

        st.dataframe(
            merged,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Δ Score (🅰️−🅱️)": st.column_config.NumberColumn(format="%.4f"),
            },
        )

    # ------------------------------------------------------------------
    # Scenario selector
    # ------------------------------------------------------------------

    def render_scenario_selector(self) -> tuple[ScenarioResult | None, ScenarioResult | None]:
        """
        Let the user pick a scenario by label.
        Returns the matching ScenarioResult from each result set (or None if not found).
        """
        all_labels = sorted(
            {self._scenario_label(s) for s in self.results_a.scenarios}
            | {self._scenario_label(s) for s in self.results_b.scenarios}
        )

        selected_label = st.selectbox("Select Scenario", all_labels)

        scenario_a = next(
            (s for s in self.results_a.scenarios if self._scenario_label(s) == selected_label),
            None,
        )
        scenario_b = next(
            (s for s in self.results_b.scenarios if self._scenario_label(s) == selected_label),
            None,
        )
        return scenario_a, scenario_b

    # ------------------------------------------------------------------
    # Scenario details
    # ------------------------------------------------------------------

    def render_scenario_details(self) -> None:
        st.header("Scenario Details")
        scenario_a, scenario_b = self.render_scenario_selector()

        self._render_category_score_comparison(
            "Scenario Metric Category Scores",
            self._scenario_category_scores(self.results_a, scenario_a, self.statistics_a) if scenario_a else {},
            self._scenario_category_scores(self.results_b, scenario_b, self.statistics_b) if scenario_b else {},
        )

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown(f"### 🅰️ {self.results_a.name}")
            if scenario_a:
                self._render_scenario_info(scenario_a)
                self._render_scenario_plots(scenario_a, self.plotter_a, prefix="a")
                self._render_result_section(scenario_a, prefix="a")
            else:
                st.warning("Scenario not available in result 🅰️.")

        with col_b:
            st.markdown(f"### 🅱️ {self.results_b.name}")
            if scenario_b:
                self._render_scenario_info(scenario_b)
                self._render_scenario_plots(scenario_b, self.plotter_b, prefix="b")
                self._render_result_section(scenario_b, prefix="b")
            else:
                st.warning("Scenario not available in result 🅱️.")

    def _render_scenario_info(self, scenario: ScenarioResult) -> None:
        st.write(f"**Category:** {scenario.network_category}")
        st.metric("Avg Score", f"{scenario.average_score:.4f}")

    def _render_scenario_plots(self, scenario: ScenarioResult, plotter, prefix: str) -> None:
        with st.expander("📈 Show Plots", expanded=False):
            try:
                figs = plotter.plot_scenario(scenario, save_plot=False)
                valid_figs = [f for f in (figs or []) if f is not None]
                if not valid_figs:
                    st.info("No plots available.")
                    return
                self._display_figs(valid_figs, key_prefix=f"scenario_{prefix}_{scenario.scenario_config}", columns=1)
            except Exception as e:
                st.warning(f"Could not generate plots: {e}")

    def _render_result_section(self, scenario: ScenarioResult, prefix: str) -> None:
        result_options = [
            f"Result {r.index} ({r.amount_of_days} days)" for r in scenario.results
        ]
        selected = st.selectbox(
            "Select Result",
            result_options,
            key=f"result_selector_{prefix}_{scenario.scenario_config}",
        )
        result = scenario.results[result_options.index(selected)]

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Score", f"{result.total_score:.4f}")
        with col2:
            st.metric("Days", result.amount_of_days)

        self._render_metric_categories(result)

    def _render_metric_categories(self, result: SingleResult) -> None:
        if not result.metric_categories:
            st.info("No metric categories available.")
            return

        tabs = st.tabs([cat.name for cat in result.metric_categories])
        for tab, category in zip(tabs, result.metric_categories):
            with tab:
                st.metric("Category Score", f"{category.category_score:.4f}")
                st.metric("Weight", f"{category.weight:.2f}")

                metrics_data = []
                for metric in category.metrics:
                    for eval_crit, score in metric.score_dict().items():
                        metrics_data.append({
                            "Metric": metric.name,
                            "Eval Criterion": eval_crit,
                            "Value": metric.value_dict().get(eval_crit, "N/A"),
                            "Score": score,
                            "Weight": metric.weight,
                        })

                st.dataframe(
                    pd.DataFrame(metrics_data),
                    use_container_width=True,
                    hide_index=True,
                )

    # ------------------------------------------------------------------
    # All plots
    # ------------------------------------------------------------------

    def render_all_plots(self) -> None:
        st.header("All Plots")
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader(self.results_a.name)
            with st.expander("Single Scenario Plots", expanded=False):
                self._render_scenario_plots_for(self.plotter_a, prefix="all_a_scenario")
            with st.expander("Overview Plots", expanded=False):
                self._render_overview_plots_for(self.plotter_a, prefix="all_a_overview")

        with col_b:
            st.subheader(self.results_b.name)
            with st.expander("Single Scenario Plots", expanded=False):
                self._render_scenario_plots_for(self.plotter_b, prefix="all_b_scenario")
            with st.expander("Overview Plots", expanded=False):
                self._render_overview_plots_for(self.plotter_b, prefix="all_b_overview")

    def _render_scenario_plots_for(self, plotter, prefix: str) -> None:
        try:
            figs = []
            for scenario in plotter.results.scenarios:
                figs.extend(plotter.plot_scenario(scenario, save_plot=False) or [])
            valid_figs = [f for f in (figs or []) if f is not None]
            if not valid_figs:
                st.info("No scenario plots available.")
                return
            self._display_figs(valid_figs, key_prefix=prefix, columns=1)
        except Exception as e:
            st.warning(f"Could not generate scenario plots: {e}")

    def _render_overview_plots_for(self, plotter, prefix: str) -> None:
        try:
            figs = plotter._create_summary_plots(save_plots=False)
            valid_figs = [f for f in (figs or []) if f is not None]
            if not valid_figs:
                st.info("No overview plots available.")
                return
            self._display_figs(valid_figs, key_prefix=prefix, columns=1)
        except Exception as e:
            st.warning(f"Could not generate overview plots: {e}")

    # ------------------------------------------------------------------
    # Plot rendering (handles both Matplotlib and Plotly)
    # ------------------------------------------------------------------

    def _display_figs(self, figs: list, key_prefix: str, columns: int = 2) -> None:
        if self.plotter_type == "Plotly":
            for i in range(0, len(figs), columns):
                cols = st.columns(columns)
                for j, col in enumerate(cols):
                    idx = i + j
                    if idx < len(figs):
                        with col:
                            st.plotly_chart(
                                figs[idx],
                                use_container_width=True,
                                key=f"{key_prefix}_{idx}",
                            )
        else:  # Matplotlib
            cols = st.columns(columns)
            for i, fig in enumerate(figs):
                fig.set_size_inches(4, 3)
                fig.tight_layout()
                with cols[i % columns]:
                    st.pyplot(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run the comparison Streamlit web application."""
        self.configure_page()
        self.render_header()

        st.divider()
        self.render_overview_metrics()

        st.divider()
        self.render_scenario_table()

        st.divider()
        self.render_scenario_details()

        st.divider()
        self.render_all_plots()
