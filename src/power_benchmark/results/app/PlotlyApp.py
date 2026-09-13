import streamlit as st

from power_benchmark.results.data.ResultData import ScenarioResult
from power_benchmark.results.app.ResultApp import ResultApp


class PlotlyApp(ResultApp):
    """Streamlit application using Plotly for visualization."""

    def _render_figs(
        self,
        figs,
        empty_message: str,
        height: int,
        columns: int,
        key_prefix: str,
    ) -> None:
        if not figs:
            st.info(empty_message)
            return

        valid_figs = [fig for fig in figs if fig is not None]
        if not valid_figs:
            st.info(empty_message)
            return

        for i in range(0, len(valid_figs), columns):
            cols = st.columns(columns)
            for j, col in enumerate(cols):
                idx = i + j
                if idx < len(valid_figs):
                    with col:
                        st.plotly_chart(
                            valid_figs[idx],
                            use_container_width=True,
                            height=height,
                            key=f"{key_prefix}_{idx}",
                        )

    def _all_scenario_figs(self):
        figs = []
        for scenario in self.results.scenarios:
            figs.extend(self.plotter.plot_scenario(scenario, save_plot=False) or [])
        return figs

    def render_scenario_plot(
        self,
        scenario: 'ScenarioResult',
        height: int = 400,
        columns: int = 2
    ) -> None:
        """Render interactive Plotly plots for a scenario."""
        with st.expander("Show Plots", expanded=False):
            try:
                self._render_figs(
                    self.plotter.plot_scenario(scenario, save_plot=False),
                    "No plots available for this scenario.",
                    height,
                    columns,
                    f"scenario_plot_{scenario.scenario_config}",
                )
            except Exception as e:
                st.warning(f"Could not generate plots: {e}")

    def render_all_plots(self, height: int = 400, columns: int = 2) -> None:
        """Render all Plotly plots grouped by plot type."""
        st.header("All Plots")

        with st.expander("Single Scenario Plots", expanded=False):
            try:
                self._render_figs(
                    self._all_scenario_figs(),
                    "No scenario plots available.",
                    height,
                    columns,
                    "all_scenario_plot",
                )
            except Exception as e:
                st.warning(f"Could not generate scenario plots: {e}")

        with st.expander("Overview Plots", expanded=False):
            try:
                self._render_figs(
                    self.plotter._create_summary_plots(save_plots=False),
                    "No overview plots available.",
                    height,
                    columns,
                    "overview_plot",
                )
            except Exception as e:
                st.warning(f"Could not generate overview plots: {e}")
