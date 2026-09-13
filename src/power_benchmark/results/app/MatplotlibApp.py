import streamlit as st

from power_benchmark.results.data.ResultData import ScenarioResult
from power_benchmark.results.app.ResultApp import ResultApp


class MatplotlibApp(ResultApp):
    """Streamlit application using Matplotlib for visualization."""

    def _render_figs(
        self,
        figs,
        empty_message: str,
        fig_width: float,
        fig_height: float,
        columns: int,
        use_container_width: bool = True,
    ) -> None:
        if not figs:
            st.info(empty_message)
            return

        valid_figs = [fig for fig in figs if fig is not None]
        if not valid_figs:
            st.info(empty_message)
            return

        cols = st.columns(columns)
        for i, fig in enumerate(valid_figs):
            fig.set_size_inches(fig_width, fig_height)
            fig.tight_layout()

            with cols[i % columns]:
                st.pyplot(fig, use_container_width=use_container_width)

    def _all_scenario_figs(self):
        figs = []
        for scenario in self.results.scenarios:
            figs.extend(self.plotter.plot_scenario(scenario, save_plot=False) or [])
        return figs

    def render_scenario_plot(
        self,
        scenario: 'ScenarioResult',
        fig_width: float = 4,
        fig_height: float = 3,
        columns: int = 4
    ) -> None:
        """Render Matplotlib plots for a scenario."""
        with st.expander("Show Plots", expanded=False):
            try:
                self._render_figs(
                    self.plotter.plot_scenario(scenario, save_plot=False),
                    "No plots available for this scenario.",
                    fig_width,
                    fig_height,
                    columns,
                )
            except Exception as e:
                st.warning(f"Could not generate plots: {e}")

    def render_all_plots(
        self,
        fig_width: float = 4,
        fig_height: float = 3,
        columns: int = 4
    ) -> None:
        """Render all Matplotlib plots grouped by plot type."""
        st.header("All Plots")

        with st.expander("Single Scenario Plots", expanded=False):
            try:
                self._render_figs(
                    self._all_scenario_figs(),
                    "No scenario plots available.",
                    fig_width,
                    fig_height,
                    columns,
                )
            except Exception as e:
                st.warning(f"Could not generate scenario plots: {e}")

        with st.expander("Overview Plots", expanded=False):
            try:
                self._render_figs(
                    self.plotter._create_summary_plots(save_plots=False),
                    "No overview plots available.",
                    fig_width,
                    fig_height,
                    columns,
                )
            except Exception as e:
                st.warning(f"Could not generate overview plots: {e}")
