from functools import cached_property
from typing import List, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from power_benchmark.results.data.ResultData import ResultData, ScenarioResult
from power_benchmark.results.textual.TextualReport import TextualReport
from power_benchmark.results.plotting.ResultPlotter import ResultPlotter
from typing import Literal
from power_benchmark.configs.config import get_config
from power_benchmark.configs.task_config import Task
import json


class Presenter:
    def __init__(self):
        self.config = get_config()
        self.result_dir = Path(self.config.output_dir)
        self.compare_with = Path(self.config.compare_with) if self.config.compare_with else None
        self.plotter_type = None

        self.modify_weights()


    @cached_property
    def results(self) -> ResultData:
        return ResultData.load_from_dir(self.result_dir)

    @cached_property
    def plot_plotter(self) -> ResultPlotter:
        return self._build_plotter(self._get_plotter_type("Plot"))

    @cached_property
    def webapp_plotter(self) -> ResultPlotter:
        return self._build_plotter(self._get_plotter_type("Webapp"))

    def _get_plotter_type(self, type: Literal["Plot", "Webapp"]) -> Literal["Matplotlib", "Plotly"]:
        if self.plotter_type is not None:
            return self.plotter_type

        other = "Webapp" if type == "Plot" else "Plot"

        plotter = self.config.plotter
        if isinstance(plotter, dict):
            plotter = plotter[type] if type in plotter else plotter[other]
        return plotter

    def _build_plotter(self, plotter_type: Literal["Matplotlib", "Plotly"]) -> ResultPlotter:
        if plotter_type == "Matplotlib":
            from power_benchmark.results.plotting.MatplotlibPlotter import MatplotlibPlotter
            return MatplotlibPlotter(self.results, self.result_dir / Path("plots"))
        elif plotter_type == "Plotly":
            from power_benchmark.results.plotting.PlotlyPlotter import PlotlyPlotter
            return PlotlyPlotter(self.results, self.result_dir / Path("plots"))
        else:
            raise ValueError(f"Unknown plotter type: {plotter_type}")

    def run(self):
        if Task.PLOT in self.config.task:
            self.plot()
        if Task.WEBAPP in self.config.task:
            from power_benchmark.results.app.WebappLauncher import WebappLauncher
            WebappLauncher(
                result_dir=self.result_dir if self.compare_with is None else (self.result_dir, self.compare_with),
                plotter_type=self._get_plotter_type("Webapp"),
            ).launch()
        if Task.TEXTUAL in self.config.task or Task.INTERACTIVE in self.config.task:
            reporter = TextualReport(self.results, self.result_dir / Path("report"))

            if Task.TEXTUAL in self.config.task:
                reporter.report()
            if Task.INTERACTIVE in self.config.task:
                reporter.interactive_report()

    def plot(self, save_plots: bool = True) -> List[plt.Figure] | List[go.Figure]:
        return self.plot_plotter.plot(save_plots)

    def plot_scenario(self, scenario: ScenarioResult, save_plot: bool = True) -> List[plt.Figure] | List[go.Figure]:
        return self.plot_plotter.plot_scenario(scenario, save_plot)

    def modify_weights(self):
        config_weights = self.config.eval_weights

        if config_weights == {}:
            return

        with open(self.result_dir / Path("meta_data.json")) as f:
            metadata = json.load(f)

        weights = metadata["weights"]

        for category, changes in config_weights.items():
            for metric, new_weight in changes.items():
                weights[category][metric] = new_weight

        with open(self.result_dir / Path("meta_data.json"), 'w') as f:
            json.dump(metadata, f, indent=4)

        ResultData.load_from_dir(self.result_dir).save_scenarios()