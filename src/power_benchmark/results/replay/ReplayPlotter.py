import copy
from pathlib import Path
from typing import List, Tuple, Union, Optional
import os

import plotly.graph_objects as go
import pandapower as pp

from pandapower_env.toolbox.env_specs import LoggedArray

import logging

from power_benchmark.results.replay.action_figure import ActionFigureBuilder, SubstationStateTracker
from power_benchmark.results.replay.combiner import FigureCombiner
from power_benchmark.results.replay.exporter import HTMLExporter, CSVExportBuilder
from power_benchmark.results.replay.net_figure import NetworkFigureBuilder

logger = logging.getLogger("Benchmarker")


class ReplayPlotter:
    def __init__(self, agent_config, scenario_name: str):
        self.scenario_name = scenario_name
        self.net_figs: List[go.Figure] = []
        self.action_figs: List[go.Figure] = []
        self.nets: List[pp.pandapowerNet] = []
        self.final_figs: Union[Tuple[go.Figure, go.Figure], go.Figure, None] = None
        self.agent_config = copy.deepcopy(agent_config)
        self.action_lookup = {index: d for index, d in enumerate(self.agent_config["action_space"])}

        # Tracker is initialised lazily on the first add_net call,
        # because we need a net instance to set up the default states.
        self._state_tracker: Optional[SubstationStateTracker] = None

    def add_net(self, after: pp.pandapowerNet, action: int):
        self.net_figs.append(NetworkFigureBuilder(after).build())

        action_dict = self.action_lookup.get(action, None)
        if action_dict is None:
            logger.warning(
                f"Problem in {self.scenario_name}: Could not find action {action} "
                f"with possible actions: {self.action_lookup.keys()}"
            )
            action_dict = self.action_lookup[0]

        if "substations" in action_dict:
            # Initialise tracker on first use
            if self._state_tracker is None:
                self._state_tracker = SubstationStateTracker(after)

            # Persist the new substation states BEFORE building the figure
            self._state_tracker.update(action_dict)

            self.action_figs.append(
                ActionFigureBuilder(
                    net=after,
                    action_dict=action_dict,
                    layout_action_dict=self._state_tracker.get_full_layout_dict(),
                ).build()
            )

        self.nets.append(copy.deepcopy(after))

    def combine_figs(self, run_actions: LoggedArray):
        actions_list = [self.action_lookup[a] for a in run_actions if a in self.action_lookup]

        if self.action_figs:
            self.final_figs = (
                FigureCombiner(self.action_figs, None).build(),
                FigureCombiner(self.net_figs, actions_list).build(),
            )
        else:
            self.final_figs = FigureCombiner(self.net_figs, actions_list).build()

    def save(self, output_file: str | Path):
        os.makedirs(Path(output_file).parent, exist_ok=True)

        if isinstance(self.final_figs, tuple):
            exporter = HTMLExporter(
                self.final_figs[0], self.nets, self.final_figs[1],
                "Action Graph", "Network Graph",
            )
        elif isinstance(self.final_figs, go.Figure):
            exporter = HTMLExporter(self.final_figs, self.nets, None, "Action Graph", "")
        else:
            raise ValueError("Final figure not created. Call combine_figs() before saving.")

        html_out_file = str(output_file)
        if not html_out_file.endswith(".html"):
            html_out_file += ".html"
        exporter.export(str(output_file))

        csv_out_folder = html_out_file.replace(".html", "") + "_csv"
        os.makedirs(csv_out_folder, exist_ok=True)
        CSVExportBuilder(self.nets).build_files(csv_out_folder)

    @staticmethod
    def export_combined_html(
        combined_fig_1: go.Figure,
        combined_fig_2: Optional[go.Figure],
        nets: list,
        output_path: str = "output.html",
        title_1: str = "Figure 1",
        title_2: str = "Figure 2",
    ):
        """Export an HTML page with one or two combined Plotly figures and net DataFrames."""
        HTMLExporter(
            combined_fig_1=combined_fig_1,
            nets=nets,
            combined_fig_2=combined_fig_2,
            title_1=title_1,
            title_2=title_2,
        ).export(output_path)