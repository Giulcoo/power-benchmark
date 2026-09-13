import pandapower as pp
import pandapower.plotting.plotly as pplotly
import pandas as pd
import plotly.graph_objects as go
import math

from power_benchmark.results.replay.powerflow import run_nminus1_with_line_outage_impact
from power_benchmark.results.replay.traces import ElementTraceBuilder, FlowDirectionTraceBuilder, TrafoLoadingTraceBuilder
from power_benchmark.results.replay.utils import ColorMapHelper


class NetworkFigureBuilder:
    """Builds a Plotly figure for a pandapower network with power flow results."""

    _MAINT_RING = "#FFA500"

    def __init__(self, net: pp.pandapowerNet):
        self.net = net
        self._cmap_helper = ColorMapHelper()

        try:
            self.net.line_outage_impact = run_nminus1_with_line_outage_impact(self.net)
        except Exception:
            self.net.line_outage_impact = {}

    def build(self) -> go.Figure:
        """Build a Plotly figure from individual trace creators."""
        all_traces = []
        all_traces += self._build_line_traces()
        all_traces += self._build_trafo_traces()
        all_traces += self._build_bus_traces()

        fig = pplotly.draw_traces(
            all_traces,
            showlegend=True,
            figsize=1.0,
            aspectratio="auto",
            filename=None,
            auto_open=False,
        )

        # Add element traces (Load, Gen, SGen)
        for trace in ElementTraceBuilder(self.net).build():
            fig.add_trace(trace)

        # Add flow direction arrows
        for trace in FlowDirectionTraceBuilder(self.net).build():
            fig.add_trace(trace)

        # Add transformer loading markers
        for trace in TrafoLoadingTraceBuilder(self.net).build():
            fig.add_trace(trace)

        self._fix_colorbars(fig)

        fig.update_layout(width=900, height=700, hovermode="closest")
        return fig

    def _build_line_traces(self) -> list:
        if self.net.line.empty:
            return []

        line_info = self._build_line_hover_info()
        has_res = not self.net.res_line.empty
        if has_res:
            return pplotly.create_line_trace(
                self.net,
                lines=self.net.line.index,
                width=2,
                cmap=self._cmap_helper.get_line_loading_cmap(cmax=200),
                cmap_vals=self.net.res_line.loc[self.net.line.index, "loading_percent"].values,
                cmin=0,
                cmax=200,
                cbar_title="Line Loading [%]",
                trace_name="Lines",
                infofunc=line_info,
            )
        else:
            return pplotly.create_line_trace(
                self.net,
                lines=self.net.line.index,
                color="silver",
                width=2,
                trace_name="Lines",
                infofunc=line_info,
            )

    def _build_line_hover_info(self) -> pd.Series:
        """Build hover text for each line, including inverted N-1 impact if available."""
        nminus1_impact = getattr(self.net, "line_outage_impact", {})
        texts = {}

        for line_idx, line in self.net.line.iterrows():
            text = [
                f"<b>Line {line_idx}</b>",
                f"From bus: {line['from_bus']}",
                f"To bus: {line['to_bus']}",
                f"In service: {bool(line['in_service'])}",
            ]

            if not self.net.res_line.empty and line_idx in self.net.res_line.index:
                loading = self.net.res_line.at[line_idx, "loading_percent"]
                text.append(f"Loading: {self._format_percent(loading)}")

            impact = nminus1_impact.get(line_idx)
            if impact is None:
                text.append("N-1 outage: unavailable")
            else:
                limit_margin = 100.0 - impact["nminus1_loading"]
                if limit_margin >= 0:
                    limit_text = f"Capacity margin: {limit_margin:.1f} pp"
                else:
                    limit_text = f"Limit exceedance: {abs(limit_margin):.1f} pp"

                text.extend([
                    f"N-1 outage: worst affected line {impact['affected_line']}",
                    (
                        "Affected loading: "
                        f"{self._format_percent(impact['base_loading'])} -> "
                        f"{self._format_percent(impact['nminus1_loading'])} "
                        f"({impact['delta_loading']:+.1f} pp)"
                    ),
                    limit_text,
                ])

            texts[line_idx] = "<br>".join(text)

        return pd.Series(texts)

    @staticmethod
    def _format_percent(value: float) -> str:
        try:
            if pd.isna(value):
                return "n/a"
            return f"{float(value):.1f}%"
        except (TypeError, ValueError):
            return "n/a"

    def _build_trafo_traces(self) -> list:
        if self.net.trafo.empty:
            return []
        return pplotly.create_trafo_trace(
            self.net,
            trafos=self.net.trafo.index,
            color="dimgray",
            width=2,
            trace_name="Trafos",
        )

    @staticmethod
    def _maintenance_index(df) -> list:
        """Return indices of rows flagged for maintenance."""
        if "maintenance" not in df.columns:
            return []
        mask = df["maintenance"].fillna(False).astype(bool)
        return list(df.index[mask])

    def _bus_coords(self, bus_idx):
        """(x, y) from bus_geodata, or None if missing."""
        if bus_idx in self.net.bus_geodata.index:
            row = self.net.bus_geodata.loc[bus_idx]
            return row["x"], row["y"]
        return None, None

    def _build_bus_traces(self) -> list:
        if self.net.bus.empty:
            return []

        has_res_bus = not self.net.res_bus.empty
        if has_res_bus:
            traces = pplotly.create_bus_trace(
                self.net,
                buses=self.net.bus.index,
                size=10,
                cmap="jet",
                cmap_vals=self.net.res_bus.loc[self.net.bus.index, "vm_pu"].values,
                cmin=0.9,
                cmax=1.1,
                cpos=1.0,
                cbar_title="Bus Voltage [pu]",
                trace_name="Buses",
            )
        else:
            traces = pplotly.create_bus_trace(
                self.net,
                buses=self.net.bus.index,
                size=10,
                color="lightsteelblue",
                trace_name="Buses",
            )

        # ── Maintenance overlay ─────────────────────────────────────────
        overlay = self._build_bus_maintenance_overlay()
        if overlay is not None:
            traces = list(traces) + [overlay]

        return traces

    def _build_bus_maintenance_overlay(self):
        """Orange-ringed overlay marking buses in maintenance (gray = dead)."""
        maint_idx = self._maintenance_index(self.net.bus)
        if not maint_idx:
            return None

        xs, ys, texts = [], [], []
        for idx in maint_idx:
            x, y = self._bus_coords(idx)
            if x is None:
                continue
            vm = (self.net.res_bus.at[idx, "vm_pu"]
                  if idx in self.net.res_bus.index else float("nan"))
            vm_txt = f"{vm:.3f} pu" if not pd.isna(vm) else "n/a"
            xs.append(x)
            ys.append(y)
            texts.append(f"<b>Bus {idx}</b><br>Vm = {vm_txt}<br><b>🔧 In Maintenance</b>")

        if not xs:
            return None

        return go.Scatter(
            x=xs, y=ys,
            mode="markers",
            marker=dict(
                symbol="circle-open",
                size=18,  # größer als die 10er-Busse -> Ring liegt außen
                color=self._MAINT_RING,
                line=dict(width=3, color=self._MAINT_RING),
            ),
            name="Buses (Maintenance)",
            text=texts,
            hoverinfo="text",
            legendgroup="maintenance",
            showlegend=True,
        )

    def _fix_colorbars(self, fig: go.Figure):
        """Adjust colorbar positions and scales on the figure."""
        line_cmap = self._cmap_helper.get_line_loading_cmap(cmax=200)
        trafo_cmap = self._cmap_helper.get_line_loading_cmap(cmax=150)

        CBAR_THICKNESS = 14
        CBAR_LEN = 0.6
        CBAR_X = {
            "Buses": 1.01,
            "edge_center": 1.12,
            "trafo": 1.23,
        }
        CBAR_Y = {
            "Buses": 0.0,
            "edge_center": 0.0,
            "trafo": 0.0,
        }

        for trace in fig.data:
            try:
                if trace.name and trace.name.endswith("(Maintenance)"):
                    continue
                if trace.marker.colorbar is None:
                    continue

                if trace.name == "Lines":
                    trace.marker.showscale = False

                elif trace.name == "edge_center":
                    trace.marker.colorscale = self._cmap_helper.mpl_cmap_to_plotly(line_cmap)
                    trace.marker.cmin = 0
                    trace.marker.cmax = 200
                    trace.marker.colorbar.title = dict(text="Line Loading [%]", side="right")
                    trace.marker.colorbar.x = CBAR_X["edge_center"]
                    trace.marker.colorbar.xanchor = "left"
                    trace.marker.colorbar.y = CBAR_Y["edge_center"]
                    trace.marker.colorbar.yanchor = "bottom"
                    trace.marker.colorbar.thickness = CBAR_THICKNESS
                    trace.marker.colorbar.len = CBAR_LEN

                elif trace.name == "Buses":
                    trace.marker.colorbar.title = dict(text="Bus Voltage [p.u.]", side="right")
                    trace.marker.colorbar.x = CBAR_X["Buses"]
                    trace.marker.colorbar.xanchor = "left"
                    trace.marker.colorbar.y = CBAR_Y["Buses"]
                    trace.marker.colorbar.yanchor = "bottom"
                    trace.marker.colorbar.thickness = CBAR_THICKNESS
                    trace.marker.colorbar.len = CBAR_LEN

                # ── Trafo loading (shared 2W + 3W) ──────────────────────
                elif trace.name in ("Trafos", "trafo_center", "Trafo Loading"):
                    trace.marker.colorscale = self._cmap_helper.mpl_cmap_to_plotly(trafo_cmap)
                    trace.marker.cmin = 0
                    trace.marker.cmax = 150
                    trace.marker.colorbar.title = dict(text="Trafo Loading [%]", side="right")
                    trace.marker.colorbar.x = CBAR_X["trafo"]
                    trace.marker.colorbar.xanchor = "left"
                    trace.marker.colorbar.y = CBAR_Y["trafo"]
                    trace.marker.colorbar.yanchor = "bottom"
                    trace.marker.colorbar.thickness = CBAR_THICKNESS
                    trace.marker.colorbar.len = CBAR_LEN

            except Exception:
                continue

        # rechten Rand vergrößern und Legende neben die Bars legen
        fig.update_layout(
            margin=dict(r=255),
            legend=dict(x=1.50, xanchor="left", y=1.0),
        )
