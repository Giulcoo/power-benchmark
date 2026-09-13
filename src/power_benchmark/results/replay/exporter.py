from pathlib import Path
from typing import Optional, Sequence, Union

import pandapower as pp
import pandas as pd
import plotly.graph_objects as go

from power_benchmark.constants.replay import TABLE_CSS
from power_benchmark.results.replay.table_builder import HTMLTableBuilder


class HTMLExporter:
    """Exports combined Plotly figures and net DataFrames as an interactive HTML page."""

    def __init__(
        self,
        combined_fig_1: go.Figure,
        nets: list,
        combined_fig_2: Optional[go.Figure] = None,
        title_1: str = "Figure 1",
        title_2: str = "Figure 2",
    ):
        self.combined_fig_1 = combined_fig_1
        self.combined_fig_2 = combined_fig_2
        self.nets = nets
        self.title_1 = title_1
        self.title_2 = title_2
        self.table_builder = HTMLTableBuilder(nets)

    def export(self, output_path: str = "output.html"):
        """Write the full HTML file to disk."""
        plotly_html_1 = self.combined_fig_1.to_html(
            full_html=False,
            include_plotlyjs="cdn",
            div_id="plotly-div-1",
            config={"responsive": True},
        )

        plotly_html_2 = ""
        if self.combined_fig_2 is not None:
            plotly_html_2 = self.combined_fig_2.to_html(
                full_html=False,
                include_plotlyjs=False,
                div_id="plotly-div-2",
                config={"responsive": True},
            )

        panels_html = self.table_builder.build_panels()
        figures_row_style = self._get_figures_row_style()
        figure_sections = self._build_figure_sections(plotly_html_1, plotly_html_2)
        js = self._build_javascript()

        full_html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Network Results</title>
  {TABLE_CSS}
  {HTMLTableBuilder.SORT_CSS}
</head>
<body>
  <div class="figures-row" style="{figures_row_style}">
    {figure_sections}
  </div>
  <div id="net-tables" style="margin:24px 16px;">
    {panels_html}
  </div>
  {js}
  {HTMLTableBuilder.SORT_JS}
</body>
</html>"""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_html)

        print(f"Saved: {output_path}")

    def _get_figures_row_style(self) -> str:
        if self.combined_fig_2 is not None:
            return "display: flex; flex-wrap: wrap; gap: 16px; margin: 16px;"
        return "display: flex; flex-wrap: wrap; gap: 16px; margin: 16px; justify-content: center;"

    def _build_figure_sections(self, plotly_html_1: str, plotly_html_2: str) -> str:
        sections = f"""
    <div class="figure-section">
      <h2>{self.title_1}</h2>
      {plotly_html_1}
    </div>
"""
        if self.combined_fig_2 is not None:
            sections += f"""
    <div class="figure-section">
      <h2>{self.title_2}</h2>
      {plotly_html_2}
    </div>
"""
        return sections

    def _build_javascript(self) -> str:
        has_fig2 = self.combined_fig_2 is not None

        extra_div_id = ", 'plotly-div-2'" if has_fig2 else ""

        sync_from_fig1 = """
          if (!syncing && plotlyDiv2) {
            syncing = true;
            applySliderStep(plotlyDiv2, idx);
            syncing = false;
          }""" if has_fig2 else ""

        sync_from_fig2 = """
      if (plotlyDiv2) {
        plotlyDiv2.on('plotly_sliderchange', function(data) {
          var idx = parseInt(data.step.label);
          showPanel(idx);
          if (!syncing && plotlyDiv1) {
            syncing = true;
            applySliderStep(plotlyDiv1, idx);
            syncing = false;
          }
        });
      }""" if has_fig2 else ""

        table_js = self.table_builder.build_table_switcher_js()

        return f"""
    <script>
      var syncing = false;

      {table_js}

      // ── Apply slider step programmatically (visibility + title + slider pos) ─
      function applySliderStep(gd, idx) {{
        var sliders = gd._fullLayout.sliders;
        if (!sliders || !sliders[0]) return;
        var step = sliders[0].steps[idx];
        if (!step) return;
        var dataUpdate   = step.args[0];
        var layoutUpdate = Object.assign({{}}, step.args[1]);
        layoutUpdate['sliders[0].active'] = idx;
        Plotly.update(gd, dataUpdate, layoutUpdate);
      }}

      // ── Force correct width on load ──────────────────────────────────────────
      window.addEventListener('load', function () {{
        var ids = ['plotly-div-1'{extra_div_id}];
        ids.forEach(function(id) {{
          var el = document.getElementById(id);
          if (el) {{ Plotly.relayout(el, {{autosize: true}}); }}
        }});
      }});

      // ── Slider event wiring ──────────────────────────────────────────────────
      var plotlyDiv1 = document.getElementById('plotly-div-1');
      var plotlyDiv2 = document.getElementById('plotly-div-2');

      if (plotlyDiv1) {{
        plotlyDiv1.on('plotly_sliderchange', function(data) {{
          var idx = parseInt(data.step.label);
          showPanel(idx);
          {sync_from_fig1}
        }});
      }}

      {sync_from_fig2}
    </script>
    """


class CSVExportBuilder:
    """Exports pandapower net DataFrames to one CSV per table, all timesteps stacked."""

    SECTION_FILENAMES = {
        "Buses": "bus",
        "Lines": "line",
        "Trafo": "trafo",
        "Trafo3W": "trafo3w",
        "Loads": "load",
        "Gens": "gen",
        "Sgens": "sgen",
        "Ext. Grids": "ext_grid",
        "Res. Buses": "res_bus",
        "Res. Lines": "res_line",
        "Res. Trafo": "res_trafo",
        "Res. Trafo3W": "res_trafo3w",
        "Res. Load": "res_load",
        "Res. Gen": "res_gen",
        "Res. Sgen": "res_sgen",
        "Res. Ext. Grids": "res_ext_grid",
    }

    def __init__(
        self,
        nets: list,
        timestamps: Optional[Sequence] = None,
        sep: str = ";",
        decimal: str = ".",
        float_format: str = "%.3f",
        encoding: str = "utf-8",
    ):
        if timestamps is not None and len(timestamps) != len(nets):
            raise ValueError("timestamps must have the same length as nets")
        self.nets = nets
        self.timestamps = timestamps
        self.sep = sep
        self.decimal = decimal
        self.float_format = float_format
        self.encoding = encoding

    @property
    def num_nets(self) -> int:
        return len(self.nets)

    # ---------- public API ----------

    def build_files(self, output_dir: Union[str, Path]) -> list[Path]:
        """Write one CSV per table into output_dir, containing all timesteps."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        written = []
        for name, df in self.build_frames().items():
            path = output_dir / f"{self.SECTION_FILENAMES.get(name, name)}.csv"
            self._write_csv(df, path)
            written.append(path)
        return written

    def build_zip(self, zip_path: Union[str, Path]) -> Path:
        """Write all table CSVs into a single ZIP archive."""
        import zipfile

        zip_path = Path(zip_path)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, df in self.build_frames().items():
                fname = f"{self.SECTION_FILENAMES.get(name, name)}.csv"
                zf.writestr(fname, self._to_csv_string(df))
        return zip_path

    def build_frames(self) -> dict[str, pd.DataFrame]:
        """Return {section name: concatenated DataFrame over all timesteps}."""
        buckets: dict[str, list[pd.DataFrame]] = {}
        summary_rows: list[dict] = []

        for i, net in enumerate(self.nets):
            prev_net = self.nets[i - 1] if i > 0 else None
            sections, pl_summary = self._collect_sections(net, prev_net)

            for name, df in sections.items():
                buckets.setdefault(name, []).append(self._tag(df, i, index_name="element"))

            summary_rows.append({**self._meta(i), **pl_summary})

        frames = {name: pd.concat(parts, ignore_index=True) for name, parts in buckets.items()}
        if summary_rows:
            frames["Summary"] = pd.DataFrame(summary_rows)
            self.SECTION_FILENAMES.setdefault("Summary", "summary")
        return frames

    # ---------- internals ----------

    def _meta(self, timestep: int) -> dict:
        meta = {"timestep": timestep}
        if self.timestamps is not None:
            meta["timestamp"] = self.timestamps[timestep]
        return meta

    def _tag(self, df: pd.DataFrame, timestep: int, index_name: str) -> pd.DataFrame:
        """Move the element index into a column and prepend timestep columns."""
        out = df.copy()
        out.insert(0, index_name, out.index)
        for key, val in reversed(list(self._meta(timestep).items())):
            out.insert(0, key, val)
        return out.reset_index(drop=True)

    def _collect_sections(
        self,
        net: pp.pandapowerNet,
        prev_net: Optional[pp.pandapowerNet],
    ) -> tuple[dict[str, pd.DataFrame], dict]:
        res_line_df, pl_summary = self._build_res_line_with_deltas(net, prev_net)
        res_ext_grid_df = self._build_res_ext_grid_with_deltas(net, prev_net)

        sections = {
            "Buses": net.bus,
            "Lines": net.line,
            "Trafo": net.trafo,
            "Trafo3W": net.trafo3w,
            "Loads": net.load,
            "Gens": net.gen,
            "Sgens": net.sgen,
            "Ext. Grids": net.ext_grid,
            "Res. Buses": net.res_bus,
            "Res. Lines": res_line_df,
            "Res. Trafo": net.res_trafo,
            "Res. Trafo3W": net.res_trafo3w,
            "Res. Load": net.res_load,
            "Res. Gen": net.res_gen,
            "Res. Sgen": net.res_sgen,
            "Res. Ext. Grids": res_ext_grid_df,
        }
        sections = {k: v for k, v in sections.items() if v is not None and not v.empty}
        return sections, pl_summary

    @staticmethod
    def _build_res_line_with_deltas(
        net: pp.pandapowerNet,
        prev_net: Optional[pp.pandapowerNet],
    ) -> tuple[pd.DataFrame, dict]:
        res_line = net.res_line.copy()

        if prev_net is not None and not prev_net.res_line.empty:
            prev_res = prev_net.res_line.reindex(res_line.index)
            res_line["delta_p_from_mw"] = res_line["p_from_mw"] - prev_res["p_from_mw"]
            res_line["delta_q_from_mvar"] = res_line["q_from_mvar"] - prev_res["q_from_mvar"]
        else:
            res_line["delta_p_from_mw"] = float("nan")
            res_line["delta_q_from_mvar"] = float("nan")

        pl_mw_sum = net.res_line["pl_mw"].sum() if "pl_mw" in net.res_line.columns else 0.0
        if (
            prev_net is not None
            and not prev_net.res_line.empty
            and "pl_mw" in prev_net.res_line.columns
        ):
            delta_pl_mw = pl_mw_sum - prev_net.res_line["pl_mw"].sum()
        else:
            delta_pl_mw = float("nan")

        return res_line, {"sum_pl_mw": pl_mw_sum, "delta_pl_mw": delta_pl_mw}

    @staticmethod
    def _build_res_ext_grid_with_deltas(
        net: pp.pandapowerNet,
        prev_net: Optional[pp.pandapowerNet],
    ) -> pd.DataFrame:
        res_ext_grid = net.res_ext_grid.copy()

        if prev_net is not None and not prev_net.res_ext_grid.empty:
            prev_res = prev_net.res_ext_grid.reindex(res_ext_grid.index)
            res_ext_grid["delta_p_mw"] = res_ext_grid["p_mw"] - prev_res["p_mw"]
            res_ext_grid["delta_q_mvar"] = res_ext_grid["q_mvar"] - prev_res["q_mvar"]
        else:
            res_ext_grid["delta_p_mw"] = float("nan")
            res_ext_grid["delta_q_mvar"] = float("nan")

        return res_ext_grid

    def _write_csv(self, df: pd.DataFrame, path: Path) -> None:
        df.to_csv(
            path,
            sep=self.sep,
            decimal=self.decimal,
            float_format=self.float_format,
            encoding=self.encoding,
            index=False,
        )

    def _to_csv_string(self, df: pd.DataFrame) -> str:
        return df.to_csv(
            sep=self.sep,
            decimal=self.decimal,
            float_format=self.float_format,
            index=False,
        )