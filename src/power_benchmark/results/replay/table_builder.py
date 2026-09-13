from typing import Optional

import pandas as pd
import pandapower as pp


class HTMLTableBuilder:
    """Builds sortable, switchable HTML tables from pandapower net DataFrames."""

    SORT_CSS = """
    <style>
      .net-table th {
        cursor: pointer;
        user-select: none;
        position: relative;
        padding-right: 18px !important;
      }
      .net-table th::after {
        content: '⇅';
        position: absolute;
        right: 4px;
        opacity: 0.3;
        font-size: 12px;
      }
      .net-table th.sort-asc::after {
        content: '▲';
        opacity: 0.8;
      }
      .net-table th.sort-desc::after {
        content: '▼';
        opacity: 0.8;
      }
    </style>
    """

    SORT_JS = """
    <script>
      function sortTable(th) {
        var table = th.closest('table');
        var colIdx = Array.from(th.parentNode.children).indexOf(th);
        var tbody = table.querySelector('tbody') || table;
        var rows = Array.from(tbody.querySelectorAll('tr'));

        // Determine sort direction
        var isAsc = th.classList.contains('sort-asc');
        // Clear all sort classes in this header row
        Array.from(th.parentNode.children).forEach(function(h) {
          h.classList.remove('sort-asc', 'sort-desc');
        });

        var direction = isAsc ? -1 : 1;
        th.classList.add(isAsc ? 'sort-desc' : 'sort-asc');

        rows.sort(function(a, b) {
          var cellA = a.children[colIdx];
          var cellB = b.children[colIdx];
          if (!cellA || !cellB) return 0;
          var valA = cellA.textContent.trim();
          var valB = cellB.textContent.trim();
          var numA = parseFloat(valA);
          var numB = parseFloat(valB);
          if (!isNaN(numA) && !isNaN(numB)) {
            return (numA - numB) * direction;
          }
          return valA.localeCompare(valB) * direction;
        });

        rows.forEach(function(row) {
          tbody.appendChild(row);
        });
      }

      // Attach click handlers to all .net-table th elements
      function attachSortHandlers() {
        document.querySelectorAll('.net-table th').forEach(function(th) {
          th.addEventListener('click', function() {
            sortTable(th);
          });
        });
      }

      window.addEventListener('load', function() {
        attachSortHandlers();
      });
    </script>
    """

    def __init__(self, nets: list):
        self.nets = nets

    @property
    def num_nets(self) -> int:
        return len(self.nets)

    def build_panels(self) -> str:
        """Build all net panel HTML with switchable tables."""
        panels_html = ""
        for i, net in enumerate(self.nets):
            active = "active" if i == 0 else ""
            panel_id = f"net-panel-{i}"
            prev_net = self.nets[i - 1] if i > 0 else None
            panels_html += (
                f'<div class="net-panel {active}" id="{panel_id}">'
                f"<h3>Net {i} – DataFrames</h3>"
                f"{self._net_to_html_tables(net, panel_id, prev_net)}"
                f"</div>"
            )
        return panels_html

    def _net_to_html_tables(
        self,
        net: pp.pandapowerNet,
        panel_id: str,
        prev_net: Optional[pp.pandapowerNet] = None,
    ) -> str:
        """Render relevant DataFrames of a net as switchable HTML tables."""
        res_line_df, pl_mw_summary = self._build_res_line_with_deltas(net, prev_net)
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
        sections = {k: v for k, v in sections.items() if not v.empty}
        if not sections:
            return "<p>No data available.</p>"

        select_id = f"sel-{panel_id}"
        container_id = f"tbl-{panel_id}"

        options = "".join(
            f'<option value="{i}">{k}</option>' for i, k in enumerate(sections)
        )

        table_divs = ""
        for i, (k, df) in enumerate(sections.items()):
            display = "block" if i == 0 else "none"
            table_html = df.to_html(
                classes="net-table",
                border=0,
                float_format=lambda x: f"{x:.4f}",
            )
            extra_content = ""
            if k == "Res. Lines":
                extra_content = pl_mw_summary

            table_divs += (
                f'<div id="{container_id}-{i}" style="display:{display};overflow-x:auto;">'
                f"{extra_content}{table_html}</div>"
            )

        return f"""
<label for="{select_id}"><b>Show table:</b></label>
<select id="{select_id}"
        onchange="switchTable('{container_id}', this.value)"
        style="margin:8px;padding:4px 8px;font-size:13px;">
  {options}
</select>
<div id="{container_id}">
  {table_divs}
</div>
"""

    @staticmethod
    def _build_res_line_with_deltas(
        net: pp.pandapowerNet,
        prev_net: Optional[pp.pandapowerNet],
    ) -> tuple[pd.DataFrame, str]:
        """Build an enriched res_line DataFrame with delta columns and a pl_mw summary."""
        res_line = net.res_line.copy()

        if prev_net is not None and not prev_net.res_line.empty:
            prev_res = prev_net.res_line
            res_line["delta_p_from_mw"] = res_line["p_from_mw"] - prev_res["p_from_mw"]
            res_line["delta_q_from_mvar"] = (
                res_line["q_from_mvar"] - prev_res["q_from_mvar"]
            )
        else:
            res_line["delta_p_from_mw"] = float("nan")
            res_line["delta_q_from_mvar"] = float("nan")

        pl_mw_sum = (
            net.res_line["pl_mw"].sum() if "pl_mw" in net.res_line.columns else 0.0
        )
        if (
            prev_net is not None
            and not prev_net.res_line.empty
            and "pl_mw" in prev_net.res_line.columns
        ):
            prev_pl_mw_sum = prev_net.res_line["pl_mw"].sum()
            delta_pl_mw = pl_mw_sum - prev_pl_mw_sum
            summary_html = (
                f'<div class="pl-mw-summary" style="margin:8px 0;padding:8px 12px;'
                f'background:#f4f7fa;border-left:4px solid #4a90d9;font-size:13px;">'
                f"<b>Σ pl_mw:</b> {pl_mw_sum:.4f} MW &nbsp;|&nbsp; "
                f"<b>Δ vs prev:</b> {delta_pl_mw:+.4f} MW"
                f"</div>"
            )
        else:
            summary_html = (
                f'<div class="pl-mw-summary" style="margin:8px 0;padding:8px 12px;'
                f'background:#f4f7fa;border-left:4px solid #4a90d9;font-size:13px;">'
                f"<b>Σ pl_mw:</b> {pl_mw_sum:.4f} MW &nbsp;|&nbsp; "
                f"<b>Δ vs prev:</b> N/A (first timestep)"
                f"</div>"
            )

        return res_line, summary_html

    @staticmethod
    def _build_res_ext_grid_with_deltas(
        net: pp.pandapowerNet,
        prev_net: Optional[pp.pandapowerNet],
    ) -> pd.DataFrame:
        """Build an enriched res_ext_grid DataFrame with per-element delta columns."""
        res_ext_grid = net.res_ext_grid.copy()

        if prev_net is not None and not prev_net.res_ext_grid.empty:
            prev_res = prev_net.res_ext_grid
            res_ext_grid["delta_p_mw"] = (
                res_ext_grid["p_mw"] - prev_res["p_mw"]
            )
            res_ext_grid["delta_q_mvar"] = (
                res_ext_grid["q_mvar"] - prev_res["q_mvar"]
            )
        else:
            res_ext_grid["delta_p_mw"] = float("nan")
            res_ext_grid["delta_q_mvar"] = float("nan")

        return res_ext_grid

    def build_table_switcher_js(self) -> str:
        """Return the JavaScript for table switching and panel sync."""
        return f"""
      var numNets = {self.num_nets};
      var currentTableIdx = 0;

      function switchTable(containerId, idx) {{
        currentTableIdx = parseInt(idx);
        var container = document.getElementById(containerId);
        if (!container) return;
        var divs = container.children;
        for (var i = 0; i < divs.length; i++) {{
          divs[i].style.display = (i === currentTableIdx) ? 'block' : 'none';
        }}
      }}

      function syncTableSelection(panelIdx) {{
        var panelId  = 'net-panel-' + panelIdx;
        var selEl    = document.getElementById('sel-' + panelId);
        var contId   = 'tbl-' + panelId;
        if (selEl) {{
          var maxIdx = selEl.options.length - 1;
          var target = Math.min(currentTableIdx, maxIdx);
          selEl.value = target;
          switchTable(contId, target);
        }}
      }}

      function showPanel(idx) {{
        for (var k = 0; k < numNets; k++) {{
          var el = document.getElementById('net-panel-' + k);
          if (el) {{ el.classList.toggle('active', k === idx); }}
        }}
        syncTableSelection(idx);
      }}
"""