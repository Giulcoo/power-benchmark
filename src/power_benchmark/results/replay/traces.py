from typing import Optional, List

import math
import pandapower as pp
import plotly.graph_objects as go
import matplotlib
import matplotlib.colors as mcolors

from power_benchmark.results.replay.utils import CoordinateHelper, ColorMapHelper


class ElementTraceBuilder:
    """Builds Plotly traces for network elements (loads, generators, sgens)."""

    #  -> dark-red  (loads)
    _LOAD_CMAP = mcolors.LinearSegmentedColormap.from_list(
        "load_cmap", ["#C6FA87", "#004A00"]
    )
    # orange -> dark-red  (generators)
    _GEN_CMAP = mcolors.LinearSegmentedColormap.from_list(
        "gen_cmap", ["#FFA500", "#8B0000"]
    )
    _FALLBACK_COLOR = "rgba(128,128,128,1.00)"

    def __init__(self, net: pp.pandapowerNet):
        self.net = net
        self._coord_helper = CoordinateHelper()

    def build(self) -> list:
        """Return Scatter traces for loads, gens, sgens and ext_grid."""
        traces = []

        load_res = self.net.res_load      if not self.net.res_load.empty      else None
        gen_res  = self.net.res_gen       if not self.net.res_gen.empty       else None
        sgen_res = self.net.res_sgen      if not self.net.res_sgen.empty      else None
        ext_res  = self.net.res_ext_grid  if not self.net.res_ext_grid.empty  else None

        # ── Ext-grid split: producers (p_mw ≥ 0) vs consumers (p_mw < 0) ──
        ext_prod_idx, ext_cons_idx = self._split_ext_grid(ext_res)

        # ── LOAD pool: loads + ext-grid consumers ──────────────────────────
        load_vals     = self._collect_p_vals(self.net.load, load_res) \
                        if not self.net.load.empty else {}
        ext_cons_vals = self._collect_abs_p(ext_cons_idx, ext_res)

        combined_load = {("load", k): v for k, v in load_vals.items()}
        combined_load.update({("ext_cons", k): v for k, v in ext_cons_vals.items()})

        load_colors     = {}
        ext_cons_colors = {}
        if combined_load:
            all_lc = self._normalize_and_colorize(combined_load, self._LOAD_CMAP)
            load_colors     = {k: all_lc[("load",     k)] for k in load_vals}
            ext_cons_colors = {k: all_lc[("ext_cons", k)] for k in ext_cons_vals}

        # ── GEN pool: gens + sgens + ext-grid producers ────────────────────
        gen_vals      = self._collect_p_vals(self.net.gen,  gen_res) \
                        if not self.net.gen.empty  else {}
        sgen_vals     = self._collect_p_vals(self.net.sgen, sgen_res) \
                        if not self.net.sgen.empty else {}
        ext_prod_vals = self._collect_abs_p(ext_prod_idx, ext_res)

        combined_gen = {("gen",      k): v for k, v in gen_vals.items()}
        combined_gen.update({("sgen",     k): v for k, v in sgen_vals.items()})
        combined_gen.update({("ext_prod", k): v for k, v in ext_prod_vals.items()})

        gen_colors      = {}
        sgen_colors     = {}
        ext_prod_colors = {}
        if combined_gen:
            all_gc = self._normalize_and_colorize(combined_gen, self._GEN_CMAP)
            gen_colors      = {k: all_gc[("gen",      k)] for k in gen_vals}
            sgen_colors     = {k: all_gc[("sgen",     k)] for k in sgen_vals}
            ext_prod_colors = {k: all_gc[("ext_prod", k)] for k in ext_prod_vals}

        # ── Build traces ───────────────────────────────────────────────────
        if not self.net.load.empty:
            traces.extend(self._build_trace(
                self.net.load["bus"], "Load", "arrow-down",
                load_colors, load_res, size=14,
                src_df=self.net.load,
            ))

        if not self.net.gen.empty:
            traces.extend(self._build_trace(
                self.net.gen["bus"], "Gen", "arrow-up",
                gen_colors, gen_res, size=14,
                src_df=self.net.gen,
            ))

        if not self.net.sgen.empty:
            traces.extend(self._build_trace(
                self.net.sgen["bus"], "SGen", "star",
                sgen_colors, sgen_res, size=14,
                src_df=self.net.sgen,
            ))

        if not self.net.ext_grid.empty:
            if ext_prod_idx:
                traces.extend(self._build_trace(
                    self.net.ext_grid.loc[ext_prod_idx, "bus"],
                    "Ext Grid (Gen)", "square",
                    ext_prod_colors, ext_res, size=20,
                    src_df=self.net.ext_grid,
                ))
            if ext_cons_idx:
                traces.extend(self._build_trace(
                    self.net.ext_grid.loc[ext_cons_idx, "bus"],
                    "Ext Grid (Load)", "square",
                    ext_cons_colors, ext_res, size=20,
                    src_df=self.net.ext_grid,
                ))

        return traces

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _split_ext_grid(self, ext_res) -> tuple[list, list]:
        """
        Split ext_grid indices into producers (p_mw >= 0) and consumers (p_mw < 0).
        Entries without results default to the producer side.
        """
        prod, cons = [], []
        for idx in self.net.ext_grid.index:
            if ext_res is not None and idx in ext_res.index:
                if ext_res.at[idx, "p_mw"] >= 0:
                    prod.append(idx)
                else:
                    cons.append(idx)
            else:
                prod.append(idx)   # no result → treat as producer (fallback)
        return prod, cons

    @staticmethod
    def _collect_p_vals(df, res_df) -> dict:
        """Return {idx: abs(p_mw)} for every row in df; NaN when result missing."""
        vals = {}
        for idx in df.index:
            if res_df is not None and idx in res_df.index:
                vals[idx] = abs(res_df.at[idx, "p_mw"])
            else:
                vals[idx] = float("nan")
        return vals

    @staticmethod
    def _collect_abs_p(indices: list, res_df) -> dict:
        """Return {idx: abs(p_mw)} for a plain list of indices."""
        vals = {}
        for idx in indices:
            if res_df is not None and idx in res_df.index:
                vals[idx] = abs(res_df.at[idx, "p_mw"])
            else:
                vals[idx] = float("nan")
        return vals

    @classmethod
    def _normalize_and_colorize(cls, vals: dict, cmap) -> dict:
        """
        Map {key: magnitude} → {key: 'rgba(…)'}.
        Lowest magnitude = lightest color, highest = darkest.
        NaN → fallback gray.
        """
        valid = [p for p in vals.values() if not math.isnan(p)]
        if not valid:
            return {k: cls._FALLBACK_COLOR for k in vals}

        p_min, p_max = min(valid), max(valid)
        span = p_max - p_min

        colors = {}
        for key, p in vals.items():
            if math.isnan(p):
                colors[key] = cls._FALLBACK_COLOR
            else:
                norm = 0.5 if span == 0 else (p - p_min) / span
                r, g, b, a = cmap(norm)
                colors[key] = (
                    f"rgba({r * 255:.1f},{g * 255:.1f},{b * 255:.1f},{a:.2f})"
                )
        return colors

    def _build_trace(self, bus_series, pp_name, pp_symbol,
                     color_map: dict, pp_res_df, size: int = 14,
                     src_df=None) -> List[go.Scatter]:
        xs, ys, texts, colors = [], [], [], []
        m_xs, m_ys, m_texts, m_colors = [], [], [], []

        for idx, bus in bus_series.items():
            x, y = self._coord_helper.get_bus_coords(self.net, bus)
            if x is None:
                continue
            p = (pp_res_df.at[idx, "p_mw"]
                 if (pp_res_df is not None and idx in pp_res_df.index)
                 else float("nan"))

            if src_df is not None and self._is_maintenance(src_df, idx):
                m_xs.append(x)
                m_ys.append(y)
                m_colors.append(color_map.get(idx, self._FALLBACK_COLOR))
                m_texts.append(f"{pp_name} {idx}<br>Bus: {bus}"
                               f"<br>P = {p:.2f} MW<br><b>🔧 In Maintenance</b>")
            else:
                xs.append(x)
                ys.append(y)
                colors.append(color_map.get(idx, self._FALLBACK_COLOR))
                texts.append(f"{pp_name} {idx}<br>Bus: {bus}<br>P = {p:.2f} MW")

        traces = []

        if xs:
            traces.append(go.Scatter(
                x=xs, y=ys, mode="markers",
                marker=dict(symbol=pp_symbol, size=size, color=colors,
                            line=dict(width=1, color="black")),
                name=pp_name, text=texts, hoverinfo="text",
            ))

        if m_xs:
            traces.append(go.Scatter(
                x=m_xs, y=m_ys, mode="markers",
                marker=dict(
                    symbol=pp_symbol,
                    size=size + 4,  # etwas größer
                    color=m_colors,  # gleiche Füllung
                    line=dict(width=3, color="#FFA500"),  # oranger Wartungs-Ring
                ),
                name=f"{pp_name} (Maintenance)",
                text=m_texts, hoverinfo="text",
                legendgroup="maintenance",
            ))

        return traces

    @staticmethod
    def _is_maintenance(df, idx) -> bool:
        """True if element idx is flagged for maintenance."""
        if "maintenance" in df.columns and idx in df.index:
            val = df.at[idx, "maintenance"]
            return bool(val) and not (isinstance(val, float) and math.isnan(val))
        return False


class TrafoLoadingTraceBuilder:
    """Builds markers at transformer positions, colored by loading_percent.

    Trafo (2W) and Trafo3W share one common loading scale and colorbar.
    Elements flagged for maintenance keep their loading color but get an
    orange ring as a unique indicator (gray is reserved for 'dead' elements).
    """

    _CMAX = 150  # shared loading scale for trafo + trafo3w
    _MAINT_RING = "#FFA500"

    def __init__(self, net: pp.pandapowerNet):
        self.net = net
        self._coord_helper = CoordinateHelper()
        self._cmap_helper = ColorMapHelper()

    @staticmethod
    def _is_maintenance(df, idx) -> bool:
        if "maintenance" in df.columns and idx in df.index:
            val = df.at[idx, "maintenance"]
            return bool(val) and not (isinstance(val, float) and math.isnan(val))
        return False

    def build(self) -> list:
        traces = []
        cbar_used = False  # ensures the shared colorbar is drawn only once

        # ── 2-winding transformers ──────────────────────────────────────
        if not self.net.trafo.empty and not self.net.res_trafo.empty:
            xs, ys, loading, texts = [], [], [], []
            m_xs, m_ys, m_loading, m_texts = [], [], [], []
            for idx, row in self.net.trafo.iterrows():
                x0, y0 = self._coord_helper.get_bus_coords(self.net, row["hv_bus"])
                x1, y1 = self._coord_helper.get_bus_coords(self.net, row["lv_bus"])
                if x0 is None or x1 is None or idx not in self.net.res_trafo.index:
                    continue
                l = self.net.res_trafo.at[idx, "loading_percent"]
                mx, my = (x0 + x1) / 2, (y0 + y1) / 2

                if self._is_maintenance(self.net.trafo, idx):
                    m_xs.append(mx); m_ys.append(my); m_loading.append(l)
                    m_texts.append(f"Trafo {idx}<br>HV bus: {row['hv_bus']}"
                                   f"<br>LV bus: {row['lv_bus']}"
                                   f"<br>Loading = {l:.1f}%<br><b>🔧 In Maintenance</b>")
                else:
                    xs.append(mx); ys.append(my); loading.append(l)
                    texts.append(f"Trafo {idx}<br>HV bus: {row['hv_bus']}"
                                 f"<br>LV bus: {row['lv_bus']}<br>Loading = {l:.1f}%")

            if xs:
                traces.append(self._make_marker_trace(
                    xs, ys, loading, texts, "Trafo Loading", show_cbar=not cbar_used))
                cbar_used = True
            if m_xs:
                traces.append(self._make_marker_trace(
                    m_xs, m_ys, m_loading, m_texts, "Trafo Loading (Maintenance)",
                    show_cbar=not cbar_used, maintenance=True))
                cbar_used = True

        # ── 3-winding transformers ──────────────────────────────────────
        if not self.net.trafo3w.empty and not self.net.res_trafo3w.empty:
            xs, ys, loading, texts = [], [], [], []
            m_xs, m_ys, m_loading, m_texts = [], [], [], []
            for idx, row in self.net.trafo3w.iterrows():
                coords = [self._coord_helper.get_bus_coords(self.net, row[b])
                          for b in ("hv_bus", "mv_bus", "lv_bus")]
                coords = [c for c in coords if c[0] is not None]
                if not coords or idx not in self.net.res_trafo3w.index:
                    continue
                l = self.net.res_trafo3w.at[idx, "loading_percent"]
                mx = sum(c[0] for c in coords) / len(coords)
                my = sum(c[1] for c in coords) / len(coords)

                if self._is_maintenance(self.net.trafo3w, idx):
                    m_xs.append(mx); m_ys.append(my); m_loading.append(l)
                    m_texts.append(f"Trafo3W {idx}<br>HV: {row['hv_bus']} "
                                   f"MV: {row['mv_bus']} LV: {row['lv_bus']}"
                                   f"<br>Loading = {l:.1f}%<br><b>🔧 In Maintenance</b>")
                else:
                    xs.append(mx); ys.append(my); loading.append(l)
                    texts.append(f"Trafo3W {idx}<br>HV: {row['hv_bus']} "
                                 f"MV: {row['mv_bus']} LV: {row['lv_bus']}"
                                 f"<br>Loading = {l:.1f}%")

            if xs:
                traces.append(self._make_marker_trace(
                    xs, ys, loading, texts, "Trafo3W Loading", show_cbar=not cbar_used))
                cbar_used = True
            if m_xs:
                traces.append(self._make_marker_trace(
                    m_xs, m_ys, m_loading, m_texts, "Trafo3W Loading (Maintenance)",
                    show_cbar=not cbar_used, maintenance=True))
                cbar_used = True

        return traces

    def _make_marker_trace(self, xs, ys, loading, texts, name,
                           show_cbar, maintenance=False):
        cmap_name = self._cmap_helper.get_line_loading_cmap(cmax=self._CMAX)
        colorscale = self._cmap_helper.mpl_cmap_to_plotly(cmap_name)
        return go.Scatter(
            x=xs, y=ys,
            mode="markers",
            marker=dict(
                symbol="hexagon",
                size=14 if maintenance else 10,          # etwas größer
                color=loading,
                colorscale=colorscale,
                cmin=0,
                cmax=self._CMAX,
                line=dict(
                    width=3 if maintenance else 1,
                    color=self._MAINT_RING if maintenance else "black",
                ),
                showscale=show_cbar,
                colorbar=dict(
                    title=dict(text="Trafo Loading [%]", side="right"),
                    x=1.32,
                    thickness=20,
                    len=0.75,
                ),
            ),
            name=name,
            text=texts,
            hoverinfo="text",
            legendgroup="maintenance" if maintenance else None,
        )

class FlowDirectionTraceBuilder:
    """Builds arrow markers at line midpoints indicating power flow direction."""

    _MAINT_RING = "#FFA500"

    def __init__(self, net: pp.pandapowerNet):
        self.net = net
        self._coord_helper = CoordinateHelper()
        self._cmap_helper = ColorMapHelper()

    @staticmethod
    def _is_maintenance(df, idx) -> bool:
        if "maintenance" in df.columns and idx in df.index:
            val = df.at[idx, "maintenance"]
            return bool(val) and not (isinstance(val, float) and math.isnan(val))
        return False

    def build(self) -> list:
        if self.net.line.empty or self.net.res_line.empty:
            return []

        xs, ys, angles, loading_vals, texts = [], [], [], [], []
        m_xs, m_ys, m_angles, m_loading, m_texts = [], [], [], [], []

        for idx, row in self.net.line.iterrows():
            x0, y0 = self._coord_helper.get_bus_coords(self.net, row["from_bus"])
            x1, y1 = self._coord_helper.get_bus_coords(self.net, row["to_bus"])
            if x0 is None or x1 is None:
                continue

            p_from = (self.net.res_line.at[idx, "p_from_mw"]
                      if idx in self.net.res_line.index else 0.0)
            loading = (self.net.res_line.at[idx, "loading_percent"]
                       if idx in self.net.res_line.index else float("nan"))

            line_angle = math.degrees(math.atan2(y1 - y0, x1 - x0))
            if p_from < 0:
                line_angle += 180
            plotly_angle = (90 - line_angle) % 360

            mx, my = (x0 + x1) / 2, (y0 + y1) / 2

            if self._is_maintenance(self.net.line, idx):
                m_xs.append(mx); m_ys.append(my); m_angles.append(plotly_angle)
                m_loading.append(loading)
                m_texts.append(f"Line {idx}<br>P = {p_from:.2f} MW"
                               f"<br>Loading = {loading:.1f}%<br><b>🔧 In Maintenance</b>")
            else:
                xs.append(mx); ys.append(my); angles.append(plotly_angle)
                loading_vals.append(loading)
                texts.append(f"Line {idx}<br>P = {p_from:.2f} MW"
                             f"<br>Loading = {loading:.1f}%")

        if not xs and not m_xs:
            return []

        cmap_name = self._cmap_helper.get_line_loading_cmap(cmax=200)
        colorscale = self._cmap_helper.mpl_cmap_to_plotly(cmap_name)

        def _arrow_trace(xs, ys, angles, loading, texts, name, maintenance):
            return go.Scatter(
                x=xs, y=ys, mode="markers",
                marker=dict(
                    symbol="arrow",
                    size=18 if maintenance else 14,
                    angleref="up",
                    angle=angles,
                    color=loading,
                    colorscale=colorscale,
                    cmin=0, cmax=200,
                    showscale=False,
                    line=dict(
                        width=3 if maintenance else 0,
                        color=self._MAINT_RING if maintenance else "black",
                    ),
                ),
                name=name, text=texts, hoverinfo="text",
                legendgroup="maintenance" if maintenance else None,
            )

        traces = []
        if xs:
            traces.append(_arrow_trace(xs, ys, angles, loading_vals,
                                       texts, "Flow Dir", False))
        if m_xs:
            traces.append(_arrow_trace(m_xs, m_ys, m_angles, m_loading,
                                       m_texts, "Flow Dir (Maintenance)", True))
        return traces