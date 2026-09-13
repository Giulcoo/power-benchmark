import copy

import pandapower as pp
import plotly.graph_objects as go

from pandapower_env.substation.plot_double_busbar_substation import separate_substation_buses_visually
from pandapower_env.toolbox.plotting_helpers import create_xy_columns_from_geo, create_bus2bus_geodata
from power_benchmark.results.replay.utils import CoordinateHelper

class SubstationStateTracker:
    """
    Tracks the current topological state of every substation across iterations.
    Call update() after each action to persist splits/joins.
    """

    def __init__(self, net: pp.pandapowerNet):
        self._current_states: dict[int, str] = {}
        for i_sub in net.multi_bb_substation.index:
            nbits = len(net.multi_bb_substation.loc[i_sub, "connected_elements"])
            # Default: all elements on busbar 0 → fully coupled
            self._current_states[i_sub] = "1" * nbits

    def update(self, action_dict: dict) -> None:
        """Apply the latest action's states to the tracker."""
        for i_sub, state in zip(
            action_dict.get("substations", []),
            action_dict.get("states", []),
        ):
            self._current_states[i_sub] = state

    def get_full_layout_dict(self) -> dict:
        """
        Return an action_dict-shaped dict covering ALL substations
        with their current (possibly persisted) states.
        Used exclusively for visual positioning.
        """
        subs = list(self._current_states.keys())
        states = [self._current_states[s] for s in subs]
        return {"substations": subs, "states": states}

class ActionFigureBuilder:
    """
    Builds a Plotly figure visualizing a network topology action,
    with highlights showing what changed (split substations, opened/closed switches).
    """

    def __init__(
            self,
            net: pp.pandapowerNet,
            action_dict: dict,
            layout_action_dict: dict | None = None,  # ← NEW
            vthresh: float = 0.0,
            r_split_bus: float = 0.1,
            bus_size: int = 8,
    ):
        self.net = net
        self.action_dict = action_dict
        self.vthresh = vthresh
        self.r_split_bus = r_split_bus
        self.bus_size = bus_size

        self._coord_helper = CoordinateHelper()
        self._orig_net = copy.deepcopy(net)

        # Use full cumulative state for visual layout; fall back to action_dict
        _layout_dict = layout_action_dict if layout_action_dict is not None else action_dict

        bus_geo, self._line_geos = separate_substation_buses_visually(
            net, _layout_dict, r_split_bus=r_split_bus
        )
        self._bus_geo = bus_geo
        self._bus_geo_xy = create_xy_columns_from_geo(bus_geo)

        # Map bus → substation name
        self._bus_to_sub = {}
        for i_sub in net.multi_bb_substation.index:
            bus0 = net.multi_bb_substation.loc[i_sub, "bus_0"]
            bus1 = net.multi_bb_substation.loc[i_sub, "bus_1"]
            self._bus_to_sub[bus0] = f"sub {i_sub}.0"
            self._bus_to_sub[bus1] = f"sub {i_sub}.1"

        # Parse action details
        self._affected_subs = action_dict.get("substations", [])
        self._states = action_dict.get("states", [])
        self._open_switches = action_dict.get("open_switches", [])
        self._closed_switches = action_dict.get("closed_switches", [])
        self._action_number = action_dict.get("action", "?")

        # Compute offsets based on network extent
        y_range = self._bus_geo_xy["y"].max() - self._bus_geo_xy["y"].min()
        x_range = self._bus_geo_xy["x"].max() - self._bus_geo_xy["x"].min()
        self._y_offset = y_range * 0.03
        self._x_offset = x_range * 0.02
        self._y_range = y_range
        self._x_range = x_range

        # Decode substation split info
        self._sub_split_info = self._decode_split_info()

    def build(self) -> go.Figure:
        """Generate the complete action visualization figure."""
        traces = []

        traces += self._build_line_traces()
        traces += self._build_trafo_traces()
        traces += self._build_bus_traces()
        traces += self._build_substation_bus_traces()
        traces += self._build_ext_grid_traces()
        traces += self._build_load_traces()
        traces += self._build_load_label_traces()
        traces += self._build_generator_traces()
        traces += self._build_generator_label_traces()
        traces += self._build_bus_label_traces()
        traces += self._build_line_label_traces()
        traces += self._build_trafo_label_traces()
        traces += self._build_substation_label_traces()

        # Action highlights
        traces += self._build_affected_substation_highlights()
        traces += self._build_open_coupler_traces()
        traces += self._build_split_detail_annotations()
        traces += self._build_open_switch_traces()
        traces += self._build_closed_switch_traces()

        fig = go.Figure(data=traces)
        fig.update_layout(**self._build_layout())
        return fig

    # ── Private: Action Parsing ────────────────────────────────────────────

    def _decode_split_info(self) -> dict:
        """Decode which elements go to which bus for each affected substation."""
        sub_split_info = {}
        for idx, i_sub in enumerate(self._affected_subs):
            if i_sub not in self.net.multi_bb_substation.index:
                continue
            state_str = self._states[idx] if idx < len(self._states) else ""
            if state_str.startswith("0x"):
                state_str = state_str[2:]

            row = self.net.multi_bb_substation.loc[i_sub]
            element_types = row["element_type"]
            connected_elements = row["connected_elements"]

            bus0_elems, bus1_elems = [], []
            for bit_idx, bit in enumerate(state_str):
                if bit_idx >= len(element_types):
                    break
                elem_desc = f"{element_types[bit_idx]} {connected_elements[bit_idx]}"
                if bit == "1":
                    bus0_elems.append(elem_desc)
                else:
                    bus1_elems.append(elem_desc)

            sub_split_info[i_sub] = {
                "bus0_elements": bus0_elems,
                "bus1_elements": bus1_elems,
                "state_str": state_str,
                "is_split": "0" in state_str and "1" in state_str,
            }
        return sub_split_info

    # ── Private: Network Element Traces ────────────────────────────────────

    def _build_line_traces(self) -> list:
        line_indices = self.net.line.index[
            self.net.line["from_bus"].map(self.net.bus["vn_kv"]) >= self.vthresh
        ]
        lx, ly = [], []
        for ln in line_indices:
            if ln not in self._line_geos.index:
                continue
            for c in self._coord_helper.extract_coords(self._line_geos[ln]):
                lx.append(c[0])
                ly.append(c[1])
            lx.append(None)
            ly.append(None)

        return [go.Scatter(
            x=lx, y=ly,
            mode="lines",
            line=dict(color="silver", width=1),
            hoverinfo="skip",
            showlegend=False,
            name="Lines",
        )]

    def _build_trafo_traces(self) -> list:
        trafo_indices = self.net.trafo.index[self.net.trafo["vn_lv_kv"] >= self.vthresh]
        tx, ty = [], []
        for i_trafo in trafo_indices:
            lv_bus = self.net.trafo.loc[i_trafo, "lv_bus"]
            hv_bus = self.net.trafo.loc[i_trafo, "hv_bus"]
            if lv_bus in self._bus_geo_xy.index and hv_bus in self._bus_geo_xy.index:
                tx += [self._bus_geo_xy.loc[lv_bus, "x"], self._bus_geo_xy.loc[hv_bus, "x"], None]
                ty += [self._bus_geo_xy.loc[lv_bus, "y"], self._bus_geo_xy.loc[hv_bus, "y"], None]

        return [go.Scatter(
            x=tx, y=ty,
            mode="lines",
            line=dict(color="dimgray", width=2),
            hoverinfo="skip",
            showlegend=False,
            name="Trafos",
        )]

    def _build_bus_traces(self) -> list:
        bus_indices = self.net.bus.index[self.net.bus["vn_kv"] >= self.vthresh]
        return [go.Scatter(
            x=self._bus_geo_xy.loc[bus_indices, "x"].tolist(),
            y=self._bus_geo_xy.loc[bus_indices, "y"].tolist(),
            mode="markers",
            marker=dict(color="lightsteelblue", size=self.bus_size),
            customdata=bus_indices.tolist(),
            hovertemplate="Bus %{customdata}<extra></extra>",
            showlegend=False,
            name="Buses",
        )]

    def _build_substation_bus_traces(self) -> list:
        sub_buses = self.net.multi_bb_substation[["bus_0", "bus_1"]].values.flatten().tolist()
        return [go.Scatter(
            x=self._bus_geo_xy.loc[sub_buses, "x"].tolist(),
            y=self._bus_geo_xy.loc[sub_buses, "y"].tolist(),
            mode="markers",
            marker=dict(color="darksalmon", size=self.bus_size * 2),
            hoverinfo="skip",
            showlegend=False,
            name="Substation Buses",
        )]

    def _build_ext_grid_traces(self) -> list:
        ext_buses = self.net.ext_grid["bus"].tolist()
        return [go.Scatter(
            x=self._bus_geo_xy.loc[ext_buses, "x"].tolist(),
            y=self._bus_geo_xy.loc[ext_buses, "y"].tolist(),
            mode="markers",
            marker=dict(color="yellow", size=self.bus_size * 2, symbol="square"),
            hoverinfo="skip",
            showlegend=False,
            name="Ext Grid",
        )]

    def _build_load_traces(self) -> list:
        load_buses = self.net.load["bus"]
        load_buses_valid = load_buses[load_buses.map(self.net.bus["vn_kv"]) >= self.vthresh]
        load_x, load_y, load_text = [], [], []

        for i_load in load_buses_valid.index:
            bus = self.net.load.loc[i_load, "bus"]
            if bus not in self._bus_geo_xy.index:
                continue
            bx = self._bus_geo_xy.loc[bus, "x"]
            by = self._bus_geo_xy.loc[bus, "y"] - self._y_offset
            load_x.append(bx)
            load_y.append(by)

            p_mw = self.net.load.loc[i_load, "p_mw"]
            q_mvar = (self.net.load.loc[i_load, "q_mvar"]
                      if "q_mvar" in self.net.load.columns else 0)
            connected_to = self._bus_to_sub.get(bus, f"bus {bus}")
            load_text.append(
                f"Load {i_load}<br>"
                f"P={p_mw:.1f} MW, Q={q_mvar:.1f} MVAr<br>"
                f"@ {connected_to}"
            )

        # Store for label building
        self._load_x = load_x
        self._load_y = load_y
        self._load_buses_valid = load_buses_valid

        return [go.Scatter(
            x=load_x, y=load_y,
            mode="markers",
            marker=dict(color="green", size=self.bus_size * 1.5, symbol="triangle-down"),
            customdata=load_buses_valid.index.tolist(),
            hovertemplate="%{text}<extra></extra>",
            text=load_text,
            showlegend=False,
            name="Loads",
        )]

    def _build_load_label_traces(self) -> list:
        load_labels = []
        for i_load in self._load_buses_valid.index:
            bus = self.net.load.loc[i_load, "bus"]
            connected_to = self._bus_to_sub.get(bus, f"bus {bus}")
            load_labels.append(f"L{i_load} @ {connected_to}")

        load_label_y = [y - 0.04 for y in self._load_y]
        return [go.Scatter(
            x=self._load_x, y=load_label_y,
            mode="text",
            text=load_labels,
            textposition="bottom center",
            textfont=dict(size=9, color="green"),
            hoverinfo="skip",
            showlegend=True,
            legendgroup="Load Labels",
            name="Load Labels",
            visible="legendonly",
        )]

    def _build_generator_traces(self) -> list:
        gen_x, gen_y, gen_text, gen_labels = [], [], [], []

        if len(self.net.gen) > 0:
            gen_buses = self.net.gen["bus"]
            gen_buses_valid = gen_buses[gen_buses.map(self.net.bus["vn_kv"]) >= self.vthresh]

            for i_gen in gen_buses_valid.index:
                bus = self.net.gen.loc[i_gen, "bus"]
                if bus not in self._bus_geo_xy.index:
                    continue
                bx = self._bus_geo_xy.loc[bus, "x"]
                by = self._bus_geo_xy.loc[bus, "y"] + self._y_offset
                gen_x.append(bx)
                gen_y.append(by)

                p_mw = self.net.gen.loc[i_gen, "p_mw"]
                connected_to = self._bus_to_sub.get(bus, f"bus {bus}")
                gen_text.append(
                    f"Gen {i_gen}<br>P={p_mw:.1f} MW<br>@ {connected_to}"
                )
                gen_labels.append(f"G{i_gen} @ {connected_to}")

        if len(self.net.sgen) > 0:
            sgen_buses = self.net.sgen["bus"]
            sgen_buses_valid = sgen_buses[sgen_buses.map(self.net.bus["vn_kv"]) >= self.vthresh]

            for i_sgen in sgen_buses_valid.index:
                bus = self.net.sgen.loc[i_sgen, "bus"]
                if bus not in self._bus_geo_xy.index:
                    continue
                bx = self._bus_geo_xy.loc[bus, "x"]
                by = self._bus_geo_xy.loc[bus, "y"] + self._y_offset
                gen_x.append(bx)
                gen_y.append(by)

                p_mw = self.net.sgen.loc[i_sgen, "p_mw"]
                connected_to = self._bus_to_sub.get(bus, f"bus {bus}")
                gen_text.append(
                    f"SGen {i_sgen}<br>P={p_mw:.1f} MW<br>@ {connected_to}"
                )
                gen_labels.append(f"SG{i_sgen} @ {connected_to}")

        # Store for label building
        self._gen_x = gen_x
        self._gen_y = gen_y
        self._gen_labels = gen_labels

        return [go.Scatter(
            x=gen_x, y=gen_y,
            mode="markers",
            marker=dict(color="red", size=self.bus_size * 1.5, symbol="triangle-up"),
            hovertemplate="%{text}<extra></extra>",
            text=gen_text,
            showlegend=False,
            name="Generators",
        )]

    def _build_generator_label_traces(self) -> list:
        gen_label_y = [y + 0.04 for y in self._gen_y]
        return [go.Scatter(
            x=self._gen_x, y=gen_label_y,
            mode="text",
            text=self._gen_labels,
            textposition="top center",
            textfont=dict(size=9, color="red"),
            hoverinfo="skip",
            showlegend=True,
            legendgroup="Gen Labels",
            name="Gen Labels",
            visible="legendonly",
        )]

    def _build_bus_label_traces(self) -> list:
        sub_bus_set = set(
            self.net.multi_bb_substation[["bus_0", "bus_1"]].values.flatten().tolist()
        )
        non_sub_buses = [b for b in self._orig_net.bus.index if b not in sub_bus_set]

        return [go.Scatter(
            x=self._bus_geo_xy.loc[non_sub_buses, "x"].tolist(),
            y=self._bus_geo_xy.loc[non_sub_buses, "y"].tolist(),
            mode="text",
            text=[f"bus {b}" for b in non_sub_buses],
            textposition="middle center",
            hoverinfo="skip",
            showlegend=True,
            legendgroup="Bus Labels",
            name="Bus Labels",
            visible="legendonly",
        )]

    def _build_line_label_traces(self) -> list:
        line_b2b = create_bus2bus_geodata(
            self.net,
            self.net.line["from_bus"],
            self.net.line["to_bus"],
            bus_geodata=self._bus_geo,
        )
        return [go.Scatter(
            x=line_b2b["x_midpoint"].tolist(),
            y=line_b2b["y_midpoint"].tolist(),
            mode="text",
            text=[str(ln) for ln in self.net.line.index],
            textposition="middle center",
            hoverinfo="skip",
            showlegend=True,
            legendgroup="Line Labels",
            name="Line Labels",
            visible=True,
        )]

    def _build_trafo_label_traces(self) -> list:
        tx, ty, tt = [], [], []

        # ── 2-winding trafos ────────────────────────────────────────────
        trafo_indices = self.net.trafo.index[self.net.trafo["vn_lv_kv"] >= self.vthresh]
        for i_trafo in trafo_indices:
            lv_bus = self.net.trafo.loc[i_trafo, "lv_bus"]
            hv_bus = self.net.trafo.loc[i_trafo, "hv_bus"]
            if lv_bus in self._bus_geo_xy.index and hv_bus in self._bus_geo_xy.index:
                tx.append((self._bus_geo_xy.loc[lv_bus, "x"] + self._bus_geo_xy.loc[hv_bus, "x"]) / 2)
                ty.append((self._bus_geo_xy.loc[lv_bus, "y"] + self._bus_geo_xy.loc[hv_bus, "y"]) / 2)
                tt.append(f"T{i_trafo}")

        # ── 3-winding trafos ────────────────────────────────────────────
        if hasattr(self.net, "trafo3w") and len(self.net.trafo3w) > 0:
            for i_trafo in self.net.trafo3w.index:
                buses = [
                    self.net.trafo3w.loc[i_trafo, b]
                    for b in ("hv_bus", "mv_bus", "lv_bus")
                ]
                buses = [b for b in buses if b in self._bus_geo_xy.index]
                if not buses:
                    continue
                tx.append(sum(self._bus_geo_xy.loc[b, "x"] for b in buses) / len(buses))
                ty.append(sum(self._bus_geo_xy.loc[b, "y"] for b in buses) / len(buses))
                tt.append(f"T3W{i_trafo}")

        if not tt:
            return []

        return [go.Scatter(
            x=tx, y=ty,
            mode="text",
            text=tt,
            textposition="middle center",
            textfont=dict(size=10, color="dimgray"),
            hoverinfo="skip",
            showlegend=True,
            legendgroup="Trafo Labels",
            name="Trafo Labels",
            visible="legendonly",
        )]

    def _build_substation_label_traces(self) -> list:
        slx, sly, slt = [], [], []
        for i_sub in self.net.multi_bb_substation.index:
            bus0 = self.net.multi_bb_substation.loc[i_sub, "bus_0"]
            bus1 = self.net.multi_bb_substation.loc[i_sub, "bus_1"]
            b0 = self._bus_geo_xy.loc[bus0, ["x", "y"]].tolist()
            b1 = self._bus_geo_xy.loc[bus1, ["x", "y"]].tolist()
            if b0 == b1:
                slx.append(b0[0])
                sly.append(b0[1])
                slt.append(f"sub {i_sub}")
            else:
                slx += [b0[0], b1[0]]
                sly += [b0[1], b1[1]]
                slt += [f"sub {i_sub}.0", f"sub {i_sub}.1"]

        return [go.Scatter(
            x=slx, y=sly,
            mode="text",
            text=slt,
            textfont=dict(size=12, color="black"),
            textposition="top center",
            hoverinfo="skip",
            showlegend=True,
            legendgroup="Sub Labels",
            name="Sub Labels",
            visible=True,
        )]

    # ── Private: Action Highlight Traces ───────────────────────────────────

    def _build_affected_substation_highlights(self) -> list:
        highlight_x, highlight_y, highlight_text = [], [], []
        for i_sub in self._affected_subs:
            if i_sub not in self.net.multi_bb_substation.index:
                continue
            bus0 = self.net.multi_bb_substation.loc[i_sub, "bus_0"]
            bus1 = self.net.multi_bb_substation.loc[i_sub, "bus_1"]
            for bus in [bus0, bus1]:
                if bus in self._bus_geo_xy.index:
                    highlight_x.append(self._bus_geo_xy.loc[bus, "x"])
                    highlight_y.append(self._bus_geo_xy.loc[bus, "y"])

            info = self._sub_split_info.get(i_sub, {})
            hover = (
                f"<b>Sub {i_sub} — {'SPLIT' if info.get('is_split') else 'COUPLED'}</b><br>"
                f"State: {info.get('state_str', '?')}<br>"
                f"<b>Bus 0:</b> {', '.join(info.get('bus0_elements', []))}<br>"
                f"<b>Bus 1:</b> {', '.join(info.get('bus1_elements', []))}"
            )
            highlight_text.append(hover)
            highlight_text.append(hover)

        return [go.Scatter(
            x=highlight_x, y=highlight_y,
            mode="markers",
            marker=dict(
                color="rgba(255, 0, 0, 0)",
                size=self.bus_size * 4,
                line=dict(color="red", width=3),
                symbol="circle",
            ),
            hovertemplate="%{text}<extra></extra>",
            text=highlight_text,
            showlegend=True,
            legendgroup="Action Highlights",
            name="⚡ Affected Substations",
            visible=True,
        )]

    def _build_open_coupler_traces(self) -> list:
        coupler_x, coupler_y = [], []
        for i_sub in self._affected_subs:
            if i_sub not in self.net.multi_bb_substation.index:
                continue
            info = self._sub_split_info.get(i_sub, {})
            if not info.get("is_split"):
                continue
            bus0 = self.net.multi_bb_substation.loc[i_sub, "bus_0"]
            bus1 = self.net.multi_bb_substation.loc[i_sub, "bus_1"]
            if bus0 in self._bus_geo_xy.index and bus1 in self._bus_geo_xy.index:
                coupler_x += [
                    self._bus_geo_xy.loc[bus0, "x"],
                    self._bus_geo_xy.loc[bus1, "x"],
                    None,
                ]
                coupler_y += [
                    self._bus_geo_xy.loc[bus0, "y"],
                    self._bus_geo_xy.loc[bus1, "y"],
                    None,
                ]

        if not coupler_x:
            return []

        return [go.Scatter(
            x=coupler_x, y=coupler_y,
            mode="lines",
            line=dict(color="red", width=2, dash="dash"),
            hoverinfo="skip",
            showlegend=True,
            legendgroup="Action Highlights",
            name="⚡ Open Couplers",
            visible=True,
        )]

    def _build_split_detail_annotations(self) -> list:
        anno_x, anno_y, anno_text = [], [], []
        for i_sub in self._affected_subs:
            if i_sub not in self.net.multi_bb_substation.index:
                continue
            info = self._sub_split_info.get(i_sub, {})
            if not info.get("is_split"):
                continue

            bus0 = self.net.multi_bb_substation.loc[i_sub, "bus_0"]
            bus1 = self.net.multi_bb_substation.loc[i_sub, "bus_1"]

            if bus0 in self._bus_geo_xy.index and info["bus0_elements"]:
                anno_x.append(self._bus_geo_xy.loc[bus0, "x"] - self._x_offset)
                anno_y.append(self._bus_geo_xy.loc[bus0, "y"] - self._y_offset * 1.5)
                elems_short = [
                    e.replace("line", "L").replace("load", "Ld")
                    .replace("gen", "G").replace("trafo", "T")
                    for e in info["bus0_elements"]
                ]
                anno_text.append(f"BB0: {', '.join(elems_short)}")

            if bus1 in self._bus_geo_xy.index and info["bus1_elements"]:
                anno_x.append(self._bus_geo_xy.loc[bus1, "x"] + self._x_offset)
                anno_y.append(self._bus_geo_xy.loc[bus1, "y"] - self._y_offset * 1.5)
                elems_short = [
                    e.replace("line", "L").replace("load", "Ld")
                    .replace("gen", "G").replace("trafo", "T")
                    for e in info["bus1_elements"]
                ]
                anno_text.append(f"BB1: {', '.join(elems_short)}")

        if not anno_text:
            return []

        return [go.Scatter(
            x=anno_x, y=anno_y,
            mode="text",
            text=anno_text,
            textfont=dict(size=10, color="darkred"),
            textposition="bottom center",
            hoverinfo="skip",
            showlegend=True,
            legendgroup="Action Highlights",
            name="⚡ Split Details",
            visible=True,
        )]

    def _build_open_switch_traces(self) -> list:
        if not self._open_switches:
            return []

        sw_x, sw_y, sw_text = [], [], []
        for sw_idx in self._open_switches:
            if sw_idx not in self.net.switch.index:
                continue
            sw = self.net.switch.loc[sw_idx]
            bus = sw["bus"]
            element = sw["element"]
            et = sw["et"]

            if et == "b":
                if bus in self._bus_geo_xy.index and element in self._bus_geo_xy.index:
                    mx = (self._bus_geo_xy.loc[bus, "x"] + self._bus_geo_xy.loc[element, "x"]) / 2
                    my = (self._bus_geo_xy.loc[bus, "y"] + self._bus_geo_xy.loc[element, "y"]) / 2
                    sw_x.append(mx)
                    sw_y.append(my)
                    sw_text.append(f"SW {sw_idx} OPEN<br>bus-bus {bus}↔{element}")
            elif et == "l":
                if element in self.net.line.index:
                    from_bus = self.net.line.loc[element, "from_bus"]
                    to_bus = self.net.line.loc[element, "to_bus"]
                    if from_bus in self._bus_geo_xy.index and to_bus in self._bus_geo_xy.index:
                        other_bus = from_bus if bus != from_bus else to_bus
                        mx = self._bus_geo_xy.loc[bus, "x"] * 0.7 + self._bus_geo_xy.loc[other_bus, "x"] * 0.3
                        my = self._bus_geo_xy.loc[bus, "y"] * 0.7 + self._bus_geo_xy.loc[other_bus, "y"] * 0.3
                        sw_x.append(mx)
                        sw_y.append(my)
                        sw_text.append(f"SW {sw_idx} OPEN<br>line {element}")

        if not sw_x:
            return []

        return [go.Scatter(
            x=sw_x, y=sw_y,
            mode="markers",
            marker=dict(
                color="red",
                size=self.bus_size * 1.8,
                symbol="x",
                line=dict(width=2),
            ),
            hovertemplate="%{text}<extra></extra>",
            text=sw_text,
            showlegend=True,
            legendgroup="Action Highlights",
            name="⚡ Opened Switches",
            visible=True,
        )]

    def _build_closed_switch_traces(self) -> list:
        if not self._closed_switches:
            return []

        csw_x, csw_y, csw_text = [], [], []
        for sw_idx in self._closed_switches:
            if sw_idx not in self.net.switch.index:
                continue
            sw = self.net.switch.loc[sw_idx]
            bus = sw["bus"]
            element = sw["element"]
            et = sw["et"]

            if et == "b":
                if bus in self._bus_geo_xy.index and element in self._bus_geo_xy.index:
                    mx = (self._bus_geo_xy.loc[bus, "x"] + self._bus_geo_xy.loc[element, "x"]) / 2
                    my = (self._bus_geo_xy.loc[bus, "y"] + self._bus_geo_xy.loc[element, "y"]) / 2
                    csw_x.append(mx)
                    csw_y.append(my)
                    csw_text.append(f"SW {sw_idx} CLOSED<br>bus-bus {bus}↔{element}")
            elif et == "l":
                if element in self.net.line.index:
                    from_bus = self.net.line.loc[element, "from_bus"]
                    to_bus = self.net.line.loc[element, "to_bus"]
                    if from_bus in self._bus_geo_xy.index and to_bus in self._bus_geo_xy.index:
                        other_bus = from_bus if bus != from_bus else to_bus
                        mx = self._bus_geo_xy.loc[bus, "x"] * 0.7 + self._bus_geo_xy.loc[other_bus, "x"] * 0.3
                        my = self._bus_geo_xy.loc[bus, "y"] * 0.7 + self._bus_geo_xy.loc[other_bus, "y"] * 0.3
                        csw_x.append(mx)
                        csw_y.append(my)
                        csw_text.append(f"SW {sw_idx} CLOSED<br>line {element}")

        if not csw_x:
            return []

        return [go.Scatter(
            x=csw_x, y=csw_y,
            mode="markers",
            marker=dict(
                color="limegreen",
                size=self.bus_size * 1.8,
                symbol="diamond",
                line=dict(width=2, color="darkgreen"),
            ),
            hovertemplate="%{text}<extra></extra>",
            text=csw_text,
            showlegend=True,
            legendgroup="Action Highlights",
            name="⚡ Closed Switches",
            visible=True,
        )]

    # ── Private: Layout ────────────────────────────────────────────────────

    def _build_layout(self) -> dict:
        # Title
        title_lines = [f"<b>Action {self._action_number}</b>"]
        for i_sub in self._affected_subs:
            info = self._sub_split_info.get(i_sub, {})
            status = "SPLIT" if info.get("is_split") else "COUPLED"
            title_lines.append(
                f"Sub {i_sub}: {status} (state={info.get('state_str', '?')})"
            )
        if self._open_switches:
            title_lines.append(f"Opened switches: {self._open_switches}")
        if self._closed_switches:
            title_lines.append(f"Closed switches: {self._closed_switches}")

        # Shapes (rectangles around affected substations)
        shapes = []
        for i_sub in self._affected_subs:
            if i_sub not in self.net.multi_bb_substation.index:
                continue
            bus0 = self.net.multi_bb_substation.loc[i_sub, "bus_0"]
            bus1 = self.net.multi_bb_substation.loc[i_sub, "bus_1"]

            xs, ys = [], []
            for bus in [bus0, bus1]:
                if bus in self._bus_geo_xy.index:
                    xs.append(self._bus_geo_xy.loc[bus, "x"])
                    ys.append(self._bus_geo_xy.loc[bus, "y"])

            if xs:
                pad_x = self._x_range * 0.04
                pad_y = self._y_range * 0.04
                shapes.append(dict(
                    type="rect",
                    x0=min(xs) - pad_x,
                    y0=min(ys) - pad_y,
                    x1=max(xs) + pad_x,
                    y1=max(ys) + pad_y,
                    line=dict(color="red", width=2, dash="dot"),
                    fillcolor="rgba(255, 200, 200, 0.15)",
                    layer="below",
                ))

        return dict(
            title=dict(
                text="<br>".join(title_lines),
                font=dict(size=13),
                x=0.01, xanchor="left",
            ),
            width=900,
            height=900,
            plot_bgcolor="white",
            showlegend=True,
            legend=dict(
                title="Toggle Labels",
                itemclick="toggle",
                itemdoubleclick="toggleothers",
            ),
            xaxis=dict(showgrid=True, zeroline=False),
            yaxis=dict(showgrid=True, zeroline=False, scaleanchor="x", scaleratio=1),
            shapes=shapes,
        )