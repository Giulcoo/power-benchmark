from typing import Optional, List, Literal, Dict, Any

import pandas as pd

import pandapower as pp

from power_benchmark.configs.range_config import RangeConfig
import logging
logger = logging.getLogger("Benchmarker")


class NetworkModifier:
    """
    Modifies a pandapower network:
      - Scale generation (gen, sgen, ext_grid)
      - Scale loads
      - Scale capacity of lines
      - Adds maintenance
    """

    def __init__(self, modifications: Dict | List[Dict]) -> None:
        self.modifications = modifications

    def modify(self, net: pp.pandapowerNet, episode_length: int) -> pp.pandapowerNet:
        if isinstance(self.modifications, dict):
            return self.modify_from_dict(self.modifications, net, episode_length)
        elif isinstance(self.modifications, list):
            for modification in self.modifications:
                net = self.modify_from_dict(modification, net, episode_length)
        return net

    def modify_from_dict(self, data: Dict[str, Any], net: pp.pandapowerNet, episode_length: int) -> pp.pandapowerNet:
        if "scale_loads" in data:
            params = data["scale_loads"]
            preserve_total = params.pop("preserve_total", True)
            if preserve_total:
                self.scale_loads_preserve_total(net, **params)
            else:
                self.scale_loads(net, **params)
        if "scale_generation" in data:
            params = data["scale_generation"]
            preserve_total = params.pop("preserve_total", True)
            if preserve_total:
                self.scale_generation_preserve_total(net, **params)
            else:
                self.scale_generation(net, **params)
        if "scale_line_capacity" in data:
            self.scale_line_capacity(net, **data["scale_line_capacity"])
        if "scale_trafo_capacity" in data:
            self.scale_trafo_capacity(net, **data["scale_trafo_capacity"])
        if len(maintenance_data := {key: data for key, data in data.items() if key.endswith("maintenance")}) > 0:
            self.add_maintenance(net, episode_length, True, **maintenance_data)

        return net


    @staticmethod
    def scale_loads(
        net: pp.pandapowerNet,
        scale_p: float = 1.0,
        scale_q: float = 1.0,
        load_indices: Optional[List[int]] = None,
    ) -> None:
        """
        Multiply active (p_mw) and/or reactive (q_mvar) power of loads.

        Args:
            net:           pandapower network to modify.
            scale_p:       Scaling factor for active power.
            scale_q:       Scaling factor for reactive power.
            load_indices:  List of load indices to scale. None → all loads.
        """
        idx = load_indices if load_indices is not None else net.load.index
        net.load.loc[idx, "p_mw"] *= scale_p
        net.load.loc[idx, "q_mvar"] *= scale_q
        logger.debug(f"[scale_loads] p×{scale_p}, q×{scale_q} applied to loads {list(idx)}")


    @staticmethod
    def scale_generation(
        net: pp.pandapowerNet,
        scale: float = 1.0,
        element_types: Optional[List[Literal['gen', 'sgen', 'ext_grid']]] = None,
        element_indices: Optional[Dict[Literal['gen', 'sgen', 'ext_grid'], List[int]]] = None,
    ) -> None:
        """
        Multiply active power of generators.

        Args:
            net:             pandapower network to modify.
            scale:           Scaling factor.
            element_types:   Subset of ['gen', 'sgen', 'ext_grid']. None → all three.
            element_indices: Dict {element_type: [indices]}. None → all rows per type.

        Examples:
            modifier.scale_generation(scale=0.8)
            modifier.scale_generation(scale=1.2, element_types=['sgen'],
                                       element_indices={'sgen': [0, 2]})
        """
        type_col_map = {
            "gen":      ("p_mw",),
            "sgen":     ("p_mw",),
            "ext_grid": (),          # ext_grid has no fixed p_mw setpoint
        }
        if element_types is None:
            element_types = ["gen", "sgen", "ext_grid"]

        for etype in element_types:
            table = getattr(net, etype, None)
            if table is None or table.empty:
                continue

            idx = (
                element_indices[etype]
                if (element_indices and etype in element_indices)
                else table.index.tolist()
            )

            missing = [i for i in idx if i not in table.index]
            if missing:
                raise ValueError(
                    f"[scale_generation] '{etype}' has no rows with index label(s) {missing}. "
                    f"Existing index: {table.index.tolist()}"
                )

            cols = type_col_map.get(etype, ("p_mw",))
            if not cols:
                logger.debug(f"[scale_generation] '{etype}' has no p_mw column, skipped.")
                continue

            for col in cols:
                if col in table.columns:
                    net[etype].loc[idx, col] *= scale

            logger.debug(f"[scale_generation] '{etype}' p_mw×{scale} applied to idx {list(idx)}")

    @staticmethod
    def scale_loads_preserve_total(
            net: pp.pandapowerNet,
            scale_p: float = 1.0,
            scale_q: float = 1.0,
            load_indices: Optional[List[int]] = None,
    ) -> None:
        """
        Scale specified loads while adjusting remaining loads to preserve
        the total p_mw and q_mvar sum across all loads.

        Args:
            net:           pandapower network to modify.
            scale_p:       Scaling factor for active power of specified loads.
            scale_q:       Scaling factor for reactive power of specified loads.
            load_indices:  Loads to scale. Must be a strict subset of all load indices.
        """
        if not load_indices:
            logger.debug("[scale_loads_preserve_total] No load_indices given, nothing to do.")
            return

        all_idx = net.load.index.tolist()
        other_idx = [i for i in all_idx if i not in load_indices]

        if not other_idx:
            logger.warning("[scale_loads_preserve_total] load_indices covers all loads — falling back to plain scaling (no compensation possible).")
            net.load.loc[load_indices, "p_mw"] *= scale_p
            net.load.loc[load_indices, "q_mvar"] *= scale_q
            return

        for col, scale, label in [("p_mw", scale_p, "p"), ("q_mvar", scale_q, "q")]:
            total = net.load[col].sum()
            scaled_sum = net.load.loc[load_indices, col].sum() * scale
            remaining = total - scaled_sum
            other_sum = net.load.loc[other_idx, col].sum()

            net.load.loc[load_indices, col] *= scale

            if other_sum == 0:
                logger.warning(
                    f"[scale_loads_preserve_total] Sum of other loads {col} is 0 — "
                    "compensation skipped."
                )
                continue

            comp_scale = remaining / other_sum
            net.load.loc[other_idx, col] *= comp_scale
            logger.debug(
                f"[scale_loads_preserve_total] {label}: specified×{scale}, "
                f"others×{comp_scale:.6f} (total preserved: {total:.3f} MW)"
            )

    @staticmethod
    def scale_generation_preserve_total(
            net: pp.pandapowerNet,
            scale: float = 1.0,
            element_types: Optional[List[Literal["gen", "sgen", "ext_grid"]]] = None,
            element_indices: Optional[Dict[Literal["gen", "sgen", "ext_grid"], List[int]]] = None,
    ) -> None:
        """
        Scale specified generators while adjusting remaining generators of the
        same type to preserve the total p_mw sum per element type.

        ext_grid is skipped (no p_mw setpoint).

        Args:
            net:             pandapower network to modify.
            scale:           Scaling factor for specified generators.
            element_types:   Types to process. None → ['gen', 'sgen'].
            element_indices: Dict {etype: [indices to scale]}.
                             None → all rows per type (no compensation possible).
        """
        scalable_types = {"gen": "p_mw", "sgen": "p_mw"}

        if element_types is None:
            element_types = list(scalable_types.keys())

        for etype in element_types:
            if etype not in scalable_types:
                logger.debug(f"[scale_generation_preserve_total] '{etype}' has no p_mw setpoint, skipped.")
                continue

            table = getattr(net, etype, None)
            if table is None or table.empty:
                continue

            col = scalable_types[etype]
            all_idx = table.index.tolist()
            scaled_idx = (
                element_indices[etype]
                if (element_indices and etype in element_indices)
                else all_idx
            )

            missing = [i for i in scaled_idx if i not in table.index]
            if missing:
                raise ValueError(
                    f"[scale_generation_preserve_total] '{etype}' has no index label(s) {missing}. "
                    f"Existing: {all_idx}"
                )

            other_idx = [i for i in all_idx if i not in scaled_idx]

            if not other_idx:
                logger.warning(
                    f"[scale_generation_preserve_total] '{etype}': scaled_idx covers all rows — "
                    "falling back to plain scaling."
                )
                net[etype].loc[scaled_idx, col] *= scale
                continue

            total = table[col].sum()
            scaled_sum = table.loc[scaled_idx, col].sum() * scale
            remaining = total - scaled_sum
            other_sum = table.loc[other_idx, col].sum()

            net[etype].loc[scaled_idx, col] *= scale

            if other_sum == 0:
                logger.warning(
                    f"[scale_generation_preserve_total] '{etype}': sum of other generators is 0 — "
                    "compensation skipped."
                )
                continue

            comp_scale = remaining / other_sum
            net[etype].loc[other_idx, col] *= comp_scale
            logger.debug(
                f"[scale_generation_preserve_total] '{etype}': specified×{scale}, "
                f"others×{comp_scale:.6f} (total preserved: {total:.3f} MW)"
            )

    @staticmethod
    def scale_line_capacity(
            net: pp.pandapowerNet,
            scale: float = 1.0,
            line_indices: Optional[List[int]] = None,
    ) -> None:
        """
        Multiply the maximum current (max_i_ka) of lines.

        A scale < 1 reduces capacity → lines overload faster.
        A scale > 1 increases capacity → lines overload slower.

        Args:
            net:           pandapower network to modify.
            scale:         Scaling factor for max_i_ka.
            line_indices:  List of line indices to scale. None → all lines.
        """
        if net.line.empty:
            logger.debug("[scale_line_capacity] No lines in network, skipped.")
            return

        idx = line_indices if line_indices is not None else net.line.index.tolist()

        missing = [i for i in idx if i not in net.line.index]
        if missing:
            raise ValueError(
                f"[scale_line_capacity] net.line has no rows with index label(s) {missing}. "
                f"Existing index: {net.line.index.tolist()}"
            )

        net.line.loc[idx, "max_i_ka"] *= scale
        logger.debug(f"[scale_line_capacity] max_i_ka×{scale} applied to lines {list(idx)}")

    @staticmethod
    def scale_trafo_capacity(
            net: pp.pandapowerNet,
            scale: float = 1.0,
            trafo_indices: Optional[List[int]] = None,
            trafo3w_indices: Optional[List[int]] = None,
    ) -> None:
        """
        Multiply the rated apparent power (sn_mva) of transformers.

        A scale < 1 reduces capacity → transformers overload faster.
        A scale > 1 increases capacity → transformers overload slower.

        For two-winding transformers, scales 'sn_mva'.
        For three-winding transformers, scales 'sn_hv_mva', 'sn_mv_mva', 'sn_lv_mva'.

        Args:
            net:             pandapower network to modify.
            scale:           Scaling factor for rated power.
            trafo_indices:   List of trafo indices to scale. None → all trafos.
            trafo3w_indices: List of trafo3w indices to scale. None → all trafo3ws.
        """
        # --- Two-winding transformers ---
        if not net.trafo.empty:
            idx = trafo_indices if trafo_indices is not None else net.trafo.index.tolist()

            missing = [i for i in idx if i not in net.trafo.index]
            if missing:
                raise ValueError(
                    f"[scale_trafo_capacity] net.trafo has no rows with index label(s) {missing}. "
                    f"Existing index: {net.trafo.index.tolist()}"
                )

            net.trafo.loc[idx, "sn_mva"] *= scale
            logger.debug(f"[scale_trafo_capacity] sn_mva×{scale} applied to trafos {list(idx)}")
        else:
            if trafo_indices is not None:
                raise ValueError(
                    "[scale_trafo_capacity] trafo_indices specified but net.trafo is empty."
                )
            logger.debug("[scale_trafo_capacity] No two-winding transformers in network, skipped.")

        # --- Three-winding transformers ---
        if not net.trafo3w.empty:
            idx3w = trafo3w_indices if trafo3w_indices is not None else net.trafo3w.index.tolist()

            missing3w = [i for i in idx3w if i not in net.trafo3w.index]
            if missing3w:
                raise ValueError(
                    f"[scale_trafo_capacity] net.trafo3w has no rows with index label(s) {missing3w}. "
                    f"Existing index: {net.trafo3w.index.tolist()}"
                )

            for col in ("sn_hv_mva", "sn_mv_mva", "sn_lv_mva"):
                net.trafo3w.loc[idx3w, col] *= scale
            logger.debug(
                f"[scale_trafo_capacity] sn_hv/mv/lv_mva×{scale} applied to trafo3ws {list(idx3w)}"
            )
        else:
            if trafo3w_indices is not None:
                raise ValueError(
                    "[scale_trafo_capacity] trafo3w_indices specified but net.trafo3w is empty."
                )
            logger.debug("[scale_trafo_capacity] No three-winding transformers in network, skipped.")

    @staticmethod
    def add_maintenance(net: pp.pandapowerNet, episode_length: int,
        day_indexes: bool = True,
        line_maintenance: Optional[Dict[int, RangeConfig]] = None,
        switch_maintenance: Optional[Dict[int, RangeConfig]] = None,
        trafo_maintenance: Optional[Dict[int, RangeConfig]] = None,
        trafo3w_maintenance: Optional[Dict[int, RangeConfig]] = None,
        gen_maintenance: Optional[Dict[int, RangeConfig]] = None,
        sgen_maintenance: Optional[Dict[int, RangeConfig]] = None,
    ) -> None:
        """
        Add line maintenance periods to the network's profiles.

        Args:
            net:                pandapower network to modify.
            episode_length:     Number of time steps in one episode/day.
            day_indexes:        Whether the indexes represent whole episodes/days
            line_maintenance:   Dict {line_index: RangeConfig} specifying maintenance periods for lines.
            switch_maintenance: Dict {switch_index: RangeConfig} specifying maintenance periods for switches.
            trafo_maintenance:  Dict {trafo_index: RangeConfig} specifying maintenance periods for transformers.
            trafo3w_maintenance: Dict {trafo3w_index: RangeConfig} specifying maintenance periods for 3-winding transformers.
            gen_maintenance:    Dict {gen_index: RangeConfig} specifying maintenance periods for generators.
            sgen_maintenance:   Dict {sgen_index: RangeConfig} specifying maintenance periods for static generators.
        """
        profile_time = NetworkModifier.get_profile_timesteps(net)

        def fill_maintenance(net_df: pd.DataFrame, profile_key: str, maintenance: Dict[int, RangeConfig]) -> None:
            # Initialize DataFrame: all False, rows=timesteps, cols=line indices
            maintenance_df = pd.DataFrame(False, index=profile_time.index, columns=net_df.index, dtype=bool)

            for element_index, rg in maintenance.items():
                if isinstance(rg, Dict):
                    rg: RangeConfig = RangeConfig(**rg)

                indexes: List[int] = rg.get_timesteps(episode_length) if day_indexes else rg.indexes

                col_pos = maintenance_df.columns.get_loc(element_index)
                maintenance_df.iloc[indexes, col_pos] = True

            net.profiles[profile_key] = maintenance_df
            net_df["maintenance"] = False

        if line_maintenance:
            fill_maintenance(net.line, "line_maintenance", line_maintenance)
        if switch_maintenance:
            fill_maintenance(net.switch, "switch_maintenance", switch_maintenance)
        if trafo_maintenance:
            fill_maintenance(net.trafo, "trafo_maintenance", trafo_maintenance)
        if trafo3w_maintenance:
            fill_maintenance(net.trafo3w, "trafo3w_maintenance", trafo3w_maintenance)
        if gen_maintenance:
            fill_maintenance(net.gen, "gen_maintenance", gen_maintenance)
        if sgen_maintenance:
            fill_maintenance(net.sgen, "sgen_maintenance", sgen_maintenance)


    @staticmethod
    def get_profile_timesteps(net: pp.pandapowerNet) -> pd.Series:
        profile_keys = [
            "load", "powerplants", "renewable", "gen_vm", "sgen_q",
            "line_maintenance", "switch_maintenance", "trafo_maintenance",
            "trafo3w_maintenance", "gen_maintenance", "sgen_maintenance"
        ]

        profile_times: List[pd.Series] = [net.profiles[key]["time"] for key in profile_keys if key in net.profiles and "time" in net.profiles[key]]

        if len(profile_keys) == 0:
            raise ValueError("No profiles found in the network. Expected at least one of: " + ", ".join(profile_keys))

        return profile_times.pop()
