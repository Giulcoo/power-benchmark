import numpy as np
import pandas as pd
import pandapower as pp
from pandapower.powerflow import _powerflow

import logging

from pandapower_env.toolbox.utils import select_topk_line_contingencies

logger = logging.getLogger("Benchmarker")


def run_nminus1_with_line_outage_impact(
    net: pp.pandapowerNet,
    pf_type: str = "ac",
    use_ls2g: str | bool = "auto",
    topk_percent: float = 100.0,
) -> dict[int, dict[str, float | int]]:
    """
    Wie run_nminus1_powerflow, aber zusätzlich wird pro Line-Ausfall x die am
    höchsten belastete (überwachte) Line y samt Loading erfasst.

    Ergebnis (Rückgabewert):
        {
            outage_line_x: {
                "affected_line":    y,                 # am staerksten belastete Line
                "nminus1_loading":  loading_y_percent, # deren Loading beim Ausfall x
                "base_loading":     loading_y_n0,       # deren N-0 Loading
            },
            ...
        }

    Die üblichen run_contingency-Aggregate (res_line.max_loading_percent,
    cause_element, cause_index, ...) werden weiterhin auf ``net`` gespeichert.
    """
    if pf_type not in {"ac", "dc"}:
        msg = "pf_type must be 'ac' or 'dc'."
        raise ValueError(msg)

    if use_ls2g != "auto" and not isinstance(use_ls2g, bool):
        msg = "use_ls2g must be bool or 'auto'."
        raise ValueError(msg)

    nminus1_cases = {
        "line": {"index": net.line.index.to_numpy()},
        "trafo": {"index": net.trafo.index.to_numpy()},
        "trafo3w": {"index": net.trafo3w.index.to_numpy()},
    }

    # Optionen einmal setzen (warmer Power Flow); Contingencies nutzen sie via _powerflow.
    base_powerflow = pp.runpp if pf_type == "ac" else pp.rundcpp
    base_powerflow(net, lightsim2grid=use_ls2g)

    # N-0 Loadings festhalten, um "base_loading" korrekt zu berichten.
    base_loading = net.res_line["loading_percent"].copy()

    # Line-Contingencies auf Top-k% beschränken (No-op bei 100 %).
    nminus1_cases["line"]["index"] = select_topk_line_contingencies(net, topk_percent)
    contingency_lines = set(int(i) for i in nminus1_cases["line"]["index"])

    # Lines, die schon im Basisfall ausser Betrieb sind -> nicht als "Ausfall" werten.
    base_out_of_service = set(int(i) for i in net.line.index[~net.line["in_service"]])

    monitored_lines = net.line.index
    outage_impact: dict[int, dict[str, float | int]] = {}

    def evaluate_contingency(net: pp.pandapowerNet, **_kwargs: object) -> None:
        # Optionen bereits gesetzt -> teures Re-Parsing überspringen.
        _powerflow(net)

        # Welche Line ist in diesem Contingency-Schritt abgeschaltet?
        now_off = set(int(i) for i in net.line.index[~net.line["in_service"]])
        tripped = now_off - base_out_of_service

        # Nur einzelne Line-Ausfälle erfassen, die auch Contingencies sind.
        if len(tripped) != 1:
            return
        outage_line = next(iter(tripped))
        if outage_line not in contingency_lines:
            return

        loadings = net.res_line.loc[monitored_lines, "loading_percent"].copy()
        loadings.loc[outage_line] = np.nan  # abgeschaltete Line selbst ignorieren

        if loadings.notna().any():
            affected_line = int(loadings.idxmax())
            nminus1_loading = float(loadings.loc[affected_line])
        else:
            affected_line = -1
            nminus1_loading = float("nan")

        base_val = float(base_loading.get(affected_line, float("nan")))
        outage_impact[outage_line] = {
            "affected_line": affected_line,
            "nminus1_loading": nminus1_loading,
            "base_loading": base_val,
            "delta_loading": nminus1_loading - base_val,
        }

    # run_contingency überwacht alle vorhandenen Elementtypen -> max_loading_percent noetig.
    elements_missing_limit = [
        element
        for element in ("trafo", "trafo3w")
        if len(net[element]) and "max_loading_percent" not in net[element].columns
    ]
    for element in elements_missing_limit:
        net[element]["max_loading_percent"] = 100.0

    try:
        pp.contingency.run_contingency(
            net=net,
            nminus1_cases=nminus1_cases,
            contingency_evaluation_function=evaluate_contingency,
        )
    finally:
        for element in elements_missing_limit:
            del net[element]["max_loading_percent"]

    if (
        pf_type == "ac"
        and use_ls2g is not False
        and net._options["lightsim2grid"] is False  # noqa: SLF001
    ):
        logger.warning(
            "Warning: use_ls2g is %s, but lightsim2grid can't be used as backend.",
            use_ls2g,
        )

    if pf_type == "dc" and use_ls2g:
        logger.warning(
            "Warning: use_ls2g is %s, but lightsim2grid can't be used for DC powerflow.",
            use_ls2g,
        )

    return outage_impact