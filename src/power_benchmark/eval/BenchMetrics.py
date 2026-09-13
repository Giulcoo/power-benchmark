from pandapower_env.metrics.evaluation_metrics import MetricRegistry
from pandapower_env.metrics.metric_utils import StepData
from pandapower_env.substation.create_double_busbar_substation import create_all_double_busbar_substations
from power_benchmark.utils.net_utils import (
    networks_equal,
    count_in_service_lines_changes,
    count_closed_switches_changes,
    count_trafo_tap_positions_changes,
    get_voltage_violations,
    get_voltage_deviations, amount_of_overloaded_lines, amount_of_overloaded_regions
)

EPSILON_TAP_POS = 1e-2
TIMESTEP_HOUR = 0.25 # Timestep duration in hours, e.g., 0.25 for 15-min intervals # TODO: Make this dynamic based on env configs

import logging
logger = logging.getLogger("METRICS_LOGGING")

@MetricRegistry.register("timesteps_without_overload")
def timesteps_without_overload(step: StepData) -> float:
    """Return the longest number of consecutive timesteps without any line overload."""
    from pandapower_env.observation_space.pp_to_observation import line_loading_max

    key = MetricRegistry.env_key(step)
    current_overload_timesteps_by_env = MetricRegistry.cache_bucket("current_overload_timesteps")

    threshold_overload = 100
    value: bool = line_loading_max(step.env.net) < threshold_overload

    prev = current_overload_timesteps_by_env.get(key, 0)
    current = prev + 1 if value else 0
    current_overload_timesteps_by_env[key] = current
    return float(current)

@MetricRegistry.register("total_lines_overloaded")
def total_lines_overloaded(step: StepData) -> float:
    """Returns the cumulative number of overloaded lines across all timesteps."""
    key = MetricRegistry.env_key(step)
    lines_overloaded_by_env = MetricRegistry.cache_bucket("lines_overloaded")

    prev = lines_overloaded_by_env.get(key, 0)
    current = prev + amount_of_overloaded_lines(step.env.net, threshold=100)
    lines_overloaded_by_env[key] = current

    return float(current)

@MetricRegistry.register("total_regions_overloaded")
def total_regions_overloaded(step: StepData) -> float:
    """Returns the cumulative number of overloaded regions across all timesteps."""
    key = MetricRegistry.env_key(step)
    regions_overloaded_by_env = MetricRegistry.cache_bucket("regions_overloaded")

    prev = regions_overloaded_by_env.get(key, 0)
    current = prev + amount_of_overloaded_regions(step.env.net, threshold=100)
    regions_overloaded_by_env[key] = current

    return float(current)

@MetricRegistry.register("regions_timestep_overloaded")
def regions_timestep_overloaded(step: StepData) -> float:
    """Returns the number of overloaded regions for the current timestep."""
    return float(amount_of_overloaded_regions(step.env.net, threshold=100))

# ---------- Topology Stability Metrics -------------
@MetricRegistry.register("timesteps_in_start_topology")
def timesteps_in_start_topology(step: StepData) -> float:
    """Return the number of timesteps the grid stayed in the starting topology."""
    key = MetricRegistry.env_key(step)
    initial_net_by_env = MetricRegistry.cache_bucket("initial_net")
    timesteps_in_start_topology_by_env = MetricRegistry.cache_bucket("timesteps_in_start_topology")

    if key not in initial_net_by_env:
        initial_net_by_env[key] = step.env.net
    initial_net = initial_net_by_env[key]

    prev = timesteps_in_start_topology_by_env.get(key, 0)
    current = prev + (1 if networks_equal(step.env.net, initial_net, epsilon=EPSILON_TAP_POS) else 0)
    timesteps_in_start_topology_by_env[key] = current
    return float(current)



# ---------- Action Metrics -------------
@MetricRegistry.register("timesteps_without_actions")
def timesteps_without_actions(step: StepData) -> float:
    """Used for calculating how many timesteps in average the agent does not perform any action. Needed for total cost calculation."""
    key = MetricRegistry.env_key(step)
    no_actions_by_env = MetricRegistry.cache_bucket("no_actions")

    prev = no_actions_by_env.get(key, 0)
    current = prev + (1 if step.action == 0 else 0)
    no_actions_by_env[key] = current
    return float(current)

@MetricRegistry.register("substation_changes")
def substation_changes(step: StepData) -> float:
    """ Count how often an action changed a substation configuration (bus split state)
    compared to the previous action during the whole run. """
    key = MetricRegistry.env_key(step)
    previous_action_by_env = MetricRegistry.cache_bucket("previous_action_for_substation_changes")
    substation_changes_by_env = MetricRegistry.cache_bucket("substation_changes")

    action_dict = {index: d for index, d in enumerate(step.env.action_dict)}[step.action]

    # current configuration: {substation_id: state_string}
    current_config = dict(zip(action_dict["substations"], action_dict["states"]))

    if key not in previous_action_by_env:
        previous_action_by_env[key] = current_config
        substation_changes_by_env[key] = 0
        return 0.0

    previous_config = previous_action_by_env[key]
    prev = substation_changes_by_env[key]

    current = prev + (1 if current_config != previous_config else 0)

    substation_changes_by_env[key] = current
    previous_action_by_env[key] = current_config
    return float(current)

@MetricRegistry.register("lines_in_service_changes")
def lines_in_service_changes(step: StepData) -> float:
    """" Count how often a line got changed from in_service=True/False in the whole network during the whole run. """
    key = MetricRegistry.env_key(step)
    previous_net_by_env = MetricRegistry.cache_bucket("previous_net_for_lines_in_service_changes")
    lines_in_service_changes_by_env = MetricRegistry.cache_bucket("lines_in_service_changes")

    if key not in previous_net_by_env:
        previous_net_by_env[key] = step.env.net
        lines_in_service_changes_by_env[key] = 0
        return 0.0

    previous_net = previous_net_by_env[key]
    prev = lines_in_service_changes_by_env[key]

    current = prev + count_in_service_lines_changes(previous_net, step.env.net)
    lines_in_service_changes_by_env[key] = current
    previous_net_by_env[key] = step.env.net
    return float(current)

@MetricRegistry.register("closed_switches_changes")
def closed_switches_changes(step: StepData) -> float:
    """" Count how often a switch got changed from open/closed in the whole network during the whole run. """
    key = MetricRegistry.env_key(step)
    previous_net_by_env = MetricRegistry.cache_bucket("previous_net_for_closed_switches")
    closed_switches_changes_by_env = MetricRegistry.cache_bucket("closed_switches_changes")

    if key not in previous_net_by_env:
        previous_net_by_env[key] = step.env.net
        closed_switches_changes_by_env[key] = 0
        return 0.0

    previous_net = previous_net_by_env[key]
    prev = closed_switches_changes_by_env[key]
    current = prev + count_closed_switches_changes(previous_net, step.env.net)
    closed_switches_changes_by_env[key] = current
    previous_net_by_env[key] = step.env.net
    return float(current)

@MetricRegistry.register("trafo_tap_changes")
def trafo_tap_changes(step: StepData) -> float:
    """" Count how often a transformer tap position got changed in the whole network during the whole run. """
    key = MetricRegistry.env_key(step)
    previous_net_by_env = MetricRegistry.cache_bucket("previous_net_for_trafo_tap_changes")
    trafo_tap_changes_by_env = MetricRegistry.cache_bucket("trafo_tap_changes")

    if key not in previous_net_by_env:
        previous_net_by_env[key] = step.env.net
        trafo_tap_changes_by_env[key] = 0
        return 0.0

    previous_net = previous_net_by_env[key]
    prev = trafo_tap_changes_by_env[key]
    current = prev + count_trafo_tap_positions_changes(previous_net, step.env.net, epsilon=EPSILON_TAP_POS)
    trafo_tap_changes_by_env[key] = current
    previous_net_by_env[key] = step.env.net
    return float(current)

# --------------- Voltage Metrics ----------------------
@MetricRegistry.register("max_voltage_timestep_deviation")
def max_voltage_timestep_deviation(step: StepData) -> float:
    """Return the maximum voltage deviationfor for each timestep."""
    return float(get_voltage_deviations(step.env.net).max())

@MetricRegistry.register("mean_voltage_timestep_deviation")
def mean_voltage_timestep_deviation(step: StepData) -> float:
    """Return the mean voltage deviation across for each timesteps."""
    return float(get_voltage_deviations(step.env.net).mean())

@MetricRegistry.register("total_timesteps_with_voltage_violation")
def total_timesteps_with_voltage_violation(step: StepData) -> float:
    """Return number of timesteps with any bus outside voltage band."""
    key = MetricRegistry.env_key(step)
    timesteps_with_voltage_violation_by_env = MetricRegistry.cache_bucket("timesteps_with_voltage_violation")

    prev = timesteps_with_voltage_violation_by_env.get(key, 0)
    current = prev + (1 if len(get_voltage_violations(step.env.net)) > 0 else 0)
    timesteps_with_voltage_violation_by_env[key] = current
    return float(current)

@MetricRegistry.register("percent_bus_voltage_violation")
def percent_bus_voltage_violation(step: StepData) -> float:
    """
    Return the percentages of buses with voltage violation for each timestep.

    For each timestep, calculates: (buses with violation / total buses) * 100
    Returns the mean percentage for the current timestep. If no voltage data is available, returns 0.0.
    """
    vm_pu = step.env.net.res_bus.vm_pu.dropna()

    if len(vm_pu) == 0:
        return 0.0

    violations = get_voltage_violations(step.env.net)
    violation_count = len(violations)
    total_buses = len(vm_pu)
    current_percent = (violation_count / total_buses) * 100.0
    return current_percent

# ------------ Active Power Loss Metrics ------------
@MetricRegistry.register("total_active_power_loss")
def total_active_power_loss(step: StepData) -> float:
    """Return the sum of active power losses (MW) over the episode."""
    key = MetricRegistry.env_key(step)
    active_power_loss_by_env = MetricRegistry.cache_bucket("total_active_power_loss")

    prev = active_power_loss_by_env.get(key, 0.0)

    # Calculate losses from lines and transformers
    loss = 0.0
    if len(step.env.net.res_line) > 0:
        loss += step.env.net.res_line['pl_mw'].sum()
    if len(step.env.net.res_trafo) > 0:
        loss += step.env.net.res_trafo['pl_mw'].sum()
    if hasattr(step.env.net, 'res_trafo3w') and len(step.env.net.res_trafo3w) > 0:
        loss += step.env.net.res_trafo3w['pl_mw'].sum()
    current = prev + loss
    active_power_loss_by_env[key] = current
    return float(current)

@MetricRegistry.register("active_power_loss")
def active_power_loss(step: StepData) -> float:
    """Return the active power losses (MW) in the current timestep."""
    # Calculate losses from lines and transformers
    loss = 0.0
    if len(step.env.net.res_line) > 0:
        loss += step.env.net.res_line['pl_mw'].sum()
    if len(step.env.net.res_trafo) > 0:
        loss += step.env.net.res_trafo['pl_mw'].sum()
    if hasattr(step.env.net, 'res_trafo3w') and len(step.env.net.res_trafo3w) > 0:
        loss += step.env.net.res_trafo3w['pl_mw'].sum()
    return float(loss)

@MetricRegistry.register("active_power_loss_timestep_percent")
def active_power_loss_timestep_percent(step: StepData) -> float:
    """Return the active power losses as % of total load."""
    # Calculate current losses
    loss = 0.0
    if len(step.env.net.res_line) > 0:
        loss += step.env.net.res_line['pl_mw'].sum()
    if len(step.env.net.res_trafo) > 0:
        loss += step.env.net.res_trafo['pl_mw'].sum()
    if hasattr(step.env.net, 'res_trafo3w') and len(step.env.net.res_trafo3w) > 0:
        loss += step.env.net.res_trafo3w['pl_mw'].sum()

    # Calculate total load
    total_load = step.env.net.res_load['p_mw'].sum() if len(step.env.net.res_load) > 0 else 0.0

    # Calculate percentage (avoid division by zero)
    loss_percent = (loss / total_load * 100.0) if total_load > 0 else 0.0
    return float(loss_percent)

# ------------ Load Shedding Metrics ------------
@MetricRegistry.register("total_load_shedding")
def total_load_shedding(step: StepData) -> float:
    """Return the total unserved energy (MWh) over the episodes."""
    key = MetricRegistry.env_key(step)
    load_shedding_by_env = MetricRegistry.cache_bucket("cumulative_load_shedding_mwh")

    prev = load_shedding_by_env.get(key, 0.0)

    # Requested load (nominal * scaling * in_service)
    mask = step.env.net.load['in_service']
    requested_load = (
            step.env.net.load.loc[mask, 'p_mw'] * step.env.net.load.loc[mask, 'scaling']
    ).sum()

    # Served load from power flow results
    served_load = step.env.net.res_load['p_mw'].sum() if len(step.env.net.res_load) > 0 else 0.0

    # Load shedding (MW), ensure non-negative
    load_shedding_mw = max(0.0, requested_load - served_load)

    current = prev + load_shedding_mw * TIMESTEP_HOUR
    load_shedding_by_env[key] = current

    return float(current)

@MetricRegistry.register("load_shedding_timestep")
def load_shedding_timestep(step: StepData) -> float:
    """Return the unserved energy (MWh) over each timestep."""
    # Requested load (nominal * scaling * in_service)
    mask = step.env.net.load['in_service']
    requested_load = (
            step.env.net.load.loc[mask, 'p_mw'] * step.env.net.load.loc[mask, 'scaling']
    ).sum()

    # Served load from power flow results
    served_load = step.env.net.res_load['p_mw'].sum() if len(step.env.net.res_load) > 0 else 0.0

    # Load shedding (MW), ensure non-negative
    load_shedding_mw = max(0.0, requested_load - served_load)
    return float(load_shedding_mw * TIMESTEP_HOUR)

@MetricRegistry.register("load_shedding_timestep_percent")
def load_shedding_timestep_percent(step: StepData) -> float:
    """Return the unserved energy as % of total load for each timestep."""
    # Requested load (nominal * scaling * in_service)
    mask = step.env.net.load['in_service']
    requested_load = (
            step.env.net.load.loc[mask, 'p_mw'] * step.env.net.load.loc[mask, 'scaling']
    ).sum()

    # Served load from power flow results
    served_load = step.env.net.res_load['p_mw'].sum() if len(step.env.net.res_load) > 0 else 0.0

    # Load shedding (MW), ensure non-negative
    load_shedding_mw = max(0.0, requested_load - served_load)

    # Calculate percentage (avoid division by zero)
    load_shedding_percent = (load_shedding_mw / requested_load * 100.0) if requested_load > 0 else 0.0
    return float(load_shedding_percent)

@MetricRegistry.register("timesteps_with_load_shedding")
def timesteps_with_load_shedding(step: StepData) -> float:
    """Return the number of timesteps with involuntary load curtailment."""
    key = MetricRegistry.env_key(step)
    timesteps_with_shedding_by_env = MetricRegistry.cache_bucket("timesteps_with_load_shedding")

    # Threshold for considering load shedding (MW)
    threshold_mw = 1e-3

    # Calculate load shedding
    mask = step.env.net.load['in_service']
    requested_load = (
            step.env.net.load.loc[mask, 'p_mw'] * step.env.net.load.loc[mask, 'scaling']
    ).sum()
    served_load = step.env.net.res_load['p_mw'].sum() if len(step.env.net.res_load) > 0 else 0.0

    load_shedding_mw = requested_load - served_load

    prev = timesteps_with_shedding_by_env.get(key, 0)
    current = prev + (1 if load_shedding_mw > threshold_mw else 0)
    timesteps_with_shedding_by_env[key] = current
    return float(current)

# --------------- Device usage ----------------------
@MetricRegistry.register("total_max_used_line")
def total_max_used_line(step: StepData) -> float:
    """Counts for each line how often it got switched to in or out of service and returns the line with the most changes."""
    key = MetricRegistry.env_key(step)
    previous_net_by_env = MetricRegistry.cache_bucket("previous_net_for_total_max_used_line")
    line_usage_by_env = MetricRegistry.cache_bucket("line_usage")

    if key not in previous_net_by_env:
        previous_net_by_env[key] = step.env.net
        line_usage_by_env[key] = {}
        return 0.0

    previous_net = previous_net_by_env[key]
    line_usage = line_usage_by_env[key]

    for line_id in step.env.net.line.index:
        prev_in_service = previous_net.line.at[line_id, 'in_service'] if line_id in previous_net.line.index else None
        current_in_service = step.env.net.line.at[line_id, 'in_service'] if line_id in step.env.net.line.index else None

        if prev_in_service is not None and current_in_service is not None and prev_in_service != current_in_service:
            line_usage[line_id] = line_usage.get(line_id, 0) + 1

    line_usage_by_env[key] = line_usage
    previous_net_by_env[key] = step.env.net

    max_usage = max(line_usage.values()) if line_usage else 0
    return float(max_usage)

@MetricRegistry.register("total_max_used_switch")
def total_max_used_switch(step: StepData) -> float:
    """Counts for each switch how often it got switched to open or closed and returns the switch with the most changes."""
    key = MetricRegistry.env_key(step)
    previous_net_by_env = MetricRegistry.cache_bucket("previous_net_for_total_max_used_switch")
    switch_usage_by_env = MetricRegistry.cache_bucket("switch_usage")

    if key not in previous_net_by_env:
        previous_net_by_env[key] = step.env.net
        switch_usage_by_env[key] = {}
        return 0.0

    previous_net = previous_net_by_env[key]
    switch_usage = switch_usage_by_env[key]

    for switch_id in step.env.net.switch.index:
        prev_closed = previous_net.switch.at[switch_id, 'closed'] if switch_id in previous_net.switch.index else None
        current_closed = step.env.net.switch.at[switch_id, 'closed'] if switch_id in step.env.net.switch.index else None

        if prev_closed is not None and current_closed is not None and prev_closed != current_closed:
            switch_usage[switch_id] = switch_usage.get(switch_id, 0) + 1

    switch_usage_by_env[key] = switch_usage
    previous_net_by_env[key] = step.env.net

    max_usage = max(switch_usage.values()) if switch_usage else 0
    return float(max_usage)

@MetricRegistry.register("total_max_used_trafo")
def total_max_used_trafo(step: StepData) -> float:
    """Counts for each transformer how often it got switched to a different tap position and returns the transformer with the most changes."""
    key = MetricRegistry.env_key(step)
    previous_net_by_env = MetricRegistry.cache_bucket("previous_net_for_total_max_used_trafo")
    trafo_usage_by_env = MetricRegistry.cache_bucket("trafo_usage")

    if key not in previous_net_by_env:
        previous_net_by_env[key] = step.env.net
        trafo_usage_by_env[key] = {}
        return 0.0

    previous_net = previous_net_by_env[key]
    trafo_usage = trafo_usage_by_env[key]

    for trafo_id in step.env.net.trafo.index:
        prev_tap_pos = previous_net.trafo.at[trafo_id, 'tap_pos'] if trafo_id in previous_net.trafo.index else None
        current_tap_pos = step.env.net.trafo.at[trafo_id, 'tap_pos'] if trafo_id in step.env.net.trafo.index else None

        if prev_tap_pos is not None and current_tap_pos is not None and abs(prev_tap_pos - current_tap_pos) > EPSILON_TAP_POS:
            trafo_usage[trafo_id] = trafo_usage.get(trafo_id, 0) + 1

    #TODO: In general also look at trafo3w (also in traofo tap changes metric) if present in the net

    trafo_usage_by_env[key] = trafo_usage
    previous_net_by_env[key] = step.env.net

    max_usage = max(trafo_usage.values()) if trafo_usage else 0
    return float(max_usage)