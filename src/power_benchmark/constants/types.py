from typing import Dict, Any, List, Union, Literal, Tuple

# Value and scoring types
VALUE_LITERAL = Literal["worst", "mean"]

VALUE_TYPE = Union[float, Dict[VALUE_LITERAL, float]]
EVAL_CRITERION = Union[VALUE_LITERAL, List[VALUE_LITERAL]]

VALUE_SCORE_TUPLE = Tuple[float, float]
VALUE_SCORE_LIST = List[VALUE_SCORE_TUPLE]
VALUE_SCORE_DICT = Dict[VALUE_LITERAL, VALUE_SCORE_TUPLE]
VALUE_SCORE_TYPES = Union[VALUE_SCORE_TUPLE, VALUE_SCORE_LIST, VALUE_SCORE_DICT]

# Weight types
METRICS_ENTRY = Union[str, Dict[str,List[VALUE_LITERAL]]]
METRICS_LIST = List[METRICS_ENTRY]
ALLOWED_METRICS: Dict[str, METRICS_LIST] = {
    "overload_mitigation": [
        "timesteps_without_overload",
        "total_lines_overloaded",
        # "total_regions_overloaded",
        {"max_line_loading_timestep_nminus0": ["mean", "worst"]},
        {"mean_line_loading_timestep_nminus0": ["mean", "worst"]},
        {"total_lines_overloaded_timestep_nminus0": ["mean", "worst"]},
        {"max_line_loading_nminus1": ["mean", "worst"]},
        {"total_lines_overloaded_timestep_nminus1": ["mean", "worst"]},
        # {"overload_cases_nminus1": ["mean", "worst"]}, # TODO: Add
        # {"regions_timestep_overloaded": ["mean", "worst"]},
    ],
    "voltage_violation_mitigation": [
        "total_timesteps_with_voltage_violation",
        {"max_voltage_timestep_deviation": ["mean", "worst"]},
        {"mean_voltage_timestep_deviation": ["mean", "worst"]},
        {"percent_bus_voltage_violation": ["mean", "worst"]},
    ],
    "survival": [
        "survived_iterations",
    ],
    "costs": [
        # "timesteps_in_start_topology",

        "timesteps_without_actions",
        "substation_changes",
        # "lines_in_service_changes",
        # "closed_switches_changes",
        # "trafo_tap_changes",
        # "total_max_used_line",
        # "total_max_used_switch",
        # "total_max_used_trafo",

        {"active_power_loss_timestep_percent": ["mean", "worst"]},
        "timesteps_with_load_shedding",
        {"load_shedding_timestep_percent": ["mean", "worst"]},
    ],
    "computational_performance": [
        {"runtime_timestep": ["mean", "worst"]},
        {"cpu_usage_timestep": ["mean", "worst"]},
        {"memory_usage_timestep": ["mean", "worst"]},
    ],
    "total": [
        "overload_mitigation",
        "voltage_violation_mitigation",
        "survival",
        "costs",
        "computational_performance",
    ],
}

WEIGHT_TYPE = Union[float, Dict[VALUE_LITERAL, float]]
METRICS_ENTRY_WITH_WEIGHTS = Dict[str, WEIGHT_TYPE]
CATEGORY_LIST_WITH_WEIGHTS = Dict[str, METRICS_ENTRY_WITH_WEIGHTS]