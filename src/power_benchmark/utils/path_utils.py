from datetime import datetime
from typing import Any, List, Optional
import yaml
from pathlib import Path
from functools import lru_cache
import re

from power_benchmark.constants.paths import SCENARIO_FOLDER

ACCEPTED_TEMPLATE_VARS = ["<project>", "<agent_type>", "<seed>", "<year>", "<month>", "<day>", "<hour>", "<minute>"]

@lru_cache()
def get_project_root() -> Path:
    """
    Returns the root directory of the project.
    """
    # __file__ -> path_utils.py
    # parents[0] = utils
    # parents[1] = power_benchmark
    # parents[2] = src
    # parents[3] = project root
    return Path(__file__).resolve().parents[3]

@lru_cache()
def get_package_root() -> Path:
    """
    Returns the root directory of the project.
    """
    # __file__ -> path_utils.py
    # parents[0] = utils
    # parents[1] = power_benchmark
    return Path(__file__).resolve().parents[1]

def get_path_from_template(path_template: str, agent_type: Optional[str] = None, seed: Optional[int] = None) -> str:
    """
    Resolves a path template by replacing placeholders with actual values.
    """
    resolved = path_template
    resolved = resolved.replace("<project>", str(get_project_root()))

    if agent_type is not None:
        resolved = resolved.replace("<agent_type>", agent_type)

    if seed is not None:
        resolved = resolved.replace("<seed>", str(seed))

    current_date = datetime.now()
    resolved = resolved.replace("<year>", f"{current_date.year:04d}")
    resolved = resolved.replace("<month>", f"{current_date.month:02d}")
    resolved = resolved.replace("<day>", f"{current_date.day:02d}")
    resolved = resolved.replace("<hour>", f"{current_date.hour:02d}")
    resolved = resolved.replace("<minute>", f"{current_date.minute:02d}")
    return str(Path(resolved).resolve())

def concat_path(root: str | Path, suffix: str | Path) -> str:
    return Path(root).joinpath(suffix).resolve().__str__()

def extract_index(s):
    """ Extracts an index from a string if it ends with [number]. """
    m = re.search(r'\[(\d+)\]$', s)
    if not m:
        return s, None
    i = int(m.group(1))
    new_s = s[:m.start()]
    return new_s, i

def load_scenario(scenario_path: str) -> Any | List[Any]:
    # Check if scenario ends with [number] to indicate index
    scenario_path, index = extract_index(scenario_path)
    path = str(Path(scenario_path).resolve())
    with open(path) as f:
        data = yaml.safe_load(f)

    if isinstance(data, list):
        return data if index is None else data[index]
    return data