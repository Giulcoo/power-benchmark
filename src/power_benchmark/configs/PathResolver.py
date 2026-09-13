# path_resolver.py

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from power_benchmark.constants.paths import SCENARIO_FOLDER
from power_benchmark.utils.path_utils import get_project_root, get_package_root


class PathResolver:
    """Resolves placeholder templates in config paths at any nesting depth."""

    PLACEHOLDER_PATTERN = re.compile(r"<(\w+)>")

    def __init__(
        self,
        raw_config: dict[str, Any],
        agent_type: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        self.raw_config = raw_config
        self.project_root = get_project_root()
        self.agent_type = agent_type
        self.seed = seed
        self.templates: dict[str, str] = {}

    def resolve(self) -> dict[str, Any]:
        """Resolve all placeholders and return the updated config dictionary."""
        self._build_templates()
        resolved = self._resolve_recursive(self.raw_config)
        return resolved

    def _build_templates(self) -> None:
        """Build the template mapping from known placeholders."""
        current_date = datetime.now()

        self.templates = {
            "project": str(self.project_root),
            "year": f"{current_date.year:04d}",
            "month": f"{current_date.month:02d}",
            "day": f"{current_date.day:02d}",
            "hour": f"{current_date.hour:02d}",
            "minute": f"{current_date.minute:02d}",
        }

        if self.agent_type is not None:
            self.templates["agent_type"] = self.agent_type

        if self.seed is not None:
            self.templates["seed"] = str(self.seed)

        # Resolve <output> from output_dir (which may itself contain placeholders)
        output_dir = self._find_key_recursive(self.raw_config, "output_dir")
        if output_dir is not None:
            resolved_output = self._replace_placeholders(str(output_dir))
            self.templates["output"] = str(Path(resolved_output).resolve())

    def _find_key_recursive(self, data: Any, target_key: str) -> Any:
        """Find a key at any depth and return its value."""
        if isinstance(data, dict):
            if target_key in data:
                return data[target_key]
            for value in data.values():
                result = self._find_key_recursive(value, target_key)
                if result is not None:
                    return result
        elif isinstance(data, list):
            for item in data:
                result = self._find_key_recursive(item, target_key)
                if result is not None:
                    return result
        return None

    def _resolve_recursive(self, data: Any) -> Any:
        """Recursively walk the config and resolve placeholders in strings."""
        if isinstance(data, dict):
            return {key: self._resolve_recursive(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._resolve_recursive(item) for item in data]
        elif isinstance(data, str):
            resolved = self._replace_placeholders(data)
            # Resolve the path to an absolute path (matching original behavior)
            if "<" not in resolved and ("/" in resolved or "\\" in resolved):
                return str(Path(resolved).resolve())
            return resolved
        return data

    def _replace_placeholders(self, value: str) -> str:
        """Replace all <placeholder> occurrences in a string."""
        def replacer(match: re.Match) -> str:
            key = match.group(1)
            if key in self.templates:
                return self.templates[key]
            # Leave unknown placeholders unchanged
            return match.group(0)

        return self.PLACEHOLDER_PATTERN.sub(replacer, value)