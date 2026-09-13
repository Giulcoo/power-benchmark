import json
import re
import textwrap
from pathlib import Path
from typing import Any, Optional


def save_json(data: Any, path: str | Path, indent: int = 4, max_fields: Optional[int] = None, max_items: Optional[int] = None, max_outer_items: Optional[int] = None) -> None:
    """
    Save *data* as JSON to *path*.
    Parent directories are created automatically.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    json_string = json_compact_dump(data, indent=indent, max_fields=max_fields, max_items=max_items, max_outer_items=max_outer_items)

    with open(path, "w", encoding="utf-8") as f:
        f.write(json_string)

def load_json(path: str | Path) -> Any:
    """Load and return the JSON content of *path*."""
    with open(Path(path), "r", encoding="utf-8") as f:
        return json.load(f)

def json_compact_dump(data: Any, indent: int = 4, max_fields: Optional[int] = None, max_items: Optional[int] = None, max_outer_items: Optional[int] = None, prefix_tabs: Optional[int] = None) -> str:
    """Serialize *data* to a JSON string, optionally applying compact rules if *max_fields* or *max_items* are set."""
    json_string = json.dumps(data, indent=indent)
    if max_fields is not None:
        json_string = compact_objects(json_string, max_fields=max_fields)
    if max_items is not None:
        json_string = compact_arrays(json_string, max_items=max_items)
    if max_outer_items is not None and max_items is not None:
        json_string = compact_double_arrays(json_string, max_inner_items=max_items, max_outer_items=max_outer_items)

    if prefix_tabs is not None:
        json_string = textwrap.indent(json_string, " " * indent * prefix_tabs)

    return json_string

def compact_objects(json_string: str, max_fields: int) -> str:
    """
    Collapse multi-line JSON objects whose values are ALL primitives
    (string, number, boolean, null) into a single line.

    Parameters
    ----------
    json_string : str
        The JSON string to process.
    max_fields : int
        Only objects with at most this many fields are compacted.
        -1 means no limit (all objects with primitive values are compacted).

    Examples
    --------
    {
        "value": 42,
        "score": 0.9
    }
    is turned to {"value": 42, "score": 0.9}

    {
        "x": 1,
        "y": 2,
        "label": "A",
        "active": true
    }
    is turned to {"x": 1, "y": 2, "label": "A", "active": true}
    """
    primitive  = r'(?:"(?:[^"\\]|\\.)*"|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|true|false|null)'
    kv_pattern = r'"(?:[^"\\]|\\.)*":\s*' + primitive

    obj_pattern = (
        r'\{\s*\n'
        r'((?:\s*' + kv_pattern + r'\s*,?\s*\n)+)'
        r'\s*\}'
    )

    def _flatten(m: re.Match) -> str:
        pairs = re.findall(kv_pattern, m.group(1))
        if len(pairs) > max_fields >= 0:
            return m.group(0)  # leave unchanged
        return '{' + ', '.join(p.strip() for p in pairs) + '}'

    return re.sub(obj_pattern, _flatten, json_string)


def compact_arrays(json_string: str, max_items: int) -> str:
    """
    Collapse multi-line arrays of primitives (numbers / bools / null / short strings)
    that have at most *max_items* elements onto a single line.

    Parameters
    ----------
    max_items : int
        Maximum number of elements to allow compaction.
        Pass -1 for no limit.

    Example:
        [
            1,
            2,
            3
        ]

        is turned to [1, 2, 3]
    """
    primitive = r'(?:"[^"]{0,40}"|true|false|null|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)'
    quantifier = r'+' if max_items == -1 else r'{1,' + str(max_items) + r'}'
    pattern = (
        r'\[\s*\n'
        r'((?:\s*' + primitive + r'\s*,?\s*\n)' + quantifier + r')'
        r'\s*\]'
    )

    def _flatten(m: re.Match) -> str:
        items = re.findall(primitive, m.group(1))
        return '[' + ', '.join(items) + ']'

    return re.sub(pattern, _flatten, json_string)

def compact_double_arrays(
    json_string: str,
    max_inner_items: int,
    max_outer_items: int,
) -> str:
    """
    Compact all double arrays (arrays whose elements are arrays of primitives)
    found anywhere in the JSON string.

    Steps
    -----
    1. Collapse inner primitive arrays onto single lines (reuses compact_arrays).
    2. Collapse outer arrays whose items are now all single-line arrays.

    Parameters
    ----------
    json_string : str
        The JSON string to process.
    max_inner_items : int
        Inner arrays with more than this many items are left expanded.
        -1 means no limit (all inner arrays with primitive items are compacted).
    max_outer_items : int
        Outer arrays with more than this many items are left expanded.
        -1 means no limit (all outer arrays with single-line items are compacted).

    Example
    -------
    [
        [
            1.390154,
            98.609846
        ],
        [
            1.424989,
            98.575011
        ]
    ]

    is turned to [[1.390154, 98.609846], [1.424989, 98.575011]]
    """
    # Collapse inner arrays first
    result = compact_arrays(json_string, max_items=max_inner_items)

    # Collapse outer arrays of single-line arrays
    single_line_array = r'\[[^\[\]\n]*\]'          # Arrays [...] with no newlines inside
    outer_pattern = (
        r'\[\s*\n'
        r'((?:\s*' + single_line_array + r'\s*,?\s*\n)+)'
        r'\s*\]'
    )

    def _flatten_outer(m: re.Match) -> str:
        items = re.findall(single_line_array, m.group(1))
        if max_outer_items is not None and len(items) > max_outer_items >= 0:
            return m.group(0)                       # leave unchanged
        return '[' + ', '.join(items) + ']'

    return re.sub(outer_pattern, _flatten_outer, result)