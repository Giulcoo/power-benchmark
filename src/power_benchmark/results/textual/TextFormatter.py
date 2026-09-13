from typing import List, Dict, Any

import pandas as pd

class TextFormatter:
    def __init__(self, markdown: bool = True) -> None:
        self.markdown = markdown

    def title(self, text: str, level: int = 1) -> str:
        """Format a title/header."""
        if self.markdown:
            return f"{'#' * level} {text}\n"
        else:
            if level == 1:
                return f"{'=' * 60}\n{text.upper()}\n{'=' * 60}\n"
            elif level == 2:
                return f"{text.upper()}\n{'-' * 40}"
            else:
                return f"{text}\n{'-' * 30}"

    def bold(self, text: str) -> str:
        """Format bold text."""
        if self.markdown:
            return f"**{text}**"
        else:
            return text

    def italic(self, text: str) -> str:
        """Format italic text."""
        if self.markdown:
            return f"*{text}*"
        else:
            return text

    def key_value(self, key: str, value: Any) -> str:
        """Format a key-value pair."""
        if self.markdown:
            return f"**{key}:** {value}  "
        else:
            return f"{key + ':':<20} {value}"

    def table(self, data: List[Dict[str, Any]]) -> str:
        """Format tabular data as markdown table or pandas DataFrame."""
        """Format tabular data as markdown table or pandas DataFrame."""
        if not data:
            return ""

        if self.markdown:
            columns = list(data[0].keys())

            # Calculate max width for each column
            col_widths = {}
            for col in columns:
                max_data_width = max(len(str(row[col])) for row in data)
                col_widths[col] = max(len(col), max_data_width)

            # Header row
            header = "| " + " | ".join(col.ljust(col_widths[col]) for col in columns) + " |"
            separator = "|" + "|".join("-" * (col_widths[col] + 2) for col in columns) + "|"

            # Data rows
            rows = []
            for row in data:
                row_str = "| " + " | ".join(
                    str(row[col]).ljust(col_widths[col]) for col in columns
                ) + " |"
                rows.append(row_str)

            return "\n".join([header, separator] + rows)
        else:
            df = pd.DataFrame(data)
            return df.to_string(index=False)

    def section(self, title: str, content: str, level: int = 2) -> str:
        """Format a complete section with title and content."""
        return f"{self.title(title, level)}\n{content}\n"

    def horizontal_bar(self, char: str = "-", length: int = 60) -> str:
        """Format a horizontal bar/divider."""
        if self.markdown:
            return "\n---\n"
        else:
            return char * length