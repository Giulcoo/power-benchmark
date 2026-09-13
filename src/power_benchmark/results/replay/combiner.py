from typing import List, Optional

import plotly.graph_objects as go


class FigureCombiner:
    """Combines multiple Plotly figures into a single figure with a slider."""

    def __init__(self, net_figs: List[go.Figure], info_dicts: Optional[List[dict]] = None):
        self.net_figs = net_figs
        self.info_dicts = info_dicts

    def build(self) -> go.Figure:
        """Combine all figures into one with a slider control."""
        combined = go.Figure()

        trace_counts = []
        for fig in self.net_figs:
            trace_counts.append(len(fig.data))
            for trace in fig.data:
                combined.add_trace(trace)

        num_figs = len(self.net_figs)
        annotation_trace_start = len(combined.data)

        # Add annotation traces
        for i in range(num_figs):
            if self.info_dicts and 0 <= i < len(self.info_dicts):
                text = "<br>".join(
                    f"<b>{k}</b>: {v}" for k, v in self.info_dicts[i].items()
                )
            else:
                text = ""

            combined.add_trace(go.Scatter(
                x=[0], y=[0],
                mode="text",
                text=[text],
                textposition="middle center",
                showlegend=False,
                visible=(i == 0),
                xaxis="x", yaxis="y",
            ))

        # Set initial visibility
        for trace in combined.data[:annotation_trace_start]:
            trace.visible = False
        for j in range(trace_counts[0]):
            combined.data[j].visible = True

        # Build slider steps
        steps = []
        current_idx = 0
        for i, count in enumerate(trace_counts):
            visibility = [False] * len(combined.data)

            for j in range(current_idx, current_idx + count):
                visibility[j] = True

            for k in range(num_figs):
                visibility[annotation_trace_start + k] = (k == i)

            steps.append(dict(
                method="update",
                args=[
                    {"visible": visibility},
                    {"title": f"Figure {i} / {num_figs - 1}"},
                ],
                label=str(i),
            ))
            current_idx += count

        combined.update_layout(
            title=f"Figure 0 / {num_figs - 1}",
            autosize=True,
            height=800,
            sliders=[dict(
                active=0,
                currentvalue={"prefix": "Index: "},
                pad={"t": 50},
                steps=steps,
            )],
        )

        return combined