from typing import Literal, Tuple, Union
import subprocess
from pathlib import Path
import sys

from power_benchmark.results.app.ComparisonApp import ComparisonApp
from power_benchmark.results.data.ResultData import ResultData

class WebappLauncher:
    def __init__(self, result_dir:Path | Tuple[Path, Path], plotter_type: Literal["Matplotlib", "Plotly"]) -> None:
        self.result_dir = result_dir
        self.plotter_type = plotter_type

        print("Result dir: ", result_dir)

    def launch(self) -> None:
        """Start subprocess and launch the Streamlit web application by calling the main script of this file."""
        if isinstance(self.result_dir, Path):
            subprocess.run([
                sys.executable, "-m", "streamlit", "run",
                __file__,
                "--",
                str(self.result_dir),
                self.plotter_type
            ])
        else:
            subprocess.run([
                sys.executable, "-m", "streamlit", "run",
                __file__,
                "--",
                str(self.result_dir[0]),
                str(self.result_dir[1]),
                self.plotter_type
            ])

    def run(self):
        """Launch the Streamlit web application."""
        if isinstance(self.result_dir, Path):
            results = ResultData.load_from_dir(self.result_dir)

            if self.plotter_type == "Matplotlib":
                from power_benchmark.results.app.MatplotlibApp import MatplotlibApp
                MatplotlibApp(results, self.plotter_type).run()
            elif self.plotter_type == "Plotly":
                from power_benchmark.results.app.PlotlyApp import PlotlyApp
                PlotlyApp(results, self.plotter_type).run()
        else:
            result_a, result_b = self.result_dir
            ComparisonApp(results_a=ResultData.load_from_dir(result_a), results_b=ResultData.load_from_dir(result_b),
                          plotter=self.plotter_type).run()


if __name__ == "__main__":
    if len(sys.argv) not in [3,4]:
        print("Usage: streamlit run WebappLauncher.py -- <result_dirs> <plotter_type>")
        sys.exit(1)

    plotter_type = sys.argv[-1]
    if plotter_type not in ("Matplotlib", "Plotly"):
        print("plotter_type must be 'Matplotlib' or 'Plotly'")
        sys.exit(1)

    launcher = WebappLauncher(
        Path(sys.argv[1]) if len(sys.argv) == 3 else (Path(sys.argv[1]), Path(sys.argv[2])),
        plotter_type
    )
    launcher.run()