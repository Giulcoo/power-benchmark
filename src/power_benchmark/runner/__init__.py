from typing import Union

from power_benchmark.runner.AgentBenchmarker import AgentBenchmarker
from power_benchmark.runner.PowerBenchmarker import PowerBenchmarker, GreedyBenchmarker, DoNothingBenchmarker

AnyBenchmarker = Union[PowerBenchmarker, GreedyBenchmarker, DoNothingBenchmarker, AgentBenchmarker]