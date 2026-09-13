import time
from typing import Any, Dict, List
import psutil

class RunMetrics:
    """ Class to measure and collect runtime metrics during benchmark runs. Needed for performance evaluation. """

    def __init__(self):
        self.start_time = time.perf_counter()
        self.runtimes: List[float] = []
        self.cpu_usages: List[float] = []
        self.memory_usages: List[float] = []

    def measure_metrics(self):
        """ Measure and record runtime metrics during runtimes iterations. """
        self.measure_runtime()
        self.measure_cpu_usage()
        self.measure_memory_usage()

    def measure_runtime(self):
        """Return the total runtime. Used for performance evaluation by calculating average runtime per timestep."""
        end_time = time.perf_counter()
        self.runtimes.append(end_time - self.start_time)

        # Update start time for next iteration of measurement
        self.start_time = end_time

    def measure_cpu_usage(self):
        """Return the CPU usage per timestep. Used for performance evaluation."""
        cpu_usage = psutil.cpu_percent(interval=1)
        self.cpu_usages.append(cpu_usage)

    def measure_memory_usage(self):
        """Return the memory usage per timestep. Used for performance evaluation."""
        memory_usage = psutil.virtual_memory().percent
        self.memory_usages.append(memory_usage)

    def gpu_usage(self):
        """Return the GPU usage per timestep. Used for performance evaluation."""
        # TODO
        pass

    def __dict__(self) -> Dict[str, List[float]]:
        """ Collect and compute final post-run metrics and return all metrics as a dictionary. """
        return {
            "runtimes": self.runtimes,
            "cpu_usages": self.cpu_usages,
            "memory_usages": self.memory_usages,
        }

    @staticmethod
    def calc_statistics(metrics: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
        """ Calculate statistics (mean and max) for each metric. """
        return {
            metric_name: {
                "mean": sum(values) / len(values) if values else 0.0,
                "max": max(values) if values else 0.0,
            }
            for metric_name, values in metrics.items()
        }