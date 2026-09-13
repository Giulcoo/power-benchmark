import sys
from typing import Optional

from power_benchmark.configs.config import set_config_file, get_config, Task
from power_benchmark.results.data.ResultData import ResultData
from power_benchmark.utils.path_utils import get_project_root
from pathlib import Path
import logging
logger = logging.getLogger("Benchmarker")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise ValueError(f"Not enough arguments provided. Usage: python main.py <config_file_path>.")

    config_file_path = sys.argv[1]
    if config_file_path.startswith("./"):
        config_file_path = str(get_project_root() / Path(config_file_path[2:]))

    set_config_file(config_file_path)
    config = get_config()

    result: Optional[ResultData] = None
    if config.task & Task.COMPUTE_TASKS:
        from power_benchmark.runner import AnyBenchmarker
        benchmarker: AnyBenchmarker
        if config.agent_type == "DoNothing":
            logger.info("[MAIN] Load DoNothing Benchmarker")
            from power_benchmark.runner.PowerBenchmarker import DoNothingBenchmarker
            benchmarker = DoNothingBenchmarker()
        elif config.agent_type == "Greedy":
            logger.info("[MAIN] Load Greedy Benchmarker")
            from power_benchmark.runner.PowerBenchmarker import GreedyBenchmarker
            benchmarker = GreedyBenchmarker()
        else:
            logger.info("[MAIN] Load Agent Benchmarker")
            from power_benchmark.runner.AgentBenchmarker import AgentBenchmarker
            benchmarker = AgentBenchmarker()

        logger.info("[MAIN] Start Benchmarking")
        result = benchmarker.run()

    if config.task & Task.PRESENTER_TASKS:
        from power_benchmark.results.Presenter import Presenter
        pres = Presenter().run()

    logger.info("[MAIN] Finished Benchmarking")
    if result is not None:
        logger.info(result.summary_str())