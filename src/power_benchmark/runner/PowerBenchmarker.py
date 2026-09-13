import logging
import shutil
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List, Union, Optional, ClassVar, Any

from joblib import Parallel, delayed
from typing_extensions import override

from power_benchmark.configs.config import get_config_file, set_config_file, get_config
from power_benchmark.configs.range_config import Index
from power_benchmark.configs.scenario_config import ScenarioConfig
from power_benchmark.configs.task_config import Task
from power_benchmark.eval.EvaluateBench import EvaluateBench
from power_benchmark.results.data.ResultData import ResultData
from power_benchmark.results.data.ScenarioData import SingleResult, ScenarioResult
from power_benchmark.runner.RunData import RunData, RunDataLight
from power_benchmark.runner.Runner import Runner, ReplayRunner, DoNothingRunner

logger = logging.getLogger("Benchmarker")

@dataclass
class ExecResult:
    single_result: Optional[SingleResult] = None
    run_data: Optional[Union[RunData, RunDataLight]] = None

class PowerBenchmarker(ABC):
    needs_dn_metrics: ClassVar[bool] = False

    def __init__(self):
        """ Initialize the PowerBenchmarker with configuration. """
        self.config = get_config()

        self.scenarios = self.config.scenarios
        self.output_dir = self.config.output_dir
        self.n_cpu = self.config.n_cpu
        self.task = self.config.task
        self.run_output_dir = self.config.run_output_dir
        self.save_run_data = self.config.save_run_data

        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir)

        for scenario in self.scenarios:
            self._cleanup_files(scenario)

        logging.getLogger("pandapower.convert_format").setLevel(logging.CRITICAL)

    def _cleanup_files(self, scenario: ScenarioConfig):
        """ Removes scenario files before the run starts. """
        def cleanup(path: Path):
            if not path.exists():
                return
            if path.is_file():
                path.unlink()
            else:
                shutil.rmtree(path)

            logger.info(f"Removed old files in {path}")

        if Task.RUN in self.task:
            cleanup(Path(self.run_output_dir + f"/{scenario.scenario_name()}"))
            cleanup(Path(self.run_output_dir + f"/DN_{scenario.scenario_name()}"))
        if Task.EVALUATE in self.task:
            cleanup(Path(self.output_dir / Path(f"results/{scenario.scenario_name()}.json")))
        if Task.REPLAY in self.task:
            cleanup(Path(self.output_dir / Path(f"replay/{scenario.scenario_name()}")))

    def run(self) -> Optional[ResultData]:
        """ Run the benchmark for all scenarios and save results. """
        result_data = ResultData.load_from_dir(self.output_dir, {
            "name": self.config.name,
            "agent": self.config.agent_type if isinstance(self.config.agent_type, str) else self.config.agent_cls.__name__,
            "description": self.config.description,
            "weights": self.config.eval_weights,
            "config_file": get_config_file()
        })

        if Task.STATISTICS in self.task:
            for s_num, scenario in enumerate(self.scenarios):
                # Load scenario results
                result_data.scenarios.append(
                    ScenarioResult.load_json(
                        self.output_dir / Path(f"results/{scenario.scenario_name()}.json"),
                        self.config.eval_weights
                    )
                )

            result_data.save_meta_data(self.config.eval_weights)
            result_data.save_statistics()
            return result_data

        if (self.n_cpu > 1 or self.n_cpu == -1) and self.config.evaluation_tasks > 1:
            result_data.scenarios = self._run_parallel()
        else:
            for s_num, scenario in enumerate(self.scenarios):
                logger.info(f"Running scenarios {s_num + 1}/{len(self.scenarios)}: {scenario.category} - {scenario.case}")

                scenario_results: List[ExecResult] = []

                indexes = scenario.eval_indexes if scenario.eval_indexes is not None and len(scenario.eval_indexes) > 0 else scenario.indexes
                for index_num, index in enumerate(indexes):
                    logger.info(f"  Run {index_num + 1}/{self.config.evaluation_tasks} for scenarios '{scenario.case}' (scenarios number {s_num + 1})")
                    scenario_results.append(self._exec_run(scenario, index, scenario.amount_of_days))

                result = ScenarioResult(
                    network_category=scenario.category,
                    name=scenario.scenario_name(),
                    scenario_config=scenario.file_name,
                    file=self.output_dir / Path(f"results/{scenario.scenario_name()}.json"),
                    results=[result.single_result for result in scenario_results if result.single_result is not None]
                )
                result_data.scenarios.append(result)

                # Save run data if existing
                for run_data in [result.run_data for result in scenario_results if result.run_data is not None]:
                    run_data.save_data(scenario.run_data_path(self.run_output_dir))

                if Task.EVALUATE in self.task:
                    result.save_json()

        result_data.save_failed_scenarios()

        if Task.EVALUATE in self.task:
            result_data.save_meta_data(self.config.eval_weights)
            result_data.save_statistics()

        return result_data

    def _run_parallel(self) -> List[ScenarioResult]:
        """ Run the benchmark for all scenarios in parallel across CPUs. """

        def process_scenario_index(config_file: str, sc_config: ScenarioConfig, index: Index) -> Tuple[ScenarioConfig, ExecResult]:
            set_config_file(config_file) # Set config file again, so that each process has it globally
            return sc_config, self._exec_run(sc_config, index, sc_config.amount_of_days)

        # Create all task parameters
        tasks = []
        for scenario in self.scenarios:
            indexes = scenario.eval_indexes if scenario.eval_indexes is not None and len(scenario.eval_indexes) > 0 else scenario.indexes
            tasks += [(get_config_file(), scenario, index) for index in indexes]

        logger.info(f"Running {len(tasks)} tasks on {self.n_cpu} CPUs")

        # Run in parallel (n_jobs=-1 uses all CPUs automatically)
        task_results: List[Tuple[ScenarioConfig, ExecResult]] = (
            Parallel(n_jobs=min(self.n_cpu, len(tasks)), timeout=None, verbose=10 if self.config.logging_level <= logging.INFO else 0)(
                delayed(process_scenario_index)(*task) for task in tasks
            )
        )

        # Save run data if existing
        gathered_run_data: List[Tuple[ScenarioConfig, Union[RunData, RunDataLight]]] = [(scenario, result.run_data) for scenario, result in task_results if result.run_data is not None]
        for scenario, run_data in gathered_run_data:
            run_data.save_data(scenario.run_data_path(self.run_output_dir))

        # Aggregate results and save
        results: List[ScenarioResult] = []

        for scenario in self.scenarios:
            result = ScenarioResult(
                network_category=scenario.category,
                name=scenario.scenario_name(),
                scenario_config=scenario.file_name,
                file=self.output_dir / Path(f"results/{scenario.scenario_name()}.json"),
                results=[
                    result.single_result for task_scenario, result in task_results
                    if task_scenario.scenario_name() == scenario.scenario_name() and result.single_result is not None
                ]
            )

            results.append(result)
            if Task.EVALUATE in self.task:
                result.save_json()

        return results


    def _exec_run(self, scenario: ScenarioConfig, index: Index, amount_of_days: int) -> ExecResult:
        result: ExecResult = ExecResult()

        try:
            dn_metrics: Dict[str, Any] = {}
            if self.needs_dn_metrics:
                dn_metrics = self._load_dn_metrics(config=scenario.eval_config, index=index, scenario=scenario)

            if Task.RUN in self.task and Task.EVALUATE in self.task:
                runner = self._load_runner(env_config=scenario.eval_config, index=index, scenario=scenario, amount_of_days=amount_of_days, dn_metrics=dn_metrics)
                result.run_data = runner.run(scenario)
                result.single_result = runner.eval()
            elif Task.RUN in self.task:
                runner = self._load_runner(env_config=scenario.eval_config, index=index, scenario=scenario, amount_of_days=amount_of_days, dn_metrics=dn_metrics)
                result.run_data = runner.run(scenario)
            elif Task.EVALUATE in self.task:
                run_data = RunData.load_data(scenario.run_data_path(self.run_output_dir), index, amount_of_days, env_config=scenario.env_config)
                result.single_result = EvaluateBench(
                    index=index,
                    amount_of_days=amount_of_days,
                    run_metrics=run_data.run_metrics,
                    weights=self.config.eval_weights,
                    env_config=run_data.scenario_config.eval_config,
                    actions=run_data.run_actions,
                    dn_metrics=dn_metrics
                ).run()

            if Task.REPLAY in self.task:
                run_data: Union[str, RunData] = result.run_data or self.run_output_dir + f"/{scenario.scenario_name()}"
                output_file = self.output_dir / Path("replay/" + f"/{scenario.scenario_name()}/index{index.index}_day{index.day}.html")
                ReplayRunner(run_data, output_file.__str__(), index=index, amount_of_days=amount_of_days, config=scenario.env_config).run(scenario)

            logger.info(f"Finished {scenario.category} - {scenario.case} - {scenario.scenario_name()} | Index {index}")
        except Exception as e:
            result.single_result = SingleResult(index=index.index, amount_of_days=amount_of_days, metric_categories=[], failed=True)
            logger.error(f"Error running scenario '{scenario.scenario_name()}' index {index}:\n   {e}\n   {traceback.format_exc()}")

        return result

    @abstractmethod
    def _load_runner(self, env_config: dict, index: Index, scenario: ScenarioConfig, amount_of_days: int, dn_metrics: Dict[str, List[float]]) -> Runner:
        """ Load the needed Runner subclass. """
        pass

    def _load_dn_metrics(self, config: dict, index: Index, scenario: ScenarioConfig) -> Dict[str, Any]:
        # Load metrics and save in dn_metrics dict
        dn_run_output_dir = self.run_output_dir + f"/DN_{scenario.scenario_name()}"
        run_data: RunDataLight

        if Task.EVALUATE in self.task and Task.RUN not in self.task:
            # Load from existing run when only doing evaluation
            run_data = RunDataLight.load_data(dn_run_output_dir, index, 1)
        else:
            # Gather new metrics with DoNothingRunner when run is set
            run_data = DoNothingRunner(config, index, amount_of_days=1).run()
            run_data.save_data(dn_run_output_dir)

        return run_data.run_metrics

class GreedyBenchmarker(PowerBenchmarker):
    needs_dn_metrics = True

    @override
    def _load_runner(self, env_config: dict, index: Index, scenario: ScenarioConfig, amount_of_days: int, dn_metrics: Dict[str, List[float]]) -> Runner:
        from power_benchmark.runner.Runner import GreedyRunner
        return GreedyRunner(env_config, index=index, amount_of_days=amount_of_days, dn_metrics=dn_metrics)

class DoNothingBenchmarker(PowerBenchmarker):
    @override
    def _load_runner(self, env_config: dict, index: Index, scenario: ScenarioConfig, amount_of_days: int, dn_metrics: Dict[str, List[float]]) -> Runner:
        return DoNothingRunner(env_config, index=index, amount_of_days=amount_of_days)