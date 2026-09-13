import logging
from typing import Dict, List, Any, Optional

import pandapower as pp
from typing_extensions import override

from power_benchmark.agent.BenchAgent import BenchAgent
from power_benchmark.configs.range_config import Index
from power_benchmark.configs.scenario_config import ScenarioConfig
from power_benchmark.configs.task_config import Task
from power_benchmark.results.data.ResultData import ResultData
from power_benchmark.runner.PowerBenchmarker import PowerBenchmarker, ExecResult
from power_benchmark.runner.RunData import RunDataLight
from power_benchmark.runner.Runner import DoNothingRunner
from power_benchmark.runner.Runner import Runner
from power_benchmark.utils.net_utils import has_gens, has_sgens, has_trafo_any_type

logger = logging.getLogger("Benchmarker")

class AgentBenchmarker(PowerBenchmarker):
    needs_dn_metrics = True

    """ Benchmarker for custom agents implementing the BenchAgent interface. """
    @override
    def __init__(self):
        super().__init__()
        self.n_gpu = self.config.n_gpu

        self.agent_cls = self.config.agent_cls
        self.agent_init_params = self.config.agent_init_params

        self.tuner_cls = self.config.tuner_cls
        self.tuner_init_params = self.config.tuner_init_params

    @override
    def run(self) -> Optional[ResultData]:
        # Only train if task includes training (TASK_LOAD_* tasks do not require training and will load existing agents)
        if Task.TRAIN in self.task or Task.TUNE in self.task:
            self._train_agent()

        if Task.RUN in self.task or Task.EVALUATE in self.task or Task.REPLAY in self.task: # TODO: Save DN runs
            return super().run()
        return None

    @override
    def _exec_run(self, scenario: ScenarioConfig, index: Index, amount_of_days: int, dn_metrics: Dict[str, List[float]] = {}) -> ExecResult:
        if scenario.eval_indexes is None:
            logger.warning(f"Scenario {scenario.scenario_name()} has no eval_indexes and is skipped.")
            return ExecResult()
        return super()._exec_run(scenario, index, amount_of_days)

    def _train_agent(self):
        for scenario in self.config.scenarios:
            if scenario.train_config is not None:
                self.train_scenario(scenario, cpu_count=self.n_cpu)
            else:
                logger.warning(f"Scenario {scenario.scenario_name()} has no train_config and is skipped.")

            import ray
            if ray.is_initialized():
                ray.shutdown()

    def train_scenario(self, scenario: ScenarioConfig, cpu_count: int = 1):
        env_config = scenario.train_config
        env_config["observation_keys"] = self._create_observation_keys(env_config["net"])

        hyper_params = None
        if self.tuner_cls is not None:
            tuner_params = self.tuner_init_params.copy()
            tuner_params["cpu_count"] = cpu_count
            tuner_params["gpu_count"] = self.n_gpu

            hyper_params = self.tuner_cls(
                scenario_name=scenario.scenario_name(),
                env_config=env_config,
                **tuner_params
            ).tune()

        if Task.TRAIN not in self.task:
            return

        agent_params = self.agent_init_params.copy()
        agent_params["checkpoint_dir"] += "/" + scenario.scenario_name()
        agent_params["cpu_count"] = cpu_count
        agent_params["gpu_count"] = self.n_gpu

        if hyper_params is not None:
            for key in hyper_params:
                env_config[key] = hyper_params[key]

        agent: BenchAgent = self.agent_cls(
            scenario_name=scenario.scenario_name(),
            env_config=env_config,
            **agent_params
        )
        result = agent.train()
        logger.info(f"Trained agent for scenario {scenario.case}, checkpoint saved at: {result.checkpoint.path}")

    @override
    def _load_runner(self, env_config: dict, index: Index, scenario: ScenarioConfig, amount_of_days: int, dn_metrics: Dict[str, List[float]]) -> Runner:
        agent_params = self.agent_init_params.copy()
        agent_params["checkpoint_dir"] += "/" + scenario.scenario_name()
        agent_params["cpu_count"] = 1
        agent_params["gpu_count"] = self.n_gpu

        env_config["observation_keys"] = self._create_observation_keys(env_config["net"])

        from power_benchmark.runner.Runner import Runner
        agent = self.agent_cls(
            scenario_name=scenario.scenario_name(),
            env_config=env_config,
            **agent_params
        )
        agent.load_agent()

        return Runner(
            agent=agent,
            env_config=env_config,
            index=index,
            amount_of_days=amount_of_days,
            dn_metrics=dn_metrics
        )

    @staticmethod
    def _create_observation_keys(net: pp.pandapowerNet) -> List[str]:
        # TODO: Make configurable in yaml/agent
        observation_keys = [
            "bus_voltage_magnitude", "bus_voltage_angle",
            "line_loadings", "line_power_flow_p_mw", "line_power_flow_q_mvar", "line_status", "line_thermal_limit",
            "load_status", "load_power_p_mw_profile", "load_power_q_mvar_profile", "load_power_p_mw_runpf",
            "load_power_q_mvar_runpf",
            "switch_positions",
            "total_power_demand_profile", "total_power_generation_profile", "total_power_demand_runpf",
            "total_power_generation_runpf", "system_losses",
            "adjacency_matrix", "bus_lookup_table", "load_bus", "line_from_bus", "line_to_bus",
        ]

        if has_trafo_any_type(net):
            observation_keys.extend([
                "transformer_loading_percent", "transformer_power_flow_p_mw", "transformer_power_flow_q_mvar",
                "transformer_tap_position", "transformer_status",
                "trafo_hv_bus", "trafo_lv_bus"
            ])

        if has_gens(net):
            observation_keys.extend([
                "gen_status", "gen_power_p_mw_profile", "gen_power_p_mw_runpf", "gen_vm_pu_profile", "gen_vm_pu_runpf",
                "gen_bus"
            ])

        if has_sgens(net):
            observation_keys.extend([
                "sgen_status", "sgen_power_p_mw_profile", "sgen_power_p_mw_runpf", "sgen_power_q_mvar_profile",
                "sgen_power_q_mvar_runpf",
                "sgen_bus"
            ])

        return observation_keys