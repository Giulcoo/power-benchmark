import copy
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

import pandapower as pp
from gymnasium import spaces
from numpy import integer
from typing_extensions import override

from power_benchmark.configs.env_config import EnvConfig
from power_benchmark.configs.range_config import Index
from pandapower_env.agents.base_agents import BaseAgent
from pandapower_env.agents.benchmark_agents import GreedyAgent, DoNothingAgent
from pandapower_env.environments.simulation_env import PPTopoGym
from power_benchmark.agent.RayAgent import BenchAgent
from power_benchmark.configs.config import get_config
from power_benchmark.configs.scenario_config import ScenarioConfig
from power_benchmark.eval.EvaluateBench import EvaluateBench
from power_benchmark.eval.RunMetrics import RunMetrics
from power_benchmark.results.replay.ReplayPlotter import ReplayPlotter
from power_benchmark.results.data.ScenarioData import SingleResult
from power_benchmark.runner.RunData import RunDataLight, RunData

logger = logging.getLogger("Benchmarker")

class Runner:
    def __init__(self,
                 agent: BaseAgent | BenchAgent,
                 env_config: dict,
                 index: Index,
                 amount_of_days: Optional[int] = None,
                 dn_metrics: Optional[Dict[str, Any]] = None
                 ):
        self.agent = agent

        self.env_config = copy.deepcopy(env_config)

        self.weights = get_config().eval_weights

        self.amount_of_days = 1

        self.index = index
        if amount_of_days is not None:
            self.env_config["episode_length"] = self.env_config["episode_length"] * amount_of_days
            self.amount_of_days = amount_of_days

        if dn_metrics is not None:
            self.dn_metrics = dn_metrics
        else:
            self.dn_metrics = None

        # Class variables used later in run and eval
        self.run_metrics: Dict[str, List[float]] = {}
        self.run_actions = None

    def run(self, scenario_config: Optional[ScenarioConfig] = None) -> Union[RunData, RunDataLight]:
        """Run the benchmark for a specified index and scenario and return the collected runtime metrics."""
        env = PPTopoGym(self.env_config)
        obs, info = env.reset(options={"index": self.index.index})

        logging.getLogger("pandapower.contingency.contingency").setLevel(logging.CRITICAL)

        metrics = RunMetrics()
        for _ in range(self.env_config["episode_length"]):
            action = self.agent_act(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            metrics.measure_metrics()

            self.post_step(env.net, action)

            if terminated or truncated:
                break

        self.run_metrics = metrics.__dict__()
        self.run_actions = env.log_actions

        if scenario_config is not None:
            return RunData(
                run_metrics=self.run_metrics,
                run_actions=self.run_actions,
                scenario_config=scenario_config,
                index=self.index,
                amount_of_days=self.amount_of_days,
            )

        return RunDataLight(
            run_metrics=self.run_metrics,
            index=self.index,
            amount_of_days=self.amount_of_days,
        )

    def agent_act(self, obs: dict, info: dict) -> int | integer:
        return self.agent.act(obs, info)

    def post_step(self, after: pp.pandapowerNet, action: int):
        pass

    def eval(self) -> SingleResult:
        return EvaluateBench(
            index=self.index,
            amount_of_days=self.amount_of_days,
            run_metrics=self.run_metrics,
            weights=self.weights,
            env_config=copy.deepcopy(self.env_config),
            actions=self.run_actions,
            dn_metrics=self.dn_metrics if self.dn_metrics is not None else {}
        ).run()

class GreedyRunner(Runner):
    @override
    def __init__(self, env_config: dict, index: Index, amount_of_days: Optional[int], dn_metrics: Optional[Dict[str, Any]]):
        config = copy.deepcopy(env_config)
        actions = spaces.Discrete(len(config["action_space"]))
        super().__init__(GreedyAgent(actions, config, n_workers=get_config().num_workers), config, index=index, amount_of_days=amount_of_days, dn_metrics=dn_metrics)

class DoNothingRunner(Runner):
    @override
    def __init__(self, env_config: dict, index: Index, amount_of_days: Optional[int] = None):
        config = copy.deepcopy(env_config)
        actions = spaces.Discrete(len(config["action_space"]))
        super().__init__(DoNothingAgent(actions), config, index=index, amount_of_days=amount_of_days)

    @override
    def agent_act(self, obs: dict, info: dict) -> int | integer:
        return self.agent.act(obs)

class ReplayRunner(Runner):
    @override
    def __init__(self,
                run_output: Union[str, Path, RunData],
                output_file: str | Path,
                index: Index,
                config: EnvConfig,
                amount_of_days: Optional[int] = None,
                ):
        self.run_output = run_output
        self.output_file = output_file

        from power_benchmark.runner.RunData import RunData

        if isinstance(run_output, RunData):
            run_data = run_output
        else:
            run_data = RunData.load_data(run_output, index, amount_of_days, env_config=config)
        self.actions = run_data.run_actions.to_numpy()

        config = copy.deepcopy(run_data.scenario_config.eval_config)
        self.scenario_name = f"scenario_{run_output}_index{index.index}_day{index.day}"
        self.plotter = ReplayPlotter(config, self.scenario_name)
        action_space = spaces.Discrete(len(config["action_space"]))
        config["episode_length"] = len(self.actions)

        super().__init__(DoNothingAgent(action_space), config, index=index, amount_of_days=None)

    @override
    def agent_act(self, obs: dict, info: dict) -> int | integer:
        step = info.get("current_step", 0)
        logger.debug(f"Current step {step} and action {self.actions[step]}")
        return self.actions[step]

    @override
    def post_step(self, after: pp.pandapowerNet, action: int):
        self.plotter.add_net(after, action)

    @override
    def run(self, scenario_config: Optional[ScenarioConfig] = None) -> Dict[str, List[float]]:
        if len(self.actions) == 0:

            logger.info(f"No actions available for {self.scenario_name}")
            return {}

        super().run()

        self.plotter.combine_figs(self.run_actions)
        self.plotter.save(Path(self.output_file) )

        return self.run_metrics
