from typing import Dict, Optional, Any

from typing_extensions import override

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.tune.registry import register_env

from pandapower_env.environments.simulation_env import PPTopoGym
from pandapower_env.rlib_agents.gnn_agents import GINETorchRLModule
from power_benchmark.agent.RayAgent import RayAgent

from torch.utils.tensorboard import SummaryWriter

class PpoAgent(RayAgent):
    """ GNN-based agent for power system control using Graph Isomorphism Network with Edge features (GINE)."""

    def __init__(self,
                 scenario_name: str,
                 env_config: Dict[str, Any],
                 checkpoint_dir: str,
                 resume_from_checkpoint: bool = True,
                 log_dir: Optional[str] = None,
                 training_iters: int = 1000,
                 checkpoint_every_iters: int = 100,
                 explore_during_act: bool = False,
                 cpu_count: int = 1,
                 gpu_count: int = 0,
                 patience: Optional[int] = None,
                 # Tunable hyperparameters
                 model_config: Optional[Dict[str, Any]] = None,
                 ppo_config: Optional[Dict[str, Any]] = None,
                 ):

        self.model_hparams = model_config or {}
        self.ppo_hparams = ppo_config or {}

        super().__init__(
            scenario_name=scenario_name,
            env_config=env_config,
            checkpoint_dir=checkpoint_dir,
            resume_from_checkpoint=resume_from_checkpoint,
            training_iters=training_iters,
            checkpoint_every_iters=checkpoint_every_iters,
            explore_during_act=explore_during_act,
            cpu_count=cpu_count,
            gpu_count=gpu_count,
            patience=patience
        )

        self.writer: Optional[SummaryWriter] = None

        if log_dir is not None:
            self.writer = SummaryWriter(log_dir=log_dir + "/" + scenario_name)

    def build_agent(self):
        def env_creator(env_config):
            return PPTopoGym(env_config)

        register_env("SimulationEnv", env_creator)

        # Model config with defaults, overridden by tunable params
        model_config = {
            "gine_hidden_dims": self.model_hparams.get("gine_hidden_dims", [512, 512, 512]),
            "gine_mlp_layers": self.model_hparams.get("gine_mlp_layers", 3),
            "gine_mlp_activation": self.model_hparams.get("gine_mlp_activation", "relu"),
            "pooling": self.model_hparams.get("pooling", "max"),
            "fc_hidden_dims": self.model_hparams.get("fc_hidden_dims", [512, 512]),
            "fc_activation": self.model_hparams.get("fc_activation", "relu"),
            "norm": self.model_hparams.get("norm", "graph"),
            "dropout": self.model_hparams.get("dropout", 0.02),
        }

        module_spec = RLModuleSpec(
            module_class=GINETorchRLModule,
            observation_space=self.env.observation_space,
            action_space=self.action_space,
            model_config=model_config,
            inference_only=False,
            learner_only=False,
        )

        # PPO config with defaults, overridden by tunable params
        ppo_config = (
            PPOConfig()
            .api_stack(
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False,
            )
            .environment("SimulationEnv", env_config=self.agent_config)
            .rl_module(rl_module_spec=module_spec)
            .training(
                train_batch_size=self.ppo_hparams.get("train_batch_size", 4096),
                minibatch_size=self.ppo_hparams.get("minibatch_size", 1024),
                num_epochs=self.ppo_hparams.get("num_epochs", 2),
                gamma=self.ppo_hparams.get("gamma", 0.99),
                lr=self.ppo_hparams.get("lr", 1e-4),
                entropy_coeff=self.ppo_hparams.get("entropy_coeff", 0.01),
                clip_param=self.ppo_hparams.get("clip_param", 0.3),
                vf_clip_param=self.ppo_hparams.get("vf_clip_param", 10.0),
                lambda_=self.ppo_hparams.get("lambda_", 0.95),
            )
            .env_runners(
                num_cpus_per_env_runner=1,
                num_gpus_per_env_runner=0,
                num_env_runners=max(0, self.cpu_count - 1),
                num_envs_per_env_runner=self.ppo_hparams.get("num_envs_per_env_runner", 8),
                batch_mode="truncate_episodes",
                rollout_fragment_length="auto",
            )
            .learners(num_learners=0)
            .resources(
                num_cpus_for_main_process=1,
                num_gpus=self.gpu_count,
            )
            .framework("torch")
        )

        return ppo_config.build_algo()

    def train(self) -> str:
        try:
            checkpoint = super().train()
            return checkpoint
        finally:
            if self.writer is not None:
                self.writer.flush()
                self.writer.close()

    @override
    def log_train_results(self, iteration: int, max_iter: int, iter_done: int, total_elapsed_time: float, iteration_elapsed_time: float, result: Dict):
        super().log_train_results(iteration, max_iter, iter_done, total_elapsed_time, iteration_elapsed_time, result)

        if self.writer is None:
            return

        # Rewards are under 'env_runners'
        env_stats = result.get("env_runners", {})
        reward_mean = env_stats.get("episode_reward_mean", env_stats.get("episode_return_mean", 0))
        reward_min = env_stats.get("episode_reward_min", env_stats.get("episode_return_min", 0))
        reward_max = env_stats.get("episode_reward_max", env_stats.get("episode_return_max", 0))

        self.writer.add_scalar("reward/mean", reward_mean, iteration)
        self.writer.add_scalar("reward/min", reward_min, iteration)
        self.writer.add_scalar("reward/max", reward_max, iteration)
        self.writer.add_scalar("num_env_steps_sampled", result.get("num_env_steps_sampled", 0), iteration)

        # Timers - compatible with Ray 2.44+ (in seconds)
        timers = result.get("timers", {})
        sample_time_s = (
                timers.get("env_runner_sampling_timer")
                or timers.get("sample_time_ms", 0) / 1000.0
        )
        learn_time_s = (
                timers.get("learner_update_timer")
                or timers.get("learn_time_ms", 0) / 1000.0
        )
        self.writer.add_scalar("time/sample_s", sample_time_s, iteration)
        self.writer.add_scalar("time/learn_s", learn_time_s, iteration)
        self.writer.add_scalar("time/iteration_s", iteration_elapsed_time, iteration)

        # Learner stats - Ray 2.44+ uses result["learners"]["default_learner"]
        learner_stats = {}
        learners = result.get("learners", {})
        default_learner = learners.get("default_learner", {})
        if default_learner:
            learner_stats = default_learner
        else:
            try:
                learner_stats = result["info"]["learner"]["default_policy"]["learner_stats"]
            except (KeyError, TypeError):
                try:
                    learner_stats = result["info"]["learner"]["default_policy"]
                except (KeyError, TypeError):
                    try:
                        learner_stats = result.get("info", {}).get("learner", {})
                    except (KeyError, TypeError):
                        pass

        if learner_stats:
            policy_loss = learner_stats.get("policy_loss", learner_stats.get("mean_policy_loss", 0))
            vf_loss = learner_stats.get("vf_loss", learner_stats.get("mean_vf_loss", 0))
            entropy = learner_stats.get("entropy", learner_stats.get("mean_entropy", 0))
            kl = learner_stats.get("kl", learner_stats.get("mean_kl_loss", 0))

            self.writer.add_scalar("loss/policy", policy_loss, iteration)
            self.writer.add_scalar("loss/vf", vf_loss, iteration)
            self.writer.add_scalar("loss/entropy", entropy, iteration)
            self.writer.add_scalar("kl_divergence", kl, iteration)

        timesteps = result.get("timesteps_total", result.get("num_env_steps_trained", 0))

        self.logger.info(
            f"Iter {iteration} | reward_mean: {reward_mean:.2f} | "
            f"timesteps: {timesteps}"
        )