import signal
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import itertools

import numpy as np
import torch
from gymnasium import spaces
from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.env import AgentID
from ray.rllib.core.rl_module import MultiRLModule
from tqdm import tqdm

import matplotlib.pyplot as plt
import pettingzoo
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
from ray.rllib.core.rl_module import MultiRLModuleSpec, RLModuleSpec
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.examples.rl_modules.classes.random_rlm import RandomRLModule
from ray.tune.registry import register_env

from utils import create_environment

package_directory = Path(__file__).resolve().parent


class CustomWrapperNoArrows(BaseWrapper):
    def __init__(self, env, vis_obs=False, max_zombies=4):
        super().__init__(env)
        self.vis_obs = vis_obs
        self.max_zombies = max_zombies

    def observation_space(self, agent: AgentID) -> spaces.Space:
        original_space = super().observation_space(agent)
        # if the full image is used as observation, return the original space
        if self.vis_obs:
            return spaces.Box(
                low=0.0, high=1.0, shape=original_space.shape, dtype=np.float32
            )

        size = (1 + self.max_zombies) * 8

        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(size,),
            dtype=np.float32,
        )

    def observe(self, agent: AgentID) -> Optional[np.ndarray]:
        raw_obs = super().observe(agent)
        if raw_obs is None:
            return None
        if self.vis_obs:
            return raw_obs.astype(np.float32) / 255.0

        agent_row = raw_obs[:1]
        zombie_rows = raw_obs[-self.max_zombies :]

        ax, ay = agent_row[0, 1], agent_row[0, 2]

        # Pad agent_row with zeros to match the max_zombies size
        agent_row = np.concatenate((agent_row, np.zeros((1, 3))), axis=1)

        new_features = np.zeros((self.max_zombies, 3), dtype=np.float32)
        for i, zrow in enumerate(zombie_rows):
            if zrow[0] > 0:
                zx = ax + zrow[1]
                zy = ay + zrow[2]

                edge_distance = 1.0 - zy  # Distance to bottom edge
                lateral_position = zx - 0.5
                urgency = edge_distance / 0.05

                new_features[i] = [edge_distance, lateral_position, urgency]

        zombie_rows = np.concatenate((zombie_rows, new_features), axis=1)

        combined_obs = np.concatenate(
            (agent_row, zombie_rows), axis=0, dtype=np.float32
        ).flatten()

        return combined_obs


class CustomWrapperWithArrows(BaseWrapper):
    def __init__(self, env, max_zombies=4):
        super().__init__(env)
        self.max_zombies = max_zombies

    def observation_space(self, agent: AgentID) -> spaces.Space:
        original_space = super().observation_space(agent)

        original_size = np.prod(original_space.shape)
        new_features_size = 0 + self.max_zombies * 3

        return spaces.Box(
            low=-1,
            high=255,
            shape=(original_size + new_features_size,),
            dtype=np.float32,
        )

    def observe(self, agent: AgentID) -> Optional[np.ndarray]:
        raw_obs = super().observe(agent)
        if raw_obs is None:
            return None

        original_flat = raw_obs.flatten()

        agent_row = raw_obs[0]
        zombie_rows = raw_obs[-self.max_zombies :]

        ax, ay = agent_row[1], agent_row[2]

        zombie_features = []
        for zrow in zombie_rows:
            if zrow[0] > 0:
                zx = ax + zrow[1]
                zy = ay + zrow[2]

                edge_distance = 1.0 - zy  # Distance to bottom edge
                lateral_position = zx - 0.5
                urgency = edge_distance / 0.05

                zombie_features += [edge_distance, lateral_position, urgency]
            else:
                zombie_features += [0.0, 0.0, 0.0]

        new_features = [*zombie_features]

        combined_obs = np.concatenate(
            [original_flat.astype(np.float32), np.array(new_features, dtype=np.float32)]
        )

        return combined_obs


def algo_config_ppo(id_env, policies, policies_to_train, config_params):
    return (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .environment(
            env=id_env,
            disable_env_checking=True,
            env_config={"vector_state": True},
        )
        .env_runners(
            num_env_runners=8,
            explore=True,
            rollout_fragment_length=config_params["rollout_length"],
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id,
            policies_to_train=policies_to_train,
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    x: RLModuleSpec(
                        module_class=PPOTorchRLModule,
                        model_config=DefaultModelConfig(
                            fcnet_hiddens=[256, 128],
                            fcnet_activation="relu",
                            vf_share_layers=False,
                        ),
                    )
                    if x in policies_to_train
                    else RLModuleSpec(module_class=RandomRLModule)
                    for x in policies
                },
            )
        )
        .training(
            use_critic=True,
            use_gae=True,
            train_batch_size=config_params["rollout_length"] * 8,
            num_sgd_iter=10,
            lr=[[0, 3e-4], [1.5e6, 1e-4], [3e6, 5e-5], [4e6, 1e-5]],
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            entropy_coeff=[[0, 0.015], [1e6, 0.010], [2.5e6, 0.005], [3.5e6, 0.001]],
            vf_loss_coeff=0.5,
            vf_clip_param=10.0,
            use_kl_loss=config_params["use_kl_loss"],
            kl_coeff=0.2 if config_params["use_kl_loss"] else 0.0,
            kl_target=0.02 if config_params["use_kl_loss"] else 0.0,
            grad_clip=0.5,
        )
        .resources(
            num_gpus=1 if torch.cuda.is_available() else 0,
        )
        .debugging(log_level="ERROR")
    )


stop_training = False


def signal_handler(sig, frame):
    global stop_training
    print("Soft exit triggered - training will stop after current iteration")
    stop_training = True


def training(env, ts, max_iterations, runs_dir, config_params):
    rllib_env = ParallelPettingZooEnv(pettingzoo.utils.conversions.aec_to_parallel(env))
    id_env = "knights_archers_zombies_v10"
    register_env(id_env, lambda config: rllib_env)
    global stop_training
    signal.signal(signal.SIGINT, signal_handler)

    np.random.seed(42)
    torch.manual_seed(42)

    policies = [x for x in env.agents]
    policies_to_train = policies
    config = algo_config_ppo(id_env, policies, policies_to_train, config_params)

    with open(runs_dir / "config.txt", "w") as f:
        f.write(str(config.to_dict()))

    algo = config.build()
    best_score = 0
    history = []

    path_history = str(runs_dir / f"history_single_ppo_{ts}.npy")

    weights_dir = runs_dir / "weights"
    weights_dir.mkdir(exist_ok=True)

    for i in tqdm(range(max_iterations), desc="Training Iterations"):
        if stop_training:
            print("Stopping training loop")
            break

        torch.manual_seed(int(time.time()))
        np.random.seed(int(time.time()))

        result = algo.train()
        result.pop("config")

        episod_return = result["env_runners"]["agent_episode_returns_mean"]
        print(f"{i}: {episod_return}")
        history.append(episod_return["archer_0"])
        np.save(path_history, np.array(history))

        if best_score < episod_return["archer_0"]:
            best_score = episod_return["archer_0"]

            path_best = (
                weights_dir / f"best_{ts}_{i}_single_ppo_{episod_return['archer_0']}"
            ).resolve()
            save_result = algo.save(path_best)
            path_to_checkpoint = save_result.checkpoint.path
            print(f"Save best checkpoint '{path_to_checkpoint}'.")

        if i % 10 == 0:
            checkpoint_path = str(
                weights_dir / f"results_{ts}_{i}_single_ppo_{episod_return['archer_0']}"
            )
            save_result = algo.save(checkpoint_path)
            path_to_checkpoint = save_result.checkpoint.path
            print(f"Save checkpoint {path_to_checkpoint}")

    # cleaning up the algo
    algo.stop()
    return history


def run_grid_search():
    param_grid = {
        "max_cycles": [2000, 1000000],
        "use_kl_loss": [True, False],
        "rollout_length": [256, 512],
        "use_arrows": [True, False],
        "max_iterations": [1000],
    }

    param_combinations = [
        dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())
    ]

    ts = time.time()
    time_ = datetime.fromtimestamp(ts)
    time_ = time_.strftime("%Y%m%d_%H%M%S")
    base_runs_dir = package_directory / "runs" / f"grid_search_{time_}"
    base_runs_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for i, config in enumerate(param_combinations):
        print(f"\nRunning configuration: {config} (Iteration {i + 1})")

        env = create_environment(
            num_agents=1,
            visual_observation=False,
            max_cycles=config["max_cycles"],
        )

        if config["use_arrows"]:
            env = CustomWrapperWithArrows(env)
        else:
            env = CustomWrapperNoArrows(env)

        config_name = f"c{config['max_cycles']}_{'y' if config['use_kl_loss'] else 'n'}KL_r{config['rollout_length']}_{'with' if config['use_arrows'] else 'no'}_arrows"
        runs_dir = base_runs_dir / config_name
        runs_dir.mkdir(exist_ok=True)

        history = training(env, ts, config["max_iterations"], runs_dir, config)

        results.append({"config": config, "history": history})

        np.save(
            str(runs_dir / "results.npy"),
            {"config": config, "history": history},
        )
        plt.plot(history)
        plt.savefig(runs_dir / "history.png")
        plt.close()


if __name__ == "__main__":
    run_grid_search()
