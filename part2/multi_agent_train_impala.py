#!/usr/bin/env python3
# encoding: utf-8
"""
This file contains an example of training your agents using Ray RLLib.
Notice that this code will not learn properlu, as no preprocessing of the data nor adaptation of the default PPO algorithm is done here.

"""

import signal
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pettingzoo
import torch
from ray.rllib.algorithms.impala import IMPALAConfig
from ray.rllib.core.rl_module import MultiRLModuleSpec, RLModuleSpec
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

from submission_multi import CustomWrapper
from utils import create_environment


def algo_config_impala(id_env, policies, policies_to_train):
    config = (
        IMPALAConfig()
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
            num_env_runners=6,
            explore=True,
            rollout_fragment_length=256,
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_archer_policy",
            policies_to_train=policies_to_train,
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    "shared_archer_policy": RLModuleSpec(
                        model_config=DefaultModelConfig(
                            fcnet_hiddens=[512, 256, 126],
                            fcnet_activation="relu",
                            vf_share_layers=True,
                            use_lstm=False,
                            # disable conv2d for now
                            conv_filters=[],
                        ),
                    )
                },
            )
        )
        .training(
            train_batch_size=4096,
            grad_clip=1.0,
            lr=[[0, 1e-3], [10e6, 1e-4], [20e6, 1e-5]],
            gamma=0.99,
            entropy_coeff=0.01,
            vf_loss_coeff=0.5,
            vtrace=True,
            num_sgd_iter=10,
        )
        .debugging(log_level="ERROR")
    )

    return config


stop_training = False


def signal_handler(sig, frame):
    global stop_training
    print("Soft exit triggered - training will stop after current iteration")
    stop_training = True


def training(env, ts, max_iterations):
    rllib_env = ParallelPettingZooEnv(pettingzoo.utils.conversions.aec_to_parallel(env))
    id_env = "knights_archers_zombies_v10"
    register_env(id_env, lambda config: rllib_env)

    global stop_training
    signal.signal(signal.SIGINT, signal_handler)

    np.random.seed(42)
    torch.manual_seed(42)

    policies = ["shared_archer_policy"]
    policies_to_train = ["shared_archer_policy"]
    config = algo_config_impala(id_env, policies, policies_to_train)

    algo = config.build()
    best_score = 0
    history = {"archer_0": [], "archer_1": []}
    path_history = str(Path(f"history/history_multi_impala_{ts}").resolve())
    for i in range(max_iterations):
        if stop_training:
            print("Stopping training loop")
            break

        torch.manual_seed(int(time.time()))
        np.random.seed(int(time.time()))

        result = algo.train()
        result.pop("config")

        episod_return = result["env_runners"]["agent_episode_returns_mean"]
        print(f"{i}: {episod_return}")
        history["archer_0"].append(episod_return["archer_0"])
        history["archer_1"].append(episod_return["archer_1"])
        np.save(path_history, np.array(history))

        score = episod_return["archer_0"] + episod_return["archer_1"]
        if best_score < score:
            best_score = score

            path_best = str(Path(f"weights/best_multi_impala_{ts}_{i}").resolve())
            save_result = algo.save(path_best)
            path_to_checkpoint = save_result.checkpoint.path
            print(f"Best weights checkpoint: {path_to_checkpoint}")

        if i % 10 == 0:
            checkpoint_path = str(
                Path(
                    f"weights/results_{ts}_{i}_multi_impala_{episod_return['archer_0'] + episod_return['archer_1']}"
                ).resolve()
            )
            save_result = algo.save(checkpoint_path)
            path_to_checkpoint = save_result.checkpoint.path
            print(f"Save checkpoint'{path_to_checkpoint}'.")

    # cleaning up the algo
    algo.stop()
    return history


if __name__ == "__main__":
    num_agents = 2
    visual_observation = False

    env = create_environment(
        num_agents=num_agents,
        visual_observation=visual_observation
    )
    env = CustomWrapper(env, vis_obs=visual_observation, training=True)

    ts = time.time()
    time_ = datetime.fromtimestamp(ts)
    time_ = time_.strftime("%Y%m%d_%H%M%S")

    history = training(env, time_, max_iterations=1500)
    plt.plot(history["archer_0"], label="archer_0")
    plt.plot(history["archer_1"], label="archer_1")
    plt.legend()
    plt.xlabel("Epoches")
    plt.ylabel("Average rewards")
    plt.title("Performance of Archers (IMPALA)")
    plt.savefig(f"history/history_multi_impala_{ts}.png")
    # plt.show()
