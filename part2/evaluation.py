#!/usr/bin/env python3
# encoding: utf-8
"""
Code used to load an agent and evaluate its performance.

Usage:
    python3 evaluation.py -h

"""

import argparse
import importlib.util
import logging
import sys
from math import sqrt

import pygame

from utils import create_environment

logger = logging.getLogger(__name__)


def evaluate(env, predict_function, seed_games):
    rewards = {agent: 0 for agent in env.possible_agents}
    rewards_per_game = {agent: [] for agent in env.possible_agents}
    do_terminate = False

    for i in seed_games:
        env.reset(seed=i)
        env.action_space(env.possible_agents[0]).seed(i)

        rew = {agent: 0 for agent in env.possible_agents}
        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            for a in env.agents:
                rewards[a] += env.rewards[a]
                rew[a] += env.rewards[a]

            if termination or truncation:
                break
            else:
                action = predict_function(obs, agent)
            if env.render_mode == "human":
                events = (
                    pygame.event.get()
                )  # This is required to prevent the window from freezing
                for event in events:
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            pygame.quit()
                            do_terminate = True
                if do_terminate:
                    break
            env.step(action)
            if do_terminate:
                break
        print(f"{i} reward for game: {sum(rew.values())}")
        for agent in env.possible_agents:
            rewards_per_game[agent].append(rew[agent])
        if do_terminate:
            break
    env.close()

    n_games = len(seed_games)
    avg_reward = sum(rewards.values()) / n_games
    avg_reward_per_agent = {
        agent: rewards[agent] / n_games for agent in env.possible_agents
    }
    std_per_agent = {
        agent: sqrt(
            sum((r - avg_reward_per_agent[agent]) ** 2 for r in rewards_per_game[agent])
            / n_games
        )
        for agent in env.possible_agents
    }
    print(f"Standard deviation per agent: {std_per_agent}")

    print(f"Avg reward: {avg_reward}")
    print("Avg reward per agent, per game: ", avg_reward_per_agent)
    return avg_reward, avg_reward_per_agent, std_per_agent


def main(argv=None):
    parser = argparse.ArgumentParser(description="Load an agent and play the KAZ game.")
    parser.add_argument(
        "--verbose", "-v", action="count", default=0, help="Verbose output"
    )
    parser.add_argument("--quiet", "-q", action="count", default=0, help="Quiet output")
    parser.add_argument(
        "--load",
        "-l",
        help=("Load from the given file, otherwise use rllib_student_code_to_submit."),
    )
    parser.add_argument(
        "--screen",
        "-s",
        action="store_true",
        help="Set render mode to human (show game)",
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        help="Path to the checkpoint directory relative to the package directory (e.g., 'runs/grid_search_20240307_123456/c2000_yKL_r256_with_arrows/weights/best_20240307_123456_990_single_ppo_85.555')",
    )
    args = parser.parse_args(argv)

    logger.setLevel(max(logging.INFO - 10 * (args.verbose - args.quiet), logging.DEBUG))
    logger.addHandler(logging.StreamHandler(sys.stdout))

    num_agents = 1
    visual_observation = False
    render_mode = "human" if args.screen else None  # "human" or None
    logger.info(f"Show game: {render_mode}")
    if render_mode == "human":
        logger.info("Press q to end game")
    logger.info(f"Use pixels: {visual_observation}")

    # Loading student submitted code
    if args.load is not None:
        print("loading agent from args")
        spec = importlib.util.spec_from_file_location("KAZ_agent", args.load)
        Agent = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(Agent)
        print(Agent)
        CustomWrapper = Agent.CustomWrapper
        CustomPredictFunction = Agent.CustomPredictFunction
    else:
        print("loading agent from submission_single_example_rllib")
        from diagonal_agent import CustomPredictFunction, CustomWrapper

    # Create the PettingZoo environment for evaluation (with rendering)
    env = create_environment(
        num_agents=num_agents,
        render_mode=render_mode,
        visual_observation=visual_observation,
        max_cycles=100000000,
    )

    env = CustomWrapper(env)

    # Loading best checkpoint and evaluating
    random_seeds = list(range(100, 150))
    reward = evaluate(
        env,
        CustomPredictFunction(env, checkpoint_path=args.checkpoint),
        seed_games=random_seeds,
    )
    print(args.checkpoint)
    # checkpoint_parts = re.split(r"/|\\", args.checkpoint)
    # checkpoint_name = checkpoint_parts[-1]
    # eval_scores_dir = Path("eval_scores")
    # eval_scores_dir.mkdir(parents=True, exist_ok=True)
    # # Perform the split outside the f-string to avoid escape sequence issues
    # with open(
    #     eval_scores_dir
    #     / f"{checkpoint_parts[-3]}__{checkpoint_name}____{reward[0]}.txt",
    #     "w",
    # ) as f:
    #     f.write(str(reward))
    # print(f"config: {args.checkpoint}")


if __name__ == "__main__":
    sys.exit(main())
