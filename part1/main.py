import random

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from game import *
from policy import (
    BoltzmannPolicy,
    EpsilonGreedyPolicy,
    LenientBoltzmannPolicy,
    QLearningPolicy,
)
from visualization import configure_figure, plot_vector_field, visualize_policy_traces

matplotlib.use("Qt5Agg")


def train(
    rewards_mat,
    policy: QLearningPolicy,
    episodes,
    training_runs,
):
    window_size = 20
    histories = []
    for i in range(training_runs):
        print(f"Training run: {i}")
        q_val_p1 = [random.random(), random.random()]
        q_val_p2 = [random.random(), random.random()]
        history = {"policy_p1": [], "policy_p2": []}

        for _ in range(episodes):
            policy.update_step(rewards_mat, q_val_p1, q_val_p2)

            policy_p1 = policy.get_action_probabilities(q_val_p1)
            policy_p2 = policy.get_action_probabilities(q_val_p2)
            history["policy_p1"].append(policy_p1)
            history["policy_p2"].append(policy_p2)

        df_p1 = pd.DataFrame(history["policy_p1"])
        df_p2 = pd.DataFrame(history["policy_p2"])

        history["policy_p1"] = (
            df_p1.rolling(window=window_size, min_periods=1).mean().values.tolist()
        )
        history["policy_p2"] = (
            df_p2.rolling(window=window_size, min_periods=1).mean().values.tolist()
        )

        histories.append(history)

    return histories


def run_experiment(game: Game, policy: QLearningPolicy):
    print(f"Starting training for qlearning: {policy.get_name()} and game: {game.name}")
    history = train(
        rewards_mat=game.rewards,
        policy=policy,
        episodes=1000,
        training_runs=30,
    )

    _, ax = plt.subplots(figsize=(12, 8))

    plot_vector_field(policy, game.rewards, ax=ax, grid_size=20)

    visualize_policy_traces(history, ax=ax)
    configure_figure(policy, game, ax)


def run_all_policies(game: Game, temp, lr):
    run_experiment(game, EpsilonGreedyPolicy(0.2, lr))
    run_experiment(game, BoltzmannPolicy(temp, lr))
    run_experiment(game, LenientBoltzmannPolicy(temp, lr, 1))
    run_experiment(game, LenientBoltzmannPolicy(temp, lr, 2))
    run_experiment(game, LenientBoltzmannPolicy(temp, lr, 5))
    run_experiment(game, LenientBoltzmannPolicy(temp, lr, 25))


def run_all_games(temp, lr):
    run_all_policies(StagHunt(), temp, lr)
    run_all_policies(PrisonersDilemma(), temp, lr)
    run_all_policies(MatchingPennies(), temp, lr)
    run_all_policies(SubsidyGame(), temp, lr)


if __name__ == "__main__":
    lr = 0.01
    temp = 0.1
    run_all_games(temp, lr)
    # learning = EpsilonGreedyPolicy(0.2, lr)
    # qlearning = BoltzmannPolicy(0.1, lr)
    # qlearning = LenientBoltzmannPolicy(0.1, lr, 2)
    # run_experiment(StagHunt(), qlearning)
    # run_experiment(PrisonersDilemma())
    # run_experiment(MatchingPennies())
    # run_experiment(SubsidyGame())
    # plt.show()
    plt.show()
