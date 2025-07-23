import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from game import *
from policy import BoltzmannPolicy, EpsilonGreedyPolicy, QLearningPolicy
from replicator_dynamic import compute_replicator_field


def visualize_policy_traces(histories, ax):
    labels = [f"Run {i + 1}" for i in range(len(histories))]
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i / len(histories)) for i in range(len(histories))]

    for i, history in enumerate(histories):
        p1_policies = history["policy_p1"]
        p2_policies = history["policy_p2"]
        p1_probs = [policy[0] for policy in p1_policies]
        p2_probs = [policy[0] for policy in p2_policies]

        ax.plot(p1_probs, p2_probs, linewidth=2, color=colors[i], label=labels[i])

        ax.scatter(
            p1_probs[0],
            p2_probs[0],
            s=100,
            marker="o",
            color=colors[i],
            edgecolor="black",
            linewidth=1.5,
            zorder=10,
        )

        ax.scatter(
            p1_probs[-1],
            p2_probs[-1],
            s=150,
            marker="*",
            color=colors[i],
            edgecolor="black",
            linewidth=1.5,
            zorder=10,
        )


def plot_vector_field(policy: BoltzmannPolicy, rewards, ax, grid_size):
    if isinstance(policy, EpsilonGreedyPolicy):
        return
    X, Y, U, V = compute_replicator_field(rewards, grid_size, policy)
    magnitudes = np.sqrt(U**2 + V**2)

    ax.quiver(X, Y, U, V, magnitudes, width=0.002, pivot="tail")


def configure_figure(policy: QLearningPolicy, game: Game, ax):
    if isinstance(policy, EpsilonGreedyPolicy):
        ax.set_title(f"Empirical policy traces {game.name} ({policy.get_name()})")
        ax.set_xlabel(
            f"Normalized Q-values for player 1 for action {game.action_names[0]}"
        )
        ax.set_ylabel(
            f"Normalized Q-values for player 2 for action {game.action_names[0]}"
        )
    else:
        ax.set_title(
            f"Empirical policy traces {game.name} with overlayed replicator dynamics ({policy.get_name()})"
        )
        ax.set_xlabel(
            f"Probability for player 2 of selecting action {game.action_names[0]}"
        )
        ax.set_ylabel(
            f"Probability for player 2 of selecting action {game.action_names[0]}"
        )

    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.grid(True)

    begin_point = mlines.Line2D(
        [],
        [],
        marker="o",
        linestyle="None",
        markersize=10,
        markeredgecolor="black",
        markeredgewidth=1.5,
        label="Beginning of trace",
    )

    end_point = mlines.Line2D(
        [],
        [],
        marker="*",
        linestyle="None",
        markersize=12,
        markeredgecolor="black",
        markeredgewidth=1.5,
        label="End of trace",
    )

    handles, labels = ax.get_legend_handles_labels()

    handles = [begin_point, end_point]
    labels = ["Beginning of trace", "End of trace"]

    ax.legend(handles=handles, labels=labels, loc="lower right")
    plt.savefig(f"./plots/replicator_trajectoreis_{game.name}_{policy.get_name()}.png")


# def visualize_training(histories, action_names, game_name):
#     plt.figure(figsize=(15, 12))
#     labels = [f"Run {i + 1}" for i in range(len(histories))]
#     cmap = plt.get_cmap("tab10")
#     colors = [cmap(i / len(histories)) for i in range(len(histories))]
#
#     # Player 1 Q-values
#     plt.subplot(2, 1, 1)
#     for run_idx, history in enumerate(histories):
#         episodes = history["episodes"]
#         q_values_p1 = np.array(history["q_values_p1"])
#
#         for i in range(2):
#             linestyle = "-" if i == 0 else "--"
#             plt.plot(
#                 episodes,
#                 q_values_p1[:, i],
#                 linestyle=linestyle,
#                 color=colors[run_idx],
#                 label=f"{labels[run_idx]}: {action_names[i]}",
#             )
#
#     plt.title(f"Player 1 Q-Values Over Time ({game_name})")
#     plt.xlabel("Episodes")
#     plt.ylabel("Q-Value")
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#
#     # Player 2 Q-values
#     plt.subplot(2, 1, 2)
#     for run_idx, history in enumerate(histories):
#         episodes = history["episodes"]
#         q_values_p2 = np.array(history["q_values_p2"])
#
#         for i in range(2):
#             linestyle = "-" if i == 0 else "--"
#             plt.plot(
#                 episodes,
#                 q_values_p2[:, i],
#                 linestyle=linestyle,
#                 color=colors[run_idx],
#                 label=f"{labels[run_idx]}: {action_names[i]}",
#             )
#
#     plt.title(f"Player 2 Q-Values Over Time ({game_name})")
#     plt.xlabel("Episodes")
#     plt.ylabel("Q-Value")
#     plt.legend()
#     plt.grid(True, alpha=0.3)
