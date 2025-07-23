import numpy as np
from policy import *


def leniant_boltzmann_derivative(A, policy, y_policy):
    additional_term = [0, 0]
    for i in range(2):
        for j in range(2):
            a_ij = A[i, j]
            y_j = y_policy[j]

            S_leq = sum(y_policy[k] for k in range(2) if A[i, k] <= a_ij)
            S_lt = sum(y_policy[k] for k in range(2) if A[i, k] < a_ij)
            S_eq = sum(y_policy[k] for k in range(2) if A[i, k] == a_ij)

            if S_eq != 0:
                diff_term = (S_leq**policy.K - S_lt**policy.K) / S_eq
                additional_term[i] += a_ij * y_j * diff_term
    Ay = additional_term
    return Ay


def get_derivative(policy, A, x_policy, y_policy, x_prob):
    if isinstance(policy, LenientBoltzmannPolicy):
        Ay = leniant_boltzmann_derivative(A, policy, y_policy)
    else:
        Ay = A @ y_policy
    dx = (
        policy.LR
        * x_prob
        * (
            ((Ay)[0] - (x_policy.T @ Ay)) / policy.TEMPERATURE
            - (
                np.log(x_policy[0])
                - x_policy[0] * np.log(x_policy[0])
                - x_policy[1] * np.log(x_policy[1])
            )
        )
    )
    return dx


def compute_replicator_field(rewards, grid_size, policy):
    x_policy = np.linspace(0.01, 0.99, grid_size)
    y_policy = np.linspace(0.01, 0.99, grid_size)
    X, Y = np.meshgrid(x_policy, y_policy)

    A = np.zeros((2, 2))
    B = np.zeros((2, 2))

    for x in range(2):
        for y in range(2):
            A[x, y] = rewards[x][y][0]
            B[x, y] = rewards[x][y][1]

    B = B.T
    DX = np.zeros_like(X)
    DY = np.zeros_like(Y)

    for x in range(grid_size):
        for y in range(grid_size):
            x_prob = X[x, y]
            y_prob = Y[x, y]

            x_policy = np.array([x_prob, 1 - x_prob])
            y_policy = np.array([y_prob, 1 - y_prob])

            # if isinstance(policy, LenientBoltzmannPolicy):
            #     additional_term = [0, 0]
            #     for i in range(2):
            #         for j in range(2):
            #             a_ij = A[i, j]
            #             y_j = y_policy[j]
            #
            #             S_leq = sum(y_policy[k] for k in range(2) if A[i, k] <= a_ij)
            #             S_lt = sum(y_policy[k] for k in range(2) if A[i, k] < a_ij)
            #             S_eq = sum(y_policy[k] for k in range(2) if A[i, k] == a_ij)
            #
            #             if S_eq != 0:
            #                 diff_term = (S_leq**policy.K - S_lt**policy.K) / S_eq
            #                 additional_term[i] += a_ij * y_j * diff_term
            #     Ay = additional_term
            # else:
            #     Ay = A @ y_policy
            # dx = (
            #     policy.LR
            #     * x_prob
            #     * (
            #         ((Ay)[0] - (x_policy.T @ Ay)) / policy.TEMPERATURE
            #         - (
            #             np.log(x_policy[0])
            #             - x_policy[0] * np.log(x_policy[0])
            #             - x_policy[1] * np.log(x_policy[1])
            #         )
            #     )
            # )
            # if isinstance(policy, LenientBoltzmannPolicy):
            #     additional_term = [0, 0]
            #     for i in range(2):
            #         for j in range(2):
            #             b_ji = B[j, i]
            #             x_j = x_policy[j]
            #
            #             S_leq = sum(x_policy[k] for k in range(2) if B[k, i] <= b_ji)
            #             S_lt = sum(x_policy[k] for k in range(2) if B[k, i] < b_ji)
            #             S_eq = sum(x_policy[k] for k in range(2) if B[k, i] == b_ji)
            #
            #             if S_eq != 0:
            #                 diff_term = (S_leq**policy.K - S_lt**policy.K) / S_eq
            #                 additional_term[i] += b_ji * x_j * diff_term
            #     Bx = additional_term
            # else:
            #     Bx = B @ x_policy
            # dy = (
            #     policy.LR
            #     * y_prob
            #     * (
            #         ((Bx)[0] - (y_policy.T @ B @ x_policy)) / policy.TEMPERATURE
            #         - (
            #             np.log(y_policy[0])
            #             - y_policy[0] * np.log(y_policy[0])
            #             - y_policy[1] * np.log(y_policy[1])
            #         )
            #     )
            # )
            dx = get_derivative(policy, A, x_policy, y_policy, x_prob)
            dy = get_derivative(policy, B, y_policy, x_policy, y_prob)

            DX[x, y] = dx
            DY[x, y] = dy

    return X, Y, DX, DY
