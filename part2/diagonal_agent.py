import gymnasium
from gymnasium import spaces
from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.env import AgentID, ObsType


class CustomWrapper(BaseWrapper):
    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return spaces.flatten_space(super().observation_space(agent))

    def observe(self, agent: AgentID) -> ObsType | None:
        obs = super().observe(agent)
        flat_obs = obs.flatten()
        return flat_obs


class CustomPredictFunction:
    def __init__(self, env, checkpoint_path: str = None):
        self.env = env

    def __call__(self, observation, agent, *args, **kwargs):
        curr_pos_x, curr_pos_y = observation[1], observation[2]
        curr_rotation = observation[3]

        if curr_pos_y < 0.93:
            return 1
        if curr_rotation < 1.0 and curr_pos_x < 0.89:
            return 3
        if curr_pos_x < 0.89:
            return 0
        if curr_rotation > -1.0:
            return 3

        return 4
