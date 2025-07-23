from pathlib import Path
from typing import Optional

import numpy as np
import torch
from gymnasium import spaces
from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.env import AgentID
from ray.rllib.core.rl_module import MultiRLModule


class CustomWrapper(BaseWrapper):
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


class CustomPredictFunction:
    def __init__(self, env, checkpoint_path: str = None):
        package_directory = Path(__file__).resolve().parent

        if checkpoint_path is None:
            # Default checkpoint path
            checkpoint_path = "weights/best_20250506_235745_1490_single_ppo_85.555"

        # Construct the full path
        full_checkpoint_path = (
            package_directory
            / checkpoint_path
            / "learner_group"
            / "learner"
            / "rl_module"
        ).resolve()

        print(f"Loading checkpoint from: {full_checkpoint_path}")
        self.cp = full_checkpoint_path

        if not full_checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint directory not found: {full_checkpoint_path}"
            )

        self.modules = MultiRLModule.from_checkpoint(full_checkpoint_path)

    def __call__(self, observation, agent, *args, **kwargs):
        if agent not in self.modules:
            raise ValueError(f"No policy found for agent {agent}")

        rl_module = self.modules[agent]
        fwd_ins = {"obs": torch.Tensor(observation).unsqueeze(0)}
        fwd_outputs = rl_module.forward_inference(fwd_ins)
        action_dist_class = rl_module.get_inference_action_dist_cls()
        action_dist = action_dist_class.from_logits(fwd_outputs["action_dist_inputs"])
        # action = torch.argmax(action_dist.logits, dim=-1).item()

        action = action_dist.sample()[0].numpy()
        return action
