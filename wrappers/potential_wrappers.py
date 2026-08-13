import math

import gymnasium as gym
import numpy as np

from .state_graph import shortest_path_table, traversable_positions


def find_goal(env):
    grid = env.unwrapped.grid
    for x in range(grid.width):
        for y in range(grid.height):
            cell = grid.get(x, y)
            if cell is not None and cell.type == "goal":
                return (x, y)
    raise RuntimeError("Goal cell not found")


class PotentialRewardWrapper(gym.Wrapper):
    """Base wrapper for lambda * (gamma * Phi(s') - Phi(s))."""

    def __init__(self, env, gamma, shaping_coeff=0.2):
        super().__init__(env)
        self.gamma = float(gamma)
        self.shaping_coeff = float(shaping_coeff)
        self.goal_position = None

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.goal_position = find_goal(self)
        self._prepare_layout()
        return observation, info

    def _prepare_layout(self):
        pass

    def potential(self):
        raise NotImplementedError

    def step(self, action):
        previous_phi = self.potential()
        observation, reward, terminated, truncated, info = self.env.step(action)
        # A true terminal state is absorbing and has zero potential. A time-limit
        # truncation is not terminal, so retain the actual successor potential.
        next_phi = 0.0 if terminated else self.potential()
        shaping = self.shaping_coeff * (self.gamma * next_phi - previous_phi)
        info = dict(info)
        info["potential_before"] = float(previous_phi)
        info["potential_after"] = float(next_phi)
        info["shaping_reward"] = float(shaping)
        return observation, reward + shaping, terminated, truncated, info


class GeometricPotentialWrapper(PotentialRewardWrapper):
    def _prepare_layout(self):
        distances = [
            math.dist(position, self.goal_position)
            for position in traversable_positions(self)
        ]
        self.normalizer = max(max(distances, default=1.0), 1.0)

    def potential(self):
        return -math.dist(tuple(self.unwrapped.agent_pos), self.goal_position) / self.normalizer


class WrongDirectionPotentialWrapper(GeometricPotentialWrapper):
    def potential(self):
        return math.dist(tuple(self.unwrapped.agent_pos), self.goal_position) / self.normalizer


class OptimalPathPotentialWrapper(PotentialRewardWrapper):
    def _prepare_layout(self):
        self.distances, self.optimal_actions, self.normalizer = shortest_path_table(
            self, self.goal_position
        )
        self.normalizer = max(self.normalizer, 1)

    def potential(self):
        state = (tuple(int(x) for x in self.unwrapped.agent_pos), int(self.unwrapped.agent_dir))
        return -float(self.distances[state]) / self.normalizer
