import math
from collections import deque

import gymnasium as gym


class PotentialDistanceRewardWrapper(gym.Wrapper):
    """Potential shaping with normalized negative Euclidean distance."""

    def __init__(self, env, gamma, shaping_coeff=1.0):
        super().__init__(env)
        self.gamma = float(gamma)
        self.shaping_coeff = float(shaping_coeff)
        self.goal_position = None
        self.max_distance = None

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.goal_position = self._find_goal()
        distances = [
            math.dist(position, self.goal_position)
            for position in self._reachable_positions()
        ]
        self.max_distance = max(max(distances, default=1.0), 1.0)
        return observation, info

    def _find_goal(self):
        grid = self.unwrapped.grid
        for x in range(grid.width):
            for y in range(grid.height):
                cell = grid.get(x, y)
                if cell is not None and cell.type == "goal":
                    return (x, y)
        raise RuntimeError("Goal cell not found")

    def _reachable_positions(self):
        grid = self.unwrapped.grid
        start = tuple(self.unwrapped.agent_pos)
        reachable = {start}
        frontier = deque([start])
        while frontier:
            x, y = frontier.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbor = (x + dx, y + dy)
                if neighbor in reachable:
                    continue
                nx, ny = neighbor
                if not (0 <= nx < grid.width and 0 <= ny < grid.height):
                    continue
                cell = grid.get(nx, ny)
                if cell is None or cell.can_overlap():
                    reachable.add(neighbor)
                    frontier.append(neighbor)
        return reachable

    def potential(self):
        distance = math.dist(tuple(self.unwrapped.agent_pos), self.goal_position)
        return -distance / self.max_distance

    def step(self, action):
        previous_potential = self.potential()
        observation, reward, terminated, truncated, info = self.env.step(action)
        next_potential = 0.0 if terminated else self.potential()
        shaping_reward = self.shaping_coeff * (
            self.gamma * next_potential - previous_potential
        )
        info = dict(info)
        info["potential_before"] = previous_potential
        info["potential_after"] = next_potential
        info["shaping_reward"] = shaping_reward
        return observation, reward + shaping_reward, terminated, truncated, info
