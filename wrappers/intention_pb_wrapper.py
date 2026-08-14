import gymnasium as gym
import numpy as np


class PotentialIntentionRewardWrapper(gym.Wrapper):
    """Potential-based directional intention signal, raw or shifted."""

    DIR_TO_VEC = np.asarray(((1, 0), (0, 1), (-1, 0), (0, -1)), dtype=float)
    FORMS = ("raw", "shifted")

    def __init__(self, env, gamma, shaping_coeff=1.0, potential_form="raw"):
        super().__init__(env)
        if potential_form not in self.FORMS:
            raise ValueError(f"potential_form must be one of {self.FORMS}")
        self.gamma = float(gamma)
        self.shaping_coeff = float(shaping_coeff)
        self.potential_form = potential_form

    def _goal_position(self):
        grid = self.unwrapped.grid
        for x in range(grid.width):
            for y in range(grid.height):
                cell = grid.get(x, y)
                if cell is not None and cell.type == "goal":
                    return np.asarray((x, y), dtype=float)
        raise RuntimeError("Goal cell not found")

    def potential(self):
        agent_position = np.asarray(self.unwrapped.agent_pos, dtype=float)
        direction_to_goal = self._goal_position() - agent_position
        distance = np.linalg.norm(direction_to_goal)
        raw = 1.0
        if distance:
            direction_to_goal /= distance
            agent_direction = self.DIR_TO_VEC[int(self.unwrapped.agent_dir)]
            raw = (float(np.dot(agent_direction, direction_to_goal)) + 1.0) / 2.0
        return raw if self.potential_form == "raw" else raw - 1.0

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
        info["potential_form"] = self.potential_form
        return observation, reward + shaping_reward, terminated, truncated, info
