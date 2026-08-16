import gymnasium as gym

from bfs_oracle import BFSOracle


class PotentialBFSRewardWrapper(gym.Wrapper):
    """Potential shaping from exact left/right/forward shortest-path steps."""

    def __init__(self, env, gamma, shaping_coeff=1.0):
        super().__init__(env)
        self.gamma = float(gamma)
        self.shaping_coeff = float(shaping_coeff)
        self.oracle = None

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.oracle = BFSOracle.from_env(self)
        return observation, info

    def potential(self):
        distance = self.oracle.distance(
            self.unwrapped.agent_pos,
            self.unwrapped.agent_dir,
        )
        return -distance / self.oracle.max_distance

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
