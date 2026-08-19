import gymnasium as gym
import numpy as np


class IntentionRewardWrapper(gym.Wrapper):
    def __init__(self, env, coeff=0.5, gamma=0.99):
        super().__init__(env)
        self.coeff = coeff
        self.gamma = gamma
        self.prev_cos = None  # 上一步势能 Φ(s)
        print("coeff:",self.coeff)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_cos = self._get_cos_angle()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        cur_cos = self._get_cos_angle()

        if cur_cos is not None and self.prev_cos is not None:
            intention_reward = self.coeff * (self.gamma * cur_cos - self.prev_cos)
        else:
            intention_reward = 0.0

        total_reward = reward + intention_reward
        info['intention_reward'] = intention_reward
        info['phi'] = cur_cos

        if cur_cos is not None:
            self.prev_cos = cur_cos

        return obs, total_reward, terminated, truncated, info

    def _get_cos_angle(self):
        """Φ(s) = (cosθ + 1)/2，和原始朴素版本势能保持一致"""
        agent_pos = self._get_agent_pos()
        agent_dir = self._get_agent_dir()
        target_pos = self._get_target_pos()

        if agent_pos is None or agent_dir is None or target_pos is None:
            return None

        dir_to_target = np.array(target_pos) - np.array(agent_pos)
        dist = np.linalg.norm(dir_to_target)
        if dist < 1e-6:
            return 1.0  # 到达目标，Φ=1
        dir_to_target = dir_to_target / dist

        dir_vectors = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        agent_dir_vec = np.array(dir_vectors[agent_dir])

        cos_angle = np.dot(agent_dir_vec, dir_to_target)
        phi = (cos_angle - 1.0) / 2.0   # ← 唯一改动
        return phi

    def _get_agent_pos(self):
        if hasattr(self.unwrapped, 'agent_pos'):
            return self.unwrapped.agent_pos
        return None

    def _get_agent_dir(self):
        if hasattr(self.unwrapped, 'agent_dir'):
            return self.unwrapped.agent_dir
        return None

    def _get_target_pos(self):
        if hasattr(self.unwrapped, 'grid'):
            grid = self.unwrapped.grid
            for i in range(grid.width):
                for j in range(grid.height):
                    cell = grid.get(i, j)
                    if cell is not None and cell.type == 'goal':
                        return (i, j)
        return None
