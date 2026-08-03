import gymnasium as gym
import numpy as np

class DenseDistanceWrapper(gym.Wrapper):
    """
    距离奖励：距离目标越近奖励越高
    """
    def __init__(self, env, coeff=0.1):
        super().__init__(env)
        self.coeff = coeff

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        agent_pos = self._get_agent_pos()
        target_pos = self._get_target_pos()
        
        if agent_pos is not None and target_pos is not None:
            dist = np.linalg.norm(np.array(agent_pos) - np.array(target_pos))
            dense_reward = -self.coeff * dist
        else:
            dense_reward = 0.0
        
        total_reward = reward + dense_reward
        info['dense_reward'] = dense_reward
        
        return obs, total_reward, terminated, truncated, info

    def _get_agent_pos(self):
        if hasattr(self.unwrapped, 'agent_pos'):
            return self.unwrapped.agent_pos
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
