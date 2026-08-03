import gymnasium as gym
import numpy as np

class IntentionRewardWrapper(gym.Wrapper):
    def __init__(self, env, coeff=0.5):
        super().__init__(env)
        self.coeff = coeff

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        agent_pos = self._get_agent_pos()
        agent_dir = self._get_agent_dir()
        target_pos = self._get_target_pos()
        
        if agent_pos is not None and target_pos is not None and agent_dir is not None:
            dir_to_target = np.array(target_pos) - np.array(agent_pos)
            dist = np.linalg.norm(dir_to_target)
            if dist > 0:
                dir_to_target = dir_to_target / dist
            
            dir_vectors = [(1, 0), (0, 1), (-1, 0), (0, -1)]
            agent_dir_vec = np.array(dir_vectors[agent_dir])
            
            intention = np.dot(agent_dir_vec, dir_to_target)
            intention_reward = self.coeff * (intention + 1.0) / 2.0
        else:
            intention_reward = 0.0
        
        total_reward = reward + intention_reward
        info['intention_reward'] = intention_reward
        
        return obs, total_reward, terminated, truncated, info

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