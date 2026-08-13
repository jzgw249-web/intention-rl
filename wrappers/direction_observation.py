import gymnasium as gym
import numpy as np
from gymnasium import spaces


class DirectionalImageObservation(gym.ObservationWrapper):
    """Flatten MiniGrid's image and append agent direction as a one-hot vector."""

    def __init__(self, env):
        super().__init__(env)
        image_space = env.observation_space["image"]
        self._image_size = int(np.prod(image_space.shape))
        self.observation_space = spaces.Box(
            low=0.0,
            high=255.0,
            shape=(self._image_size + 4,),
            dtype=np.float32,
        )

    def observation(self, observation):
        image = np.asarray(observation["image"], dtype=np.float32).reshape(-1)
        direction = np.zeros(4, dtype=np.float32)
        direction[int(observation["direction"])] = 1.0
        return np.concatenate((image, direction), dtype=np.float32)
