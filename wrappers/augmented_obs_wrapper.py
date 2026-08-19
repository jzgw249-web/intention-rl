"""Augmented observation: expose the potential function's arguments to the policy.

Task-05 found that every shaping condition underperformed ``sparse`` on FourRooms,
including a potential built from exact shortest-path distances.  The proposed
explanation is partial observability: the potential is a function of absolute
position, goal position and wall layout, while the memoryless policy only sees a
7x7 egocentric patch.  This wrapper makes the first two of those arguments
observable so the explanation can be tested directly.

Representation note (important for comparability with task-05):
Stable-Baselines3 classifies a ``(7, 7, 3) uint8`` Box as an image space and
divides it by 255 during preprocessing.  This wrapper therefore also scales the
image block to ``[0, 1]``, so the first 147 policy inputs are numerically
identical to what the task-05 runs fed to the network.  The only difference is
the four appended coordinates.
"""

import gymnasium as gym
import numpy as np


class AugmentedObsWrapper(gym.ObservationWrapper):
    """Flatten the egocentric image to ``[0, 1]`` and append normalised coordinates.

    Appended features (4 dims, all in ``[0, 1]``):
        agent_x / (width - 1), agent_y / (height - 1),
        goal_x  / (width - 1), goal_y  / (height - 1)
    """

    def __init__(self, env):
        super().__init__(env)
        image_space = env.observation_space
        if image_space.dtype != np.uint8 or len(image_space.shape) != 3:
            raise ValueError(
                "AugmentedObsWrapper expects an ImgObsWrapper-style uint8 image "
                f"observation, got {image_space}"
            )
        self._image_size = int(np.prod(image_space.shape))
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(self._image_size + 4,), dtype=np.float32
        )
        self._goal_position_cache = None

    # -- goal lookup is cached per episode: the layout only changes on reset ----
    def reset(self, **kwargs):
        self._goal_position_cache = None
        return super().reset(**kwargs)

    def _goal_position(self):
        if self._goal_position_cache is None:
            grid = self.unwrapped.grid
            for x in range(grid.width):
                for y in range(grid.height):
                    cell = grid.get(x, y)
                    if cell is not None and cell.type == "goal":
                        self._goal_position_cache = (float(x), float(y))
                        break
                if self._goal_position_cache is not None:
                    break
            if self._goal_position_cache is None:
                raise RuntimeError("Goal cell not found")
        return self._goal_position_cache

    def observation(self, observation):
        image = np.asarray(observation, dtype=np.float32).reshape(-1) / 255.0
        base = self.unwrapped
        width = float(base.grid.width - 1)
        height = float(base.grid.height - 1)
        agent_x, agent_y = (float(v) for v in base.agent_pos)
        goal_x, goal_y = self._goal_position()
        extra = np.array(
            [agent_x / width, agent_y / height, goal_x / width, goal_y / height],
            dtype=np.float32,
        )
        return np.concatenate([image, extra]).astype(np.float32)
