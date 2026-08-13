import gymnasium as gym


class NativeRewardInfoWrapper(gym.Wrapper):
    """Expose native transition metrics before reward shaping is applied."""

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["native_reward"] = float(reward)
        info["native_terminated"] = bool(terminated)
        info["native_truncated"] = bool(truncated)
        return observation, reward, terminated, truncated, info
