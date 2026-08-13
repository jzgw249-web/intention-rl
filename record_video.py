import argparse
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import minigrid  # noqa: F401
import numpy as np
from gymnasium import spaces
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO

from wrappers import DirectionalImageObservation


class ChannelFirstObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        height, width, channels = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(channels, height, width), dtype=np.uint8
        )

    def observation(self, observation):
        return np.transpose(observation, (2, 0, 1))


def observation_wrapper_for_model(env, model):
    expected_shape = tuple(model.observation_space.shape)
    if expected_shape == (3, 7, 7):
        return ChannelFirstObservation(ImgObsWrapper(env))
    if expected_shape == (7, 7, 3):
        return ImgObsWrapper(env)
    if expected_shape == (151,):
        return DirectionalImageObservation(env)
    raise ValueError(f"Unsupported model observation shape: {expected_shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--env-id", default="MiniGrid-Empty-8x8-v0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    model = PPO.load(args.model, device="cpu")
    frames = []
    for episode in range(args.episodes):
        env = gym.make(args.env_id, render_mode="rgb_array")
        env = observation_wrapper_for_model(env, model)
        observation, _ = env.reset(seed=args.seed + episode)
        frames.append(env.render())
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=args.deterministic)
            observation, _, terminated, truncated, _ = env.step(int(action))
            frames.append(env.render())
        env.close()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output, frames, fps=args.fps, macro_block_size=1)
    print(f"Saved {len(frames)} frames to {output}")


if __name__ == "__main__":
    main()
