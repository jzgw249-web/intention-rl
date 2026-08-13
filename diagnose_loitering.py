"""Gate 1 diagnostic for the naive intention reward.

Train one fixed seed per condition, then evaluate both policies on the native
MiniGrid reward while recording success, episode length, action counts, and
trajectories.  This script deliberately does not use EvalCallback because the
repository's existing callback evaluates shaped rewards.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401 - registers MiniGrid environments
import numpy as np
import torch
from minigrid.core.actions import Actions
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from wrappers.intention_wrapper import IntentionRewardWrapper


ENV_ID = "MiniGrid-Empty-8x8-v0"
PPO_GAMMA = 0.99


class NativeRewardInfoWrapper(gym.Wrapper):
    """Expose native transition data before any reward shaping is applied."""

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["native_reward"] = float(reward)
        info["native_terminated"] = bool(terminated)
        info["native_truncated"] = bool(truncated)
        return observation, reward, terminated, truncated, info


class EpisodeRecorder(BaseCallback):
    """Collect per-episode native metrics from a single vectorized environment."""

    def __init__(self):
        super().__init__()
        self.rows: list[dict] = []
        self._native_return = 0.0
        self._length = 0

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        self._native_return += float(info["native_reward"])
        self._length += 1
        if bool(self.locals["dones"][0]):
            terminated = bool(info["native_terminated"])
            truncated = bool(info["native_truncated"])
            self.rows.append(
                {
                    "episode": len(self.rows) + 1,
                    "episode_length": self._length,
                    "terminated": terminated,
                    "truncated": truncated,
                    "success": terminated and not truncated,
                    "native_return": self._native_return,
                }
            )
            self._native_return = 0.0
            self._length = 0
        return True


def make_training_env(condition: str, seed: int):
    def init():
        env = gym.make(ENV_ID)
        env.reset(seed=seed)
        env = NativeRewardInfoWrapper(env)
        env = ImgObsWrapper(env)
        if condition == "intention":
            env = IntentionRewardWrapper(env, coeff=0.5)
        return env

    return init


def train(condition: str, seed: int, total_timesteps: int, output_dir: Path):
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = DummyVecEnv([make_training_env(condition, seed)])
    recorder = EpisodeRecorder()
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=PPO_GAMMA,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=0,
        seed=seed,
        device="cpu",
    )
    model.learn(total_timesteps=total_timesteps, callback=recorder)
    model.save(output_dir / f"{condition}_seed{seed}")
    env.close()
    return model, recorder.rows


def evaluate(model: PPO, condition: str, episodes: int, seed: int):
    rows = []
    action_counts: Counter[str] = Counter()
    trajectories = []
    for episode in range(episodes):
        env = gym.make(ENV_ID)
        env = NativeRewardInfoWrapper(env)
        env = ImgObsWrapper(env)
        observation, _ = env.reset(seed=seed + episode)
        done = False
        native_return = 0.0
        length = 0
        trajectory = []
        while not done:
            action, _ = model.predict(observation, deterministic=True)
            action_int = int(action)
            base = env.unwrapped
            trajectory.append(
                {
                    "step": length,
                    "position": [int(x) for x in base.agent_pos],
                    "direction": int(base.agent_dir),
                    "action": Actions(action_int).name,
                }
            )
            observation, _, terminated, truncated, info = env.step(action_int)
            action_counts[Actions(action_int).name] += 1
            native_return += float(info["native_reward"])
            length += 1
            done = terminated or truncated
        rows.append(
            {
                "condition": condition,
                "episode": episode + 1,
                "episode_length": length,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "success": bool(terminated and not truncated),
                "native_return": native_return,
            }
        )
        trajectories.append(trajectory)
        env.close()
    total_actions = sum(action_counts.values())
    distribution = {
        action: {"count": count, "fraction": count / total_actions}
        for action, count in sorted(action_counts.items())
    }
    return rows, distribution, trajectories


def summarize(rows: list[dict]):
    lengths = np.asarray([row["episode_length"] for row in rows], dtype=float)
    returns = np.asarray([row["native_return"] for row in rows], dtype=float)
    successes = np.asarray([row["success"] for row in rows], dtype=float)
    return {
        "episodes": len(rows),
        "success_rate": float(successes.mean()),
        "native_return_mean": float(returns.mean()),
        "native_return_std": float(returns.std()),
        "episode_length_mean": float(lengths.mean()),
        "episode_length_std": float(lengths.std()),
    }


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-timesteps", type=int, default=30_000)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, default=Path("gate1_diagnostics"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "environment": ENV_ID,
        "seed": args.seed,
        "total_timesteps": args.total_timesteps,
        "eval_episodes": args.eval_episodes,
        "conditions": {},
    }
    for condition in ("intention", "sparse"):
        model, training_rows = train(
            condition, args.seed, args.total_timesteps, args.output_dir
        )
        evaluation_rows, action_distribution, trajectories = evaluate(
            model, condition, args.eval_episodes, args.seed + 10_000
        )
        write_csv(args.output_dir / f"{condition}_training_episodes.csv", training_rows)
        write_csv(args.output_dir / f"{condition}_evaluation_episodes.csv", evaluation_rows)
        with (args.output_dir / f"{condition}_trajectories.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(trajectories, handle, indent=2)
        report["conditions"][condition] = {
            "training": summarize(training_rows),
            "evaluation": summarize(evaluation_rows),
            "action_distribution": action_distribution,
        }
        print(condition, json.dumps(report["conditions"][condition], indent=2))

    with (args.output_dir / "gate1_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)


if __name__ == "__main__":
    main()
