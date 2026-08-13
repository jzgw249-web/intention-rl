import argparse
import csv
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from wrappers import (
    DirectionalImageObservation,
    GeometricPotentialWrapper,
    IntentionRewardWrapper,
    NativeRewardInfoWrapper,
    OptimalPathPotentialWrapper,
    WrongDirectionPotentialWrapper,
)

PPO_GAMMA = 0.99
DEFAULT_ENV_ID = "MiniGrid-FourRooms-v0"
CONDITIONS = ("sparse", "geo", "optimal", "wrong", "intention")


def apply_reward_wrapper(env, condition, gamma, shaping_coeff):
    if condition == "sparse":
        return env
    if condition == "geo":
        return GeometricPotentialWrapper(env, gamma, shaping_coeff)
    if condition == "optimal":
        return OptimalPathPotentialWrapper(env, gamma, shaping_coeff)
    if condition == "wrong":
        return WrongDirectionPotentialWrapper(env, gamma, shaping_coeff)
    if condition == "intention":
        return IntentionRewardWrapper(env, coeff=0.5)
    raise ValueError(f"Unknown condition: {condition}")


def make_env(env_id, condition, seed, *, training, gamma=PPO_GAMMA,
             shaping_coeff=0.2, monitor_path=None):
    """Evaluation gets the observation transform but never reward shaping."""
    def init():
        env = gym.make(env_id)
        env = NativeRewardInfoWrapper(env)
        if training:
            env = apply_reward_wrapper(env, condition, gamma, shaping_coeff)
        env = DirectionalImageObservation(env)
        if monitor_path is not None:
            Path(monitor_path).parent.mkdir(parents=True, exist_ok=True)
            env = Monitor(env, filename=str(monitor_path))
        env.reset(seed=seed)
        return env
    return init


def evaluate_policy_native(model, env_id, seeds, deterministic):
    values = {name: [] for name in ("success_rate", "native_return", "episode_length")}
    for seed in seeds:
        env = make_env(env_id, "sparse", seed, training=False)()
        observation, _ = env.reset(seed=seed)
        terminated = truncated = False
        native_return = 0.0
        episode_length = 0
        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=deterministic)
            observation, reward, terminated, truncated, _ = env.step(int(action))
            native_return += float(reward)
            episode_length += 1
        values["success_rate"].append(float(terminated and not truncated))
        values["native_return"].append(native_return)
        values["episode_length"].append(float(episode_length))
        env.close()
    return {name: np.asarray(items) for name, items in values.items()}


class NativeEvaluationCallback(BaseCallback):
    """Write native metrics to three independent CSV files."""
    METRICS = ("success_rate", "native_return", "episode_length")

    def __init__(self, env_id, eval_seeds, eval_freq, output_dir):
        super().__init__()
        self.env_id = env_id
        self.eval_seeds = tuple(eval_seeds)
        self.eval_freq = int(eval_freq)
        self.output_dir = Path(output_dir)
        self.next_evaluation = self.eval_freq

    def _on_training_start(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        header = ["timesteps", "stochastic_mean", "stochastic_std",
                  "deterministic_mean", "deterministic_std", "episodes"]
        for metric in self.METRICS:
            with (self.output_dir / f"{metric}.csv").open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow(header)
        self._evaluate(0)

    def _on_step(self):
        if self.num_timesteps >= self.next_evaluation:
            self._evaluate(self.num_timesteps)
            while self.next_evaluation <= self.num_timesteps:
                self.next_evaluation += self.eval_freq
        return True

    def _on_training_end(self):
        if self.num_timesteps % self.eval_freq:
            self._evaluate(self.num_timesteps)

    def _evaluate(self, timesteps):
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        try:
            np.random.seed(20260814)
            torch.manual_seed(20260814)
            stochastic = evaluate_policy_native(self.model, self.env_id, self.eval_seeds, False)
            deterministic = evaluate_policy_native(self.model, self.env_id, self.eval_seeds, True)
        finally:
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(torch_state)
        for metric in self.METRICS:
            sv, dv = stochastic[metric], deterministic[metric]
            with (self.output_dir / f"{metric}.csv").open("a", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow([timesteps, sv.mean(), sv.std(), dv.mean(), dv.std(), len(sv)])
        print(f"eval@{timesteps}: success={stochastic['success_rate'].mean():.3f}, "
              f"native_return={stochastic['native_return'].mean():.3f}, "
              f"episode_length={stochastic['episode_length'].mean():.1f}")


def validate_observation_env(env_id=DEFAULT_ENV_ID):
    env = make_env(env_id, "sparse", 0, training=False)()
    check_env(env, warn=True)
    observation, _ = env.reset(seed=0)
    if observation.shape != (151,):
        raise AssertionError(f"Expected (151,), got {observation.shape}")
    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", "--wrapper", dest="condition", choices=CONDITIONS, default="sparse")
    parser.add_argument("--env-id", default=DEFAULT_ENV_ID)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--eval-freq", type=int)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--eval-seed-base", type=int, default=10_000)
    parser.add_argument("--shaping-coeff", type=float, default=0.2)
    parser.add_argument("--log-dir", default="./logs")
    parser.add_argument("--save-dir", default="./models")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.debug:
        args.total_timesteps = 5_000
    eval_freq = args.eval_freq or max(1_000, args.total_timesteps // 20)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env_slug = args.env_id.replace("MiniGrid-", "").replace("-v0", "").lower()
    experiment = f"{env_slug}_{args.condition}_seed{args.seed}"
    log_path = Path(args.log_dir) / experiment
    save_path = Path(args.save_dir) / experiment
    log_path.mkdir(parents=True, exist_ok=True)
    save_path.mkdir(parents=True, exist_ok=True)
    validate_observation_env(args.env_id)
    env = DummyVecEnv([make_env(args.env_id, args.condition, args.seed, training=True,
                                gamma=PPO_GAMMA, shaping_coeff=args.shaping_coeff,
                                monitor_path=log_path / "monitor.csv")])
    model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=2048, batch_size=64,
                n_epochs=10, gamma=PPO_GAMMA, gae_lambda=0.95, clip_range=0.2,
                verbose=0, seed=args.seed, device="cpu")
    callback = NativeEvaluationCallback(
        args.env_id,
        range(args.eval_seed_base, args.eval_seed_base + args.eval_episodes),
        eval_freq,
        log_path / "evaluation",
    )
    try:
        model.learn(total_timesteps=args.total_timesteps, callback=callback)
        model.save(save_path / "final_model")
    finally:
        env.close()


if __name__ == "__main__":
    main()
