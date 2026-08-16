import argparse
import csv
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np
import torch
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from diagnose_loitering import NativeRewardInfoWrapper
from wrappers.intention_wrapper import IntentionRewardWrapper
from wrappers.potential_bfs_wrapper import PotentialBFSRewardWrapper
from wrappers.potential_distance_wrapper import PotentialDistanceRewardWrapper


PPO_GAMMA = 0.99
EVALUATION_POLICY_SEED = 20260817
DEFAULT_ENV_ID = "MiniGrid-FourRooms-v0"
CANONICAL_CONDITIONS = (
    "sparse", "potential_geo", "potential_bfs", "intention_naive",
)
CLI_CONDITIONS = CANONICAL_CONDITIONS + ("intention",)


def canonical_condition(condition):
    return "intention_naive" if condition == "intention" else condition


def apply_training_reward(env, condition, gamma, shaping_coeff):
    condition = canonical_condition(condition)
    if condition == "sparse":
        return env
    if condition == "potential_geo":
        return PotentialDistanceRewardWrapper(env, gamma, shaping_coeff)
    if condition == "potential_bfs":
        return PotentialBFSRewardWrapper(env, gamma, shaping_coeff)
    if condition == "intention_naive":
        return IntentionRewardWrapper(env, coeff=0.5)
    raise ValueError(condition)


def make_env(env_id, condition, seed, *, training, gamma=PPO_GAMMA,
             shaping_coeff=1.0, monitor_path=None):
    def init():
        env = gym.make(env_id)
        env = NativeRewardInfoWrapper(env)
        if training:
            env = apply_training_reward(env, condition, gamma, shaping_coeff)
        env = ImgObsWrapper(env)
        if monitor_path is not None:
            Path(monitor_path).parent.mkdir(parents=True, exist_ok=True)
            env = Monitor(env, filename=str(monitor_path))
        env.reset(seed=seed)
        return env
    return init


def evaluate_native(model, env_id, seeds, deterministic):
    metrics = {key: [] for key in ("success_rate", "native_return", "episode_length")}
    for seed in seeds:
        env = make_env(env_id, "sparse", seed, training=False)()
        observation, _ = env.reset(seed=seed)
        terminated = truncated = False
        native_return = 0.0
        length = 0
        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=deterministic)
            observation, reward, terminated, truncated, _ = env.step(int(action))
            native_return += float(reward)
            length += 1
        metrics["success_rate"].append(float(terminated and not truncated))
        metrics["native_return"].append(native_return)
        metrics["episode_length"].append(float(length))
        env.close()
    return {name: np.asarray(values) for name, values in metrics.items()}


class NativeEvaluationCallback(BaseCallback):
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
            with (self.output_dir / f"{metric}.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
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
            np.random.seed(EVALUATION_POLICY_SEED)
            torch.manual_seed(EVALUATION_POLICY_SEED)
            stochastic = evaluate_native(self.model, self.env_id, self.eval_seeds, False)
            deterministic = evaluate_native(self.model, self.env_id, self.eval_seeds, True)
        finally:
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(torch_state)
        for metric in self.METRICS:
            stochastic_values = stochastic[metric]
            deterministic_values = deterministic[metric]
            with (self.output_dir / f"{metric}.csv").open(
                "a", newline="", encoding="utf-8"
            ) as handle:
                csv.writer(handle).writerow([
                    timesteps,
                    stochastic_values.mean(), stochastic_values.std(),
                    deterministic_values.mean(), deterministic_values.std(),
                    len(stochastic_values),
                ])
        print(
            f"eval@{timesteps}: success={stochastic['success_rate'].mean():.3f}, "
            f"native_return={stochastic['native_return'].mean():.3f}, "
            f"episode_length={stochastic['episode_length'].mean():.1f}",
            flush=True,
        )


def build_model(env, seed):
    return PPO(
        "MlpPolicy", env, learning_rate=3e-4, n_steps=2048, batch_size=64,
        n_epochs=10, gamma=PPO_GAMMA, gae_lambda=0.95, clip_range=0.2,
        verbose=0, seed=seed, device="cpu",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wrapper", "--condition", dest="condition",
        choices=CLI_CONDITIONS, default="sparse",
    )
    parser.add_argument("--env-id", default=DEFAULT_ENV_ID)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--eval-seed-base", type=int, default=30_000)
    parser.add_argument(
        "--shaping-coeff", "--lambda", dest="shaping_coeff",
        type=float, default=1.0,
    )
    parser.add_argument("--log-dir", default="task05_calibration_logs")
    parser.add_argument("--save-dir", default="task05_calibration_models")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.debug:
        args.total_timesteps = 4_000

    condition = canonical_condition(args.condition)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env_slug = args.env_id.replace("MiniGrid-", "").replace("-v0", "").lower()
    experiment = f"{env_slug}_{condition}_seed{args.seed}"
    log_path = Path(args.log_dir) / experiment
    save_path = Path(args.save_dir) / experiment
    log_path.mkdir(parents=True, exist_ok=True)
    save_path.mkdir(parents=True, exist_ok=True)

    env = DummyVecEnv([make_env(
        args.env_id, condition, args.seed, training=True, gamma=PPO_GAMMA,
        shaping_coeff=args.shaping_coeff, monitor_path=log_path / "monitor.csv",
    )])
    model = build_model(env, args.seed)
    callback = NativeEvaluationCallback(
        args.env_id,
        range(args.eval_seed_base, args.eval_seed_base + args.eval_episodes),
        args.eval_freq,
        log_path / "evaluation",
    )
    try:
        model.learn(total_timesteps=args.total_timesteps, callback=callback)
        model.save(save_path / "final_model")
    finally:
        env.close()


if __name__ == "__main__":
    main()
