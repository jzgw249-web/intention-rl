"""Reproduce observation and random-policy statistics without training a model."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401 - register MiniGrid environments
import numpy as np
from minigrid.core.constants import OBJECT_TO_IDX

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wrappers.potential_distance_wrapper import PotentialDistanceRewardWrapper


ACTION_RNG_SEED = 50_000
TOLERANCE = 0.005
OUT = Path(__file__).with_name("observability_stats.csv")

ENVIRONMENTS = (
    ("Empty-Random-6x6", "MiniGrid-Empty-Random-6x6-v0", 200),
    ("Empty-8x8", "MiniGrid-Empty-8x8-v0", 200),
    ("FourRooms", "MiniGrid-FourRooms-v0", 1000),
)

PUBLISHED = {
    ("Empty-Random-6x6", "goal_visibility"): 0.562,
    ("Empty-8x8", "goal_visibility"): 0.270,
    ("FourRooms", "goal_visibility"): 0.094,
    ("Empty-Random-6x6", "initial_phi_variance"): 0.046,
    ("Empty-8x8", "initial_phi_variance"): 0.000,
    ("FourRooms", "initial_phi_variance"): 0.053,
    ("Empty-Random-6x6", "random_success_rate"): 0.670,
    ("FourRooms", "random_success_rate"): 0.093,
}


def goal_visible(observation) -> bool:
    return bool((observation["image"][:, :, 0] == OBJECT_TO_IDX["goal"]).any())


def measure_environment(env_name: str, env_id: str, n_episodes: int, policy: str):
    action_rng = np.random.default_rng(ACTION_RNG_SEED)
    visible = observations = successes = 0
    initial_phi = []

    env = gym.make(env_id)
    phi_env = PotentialDistanceRewardWrapper(
        gym.make(env_id), gamma=0.99, shaping_coeff=1.0
    )
    try:
        for seed in range(50_000, 50_000 + n_episodes):
            observation, _ = env.reset(seed=seed)
            if policy == "all7":
                # Gymnasium action spaces own their RNG. Seed it from an independent
                # Generator so action sampling never touches NumPy's global RNG.
                env.action_space.seed(
                    int(action_rng.integers(0, 2**32, dtype=np.uint32))
                )
            visible += int(goal_visible(observation))
            observations += 1

            phi_env.reset(seed=seed)
            initial_phi.append(phi_env.potential())

            terminated = truncated = False
            while not (terminated or truncated):
                action = (int(action_rng.integers(0, 3)) if policy == "nav3"
                          else env.action_space.sample())
                observation, _, terminated, truncated, _ = env.step(action)
                visible += int(goal_visible(observation))
                observations += 1
            successes += int(terminated and not truncated)
    finally:
        env.close()
        phi_env.close()

    return {
        "goal_visibility": (visible / observations, observations),
        "initial_phi_variance": (float(np.var(initial_phi, ddof=0)), 0),
        "random_success_rate": (successes / n_episodes, 0),
    }


def main():
    rows = []
    for env_name, env_id, n_episodes in ENVIRONMENTS:
        for policy in ("nav3", "all7"):
            measurements = measure_environment(env_name, env_id, n_episodes, policy)
            for metric, (value, n_observations) in measurements.items():
                published = PUBLISHED.get((env_name, metric)) if policy == "nav3" else None
                difference = abs(value - published) if published is not None else None
                agrees = difference <= TOLERANCE if difference is not None else ""
                denominator = (n_observations if metric == "goal_visibility"
                               else n_episodes)
                se = (float(np.sqrt(value * (1 - value) / denominator))
                      if metric != "initial_phi_variance" else "")
                rows.append({
                    "env": env_name,
                    "policy": policy,
                    "metric": metric,
                    "value": value,
                    "se": se,
                    "n_episodes": n_episodes,
                    "n_observations": n_observations,
                    "published_value": "" if published is None else published,
                    "abs_diff": "" if difference is None else difference,
                    "agrees": agrees,
                })

    empty_variance = next(
        r["value"] for r in rows if r["env"] == "Empty-8x8"
        and r["policy"] == "nav3" and r["metric"] == "initial_phi_variance"
    )
    if empty_variance != 0.0:
        raise AssertionError(f"Empty-8x8 initial potential variance is {empty_variance}, not 0")
    for env_name, _, _ in ENVIRONMENTS:
        phi = [r["value"] for r in rows if r["env"] == env_name
               and r["metric"] == "initial_phi_variance"]
        if len(phi) != 2 or phi[0] != phi[1]:
            raise AssertionError(f"potential variance differs by policy for {env_name}: {phi}")
    for policy in ("nav3", "all7"):
        visibility = {r["env"]: r["value"] for r in rows
                      if r["policy"] == policy and r["metric"] == "goal_visibility"}
        if not (visibility["Empty-Random-6x6"] > visibility["Empty-8x8"]
                > visibility["FourRooms"]):
            raise AssertionError(
                f"unexpected visibility ordering for {policy}: {visibility}"
            )
    for env_name in ("Empty-Random-6x6", "Empty-8x8"):
        success = {r["policy"]: r["value"] for r in rows
                   if r["env"] == env_name and r["metric"] == "random_success_rate"}
        if not success["nav3"] > success["all7"]:
            raise AssertionError(f"nav3 success is not higher for {env_name}: {success}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "env", "policy", "metric", "value", "se", "n_episodes",
            "n_observations", "published_value", "abs_diff", "agrees",
        ))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
