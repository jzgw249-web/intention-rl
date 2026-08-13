import math

import gymnasium as gym
import minigrid  # noqa: F401

from wrappers.native_metrics import NativeRewardInfoWrapper
from wrappers.potential_wrappers import (
    GeometricPotentialWrapper,
    OptimalPathPotentialWrapper,
    WrongDirectionPotentialWrapper,
)

GAMMA = 0.99
WRAPPERS = {
    "geo": GeometricPotentialWrapper,
    "optimal": OptimalPathPotentialWrapper,
    "wrong": WrongDirectionPotentialWrapper,
}


def check_ranges():
    ranges = {name: [] for name in WRAPPERS}
    for seed in range(20):
        for name, wrapper_class in WRAPPERS.items():
            env = wrapper_class(
                NativeRewardInfoWrapper(gym.make("MiniGrid-FourRooms-v0")),
                gamma=GAMMA,
                shaping_coeff=0.2,
            )
            env.reset(seed=seed)
            if name == "optimal":
                values = [-value / env.normalizer for value in env.distances.values()]
            else:
                sign = 1.0 if name == "wrong" else -1.0
                values = []
                base = env.unwrapped
                for x in range(base.grid.width):
                    for y in range(base.grid.height):
                        cell = base.grid.get(x, y)
                        if cell is None or cell.can_overlap():
                            values.append(
                                sign * math.dist((x, y), env.goal_position) / env.normalizer
                            )
            ranges[name].extend(values)
            env.close()
    assert min(ranges["geo"]) >= -1.0 and max(ranges["geo"]) <= 0.0
    assert min(ranges["optimal"]) >= -1.0 and max(ranges["optimal"]) <= 0.0
    assert min(ranges["wrong"]) >= 0.0 and max(ranges["wrong"]) <= 1.0
    return {name: [min(values), max(values)] for name, values in ranges.items()}


def check_boundaries():
    results = {}
    for name, wrapper_class in WRAPPERS.items():
        terminal_env = wrapper_class(
            NativeRewardInfoWrapper(gym.make("MiniGrid-Empty-8x8-v0")), GAMMA, 0.2
        )
        terminal_env.reset(seed=0)
        terminal_env.unwrapped.agent_pos = (5, 6)
        terminal_env.unwrapped.agent_dir = 0
        _, _, terminated, truncated, terminal_info = terminal_env.step(2)
        assert terminated and not truncated
        assert terminal_info["potential_after"] == 0.0
        terminal_env.close()

        truncation_env = wrapper_class(
            NativeRewardInfoWrapper(
                gym.make("MiniGrid-Empty-8x8-v0", max_steps=1)
            ),
            GAMMA,
            0.2,
        )
        truncation_env.reset(seed=0)
        _, _, terminated, truncated, truncation_info = truncation_env.step(0)
        assert truncated and not terminated
        expected = truncation_env.potential()
        assert truncation_info["potential_after"] == expected
        truncation_env.close()
        results[name] = {
            "terminated_phi_after": terminal_info["potential_after"],
            "truncated_phi_after": truncation_info["potential_after"],
        }
    return results


if __name__ == "__main__":
    print("ranges", check_ranges())
    print("boundaries", check_boundaries())
