import math

import gymnasium as gym
import minigrid  # noqa: F401

from diagnose_loitering import NativeRewardInfoWrapper
from wrappers.intention_pb_wrapper import PotentialIntentionRewardWrapper
from wrappers.potential_distance_wrapper import PotentialDistanceRewardWrapper


def test_distance_ranges():
    observed = []
    for seed in range(30):
        env = PotentialDistanceRewardWrapper(
            NativeRewardInfoWrapper(gym.make("MiniGrid-Empty-Random-6x6-v0")),
            gamma=0.99,
            shaping_coeff=1.0,
        )
        env.reset(seed=seed)
        assert env.unwrapped.max_steps == 144
        for position in env._reachable_positions():
            observed.append(-math.dist(position, env.goal_position) / env.max_distance)
        env.close()
    assert min(observed) >= -1.0 and max(observed) <= 0.0
    return min(observed), max(observed)


def test_intention_ranges():
    ranges = {}
    for form in ("raw", "shifted"):
        values = []
        env = PotentialIntentionRewardWrapper(
            NativeRewardInfoWrapper(gym.make("MiniGrid-Empty-Random-6x6-v0")),
            gamma=0.99,
            shaping_coeff=1.0,
            potential_form=form,
        )
        env.reset(seed=42)
        base = env.unwrapped
        for direction in range(4):
            base.agent_dir = direction
            values.append(env.potential())
        env.close()
        ranges[form] = (min(values), max(values))
    assert ranges["raw"][0] >= 0 and ranges["raw"][1] <= 1
    assert ranges["shifted"][0] >= -1 and ranges["shifted"][1] <= 0
    return ranges


def test_endings(wrapper_factory):
    terminal = wrapper_factory(gym.make("MiniGrid-Empty-8x8-v0"))
    terminal.reset(seed=0)
    terminal.unwrapped.agent_pos = (5, 6)
    terminal.unwrapped.agent_dir = 0
    _, _, terminated, truncated, info = terminal.step(2)
    assert terminated and not truncated and info["potential_after"] == 0.0
    terminal.close()

    timeout = wrapper_factory(gym.make("MiniGrid-Empty-8x8-v0", max_steps=1))
    timeout.reset(seed=0)
    _, _, terminated, truncated, info = timeout.step(0)
    assert truncated and not terminated
    assert info["potential_after"] == timeout.potential()
    timeout.close()


def main():
    print("distance range", test_distance_ranges())
    print("intention ranges", test_intention_ranges())
    test_endings(lambda env: PotentialDistanceRewardWrapper(NativeRewardInfoWrapper(env), 0.99, 1.0))
    for form in ("raw", "shifted"):
        test_endings(lambda env, form=form: PotentialIntentionRewardWrapper(
            NativeRewardInfoWrapper(env), 0.99, 1.0, form
        ))
    print("all task04 validations passed")


if __name__ == "__main__":
    main()
