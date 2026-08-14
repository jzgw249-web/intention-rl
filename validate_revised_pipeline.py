import gymnasium as gym
import minigrid  # noqa: F401

from diagnose_loitering import NativeRewardInfoWrapper
from wrappers.intention_pb_wrapper import PotentialIntentionRewardWrapper


def main():
    gamma = 0.99
    terminal = PotentialIntentionRewardWrapper(
        NativeRewardInfoWrapper(gym.make("MiniGrid-Empty-8x8-v0")), gamma, 1.0
    )
    terminal.reset(seed=0)
    terminal.unwrapped.agent_pos = (5, 6)
    terminal.unwrapped.agent_dir = 0
    _, _, terminated, truncated, info = terminal.step(2)
    assert terminated and not truncated and info["potential_after"] == 0.0
    terminal.close()

    timeout = PotentialIntentionRewardWrapper(
        NativeRewardInfoWrapper(gym.make("MiniGrid-Empty-8x8-v0", max_steps=1)), gamma, 1.0
    )
    timeout.reset(seed=0)
    _, _, terminated, truncated, info = timeout.step(0)
    assert truncated and not terminated
    assert info["potential_after"] == timeout.potential()
    timeout.close()
    print("terminal/truncation validation passed")


if __name__ == "__main__":
    main()
