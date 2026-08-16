import gymnasium as gym
import minigrid  # noqa: F401

from bfs_oracle import ACTIONS, BFSOracle
from diagnose_loitering import NativeRewardInfoWrapper
from wrappers.potential_bfs_wrapper import PotentialBFSRewardWrapper
from wrappers.potential_distance_wrapper import PotentialDistanceRewardWrapper


ENV_ID = "MiniGrid-FourRooms-v0"


def wall_signature(env):
    grid = env.unwrapped.grid
    return tuple(
        (x, y)
        for x in range(grid.width)
        for y in range(grid.height)
        if grid.get(x, y) is not None and grid.get(x, y).type == "wall"
    )


def validate_oracle(seed):
    env = gym.make(ENV_ID)
    env.reset(seed=seed)
    oracle = BFSOracle.from_env(env)
    assert env.unwrapped.max_steps == 100
    assert oracle.max_distance > 0
    for direction in range(4):
        assert oracle.distance(oracle.goal_position, direction) == 0
    positions = {candidate[0] for candidate in oracle.distances}
    for state, distance in oracle.distances.items():
        position, _ = state
        assert 0 <= distance <= oracle.max_distance
        if position == oracle.goal_position:
            continue
        action = oracle.optimal_actions[state]
        assert action in ACTIONS
        successor = BFSOracle._successor(state, action, positions)
        assert oracle.distances[successor] == distance - 1
    env.close()
    return oracle.max_distance


def validate_potential_ranges(wrapper_class):
    observed = []
    for seed in range(5):
        env = wrapper_class(NativeRewardInfoWrapper(gym.make(ENV_ID)), 0.99, 1.0)
        env.reset(seed=seed)
        if isinstance(env, PotentialBFSRewardWrapper):
            for distance in env.oracle.distances.values():
                observed.append(-distance / env.oracle.max_distance)
        else:
            for position in env._reachable_positions():
                env.unwrapped.agent_pos = position
                observed.append(env.potential())
        env.close()
    assert min(observed) >= -1.0
    assert max(observed) == 0.0
    return min(observed), max(observed)


def validate_endings(wrapper_class):
    env = wrapper_class(NativeRewardInfoWrapper(gym.make(ENV_ID)), 0.99, 1.0)
    env.reset(seed=7)
    goal = BFSOracle._find_goal(env.unwrapped.grid)
    for direction, (dx, dy) in enumerate(((1, 0), (0, 1), (-1, 0), (0, -1))):
        candidate = (goal[0] - dx, goal[1] - dy)
        cell = env.unwrapped.grid.get(*candidate)
        if cell is None or cell.can_overlap():
            env.unwrapped.agent_pos = candidate
            env.unwrapped.agent_dir = direction
            break
    _, _, terminated, truncated, info = env.step(2)
    assert terminated and not truncated and info["potential_after"] == 0.0
    env.close()

    timeout = wrapper_class(
        NativeRewardInfoWrapper(gym.make(ENV_ID, max_steps=1)), 0.99, 1.0
    )
    timeout.reset(seed=7)
    _, _, terminated, truncated, info = timeout.step(0)
    assert truncated and not terminated
    assert info["potential_after"] == timeout.potential()
    timeout.close()


def main():
    layouts = []
    max_distances = []
    for seed in range(5):
        env = gym.make(ENV_ID)
        env.reset(seed=seed)
        signature = wall_signature(env)
        assert len(signature) == 101
        layouts.append(signature)
        env.close()
        max_distances.append(validate_oracle(seed))
    assert len(set(layouts)) == 5
    print("oracle max distances", max_distances)
    print("geo range", validate_potential_ranges(PotentialDistanceRewardWrapper))
    print("bfs range", validate_potential_ranges(PotentialBFSRewardWrapper))
    validate_endings(PotentialDistanceRewardWrapper)
    validate_endings(PotentialBFSRewardWrapper)
    print("all task05 validations passed")


if __name__ == "__main__":
    main()
