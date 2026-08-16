from collections import defaultdict, deque
from dataclasses import dataclass


DIR_TO_VEC = ((1, 0), (0, 1), (-1, 0), (0, -1))
LEFT = 0
RIGHT = 1
FORWARD = 2
ACTIONS = (LEFT, RIGHT, FORWARD)


@dataclass(frozen=True)
class BFSOracle:
    """Shortest-path distances and one optimal action on a MiniGrid layout."""

    distances: dict
    optimal_actions: dict
    goal_position: tuple
    max_distance: int

    @classmethod
    def from_env(cls, env):
        grid = env.unwrapped.grid
        goal = cls._find_goal(grid)
        positions = {
            (x, y)
            for x in range(grid.width)
            for y in range(grid.height)
            if cls._passable(grid.get(x, y))
        }
        states = {(position, direction) for position in positions for direction in range(4)}
        reverse = defaultdict(list)
        successors = {}
        for state in states:
            for action in ACTIONS:
                successor = cls._successor(state, action, positions)
                successors[(state, action)] = successor
                reverse[successor].append(state)

        distances = {}
        frontier = deque()
        for direction in range(4):
            goal_state = (goal, direction)
            distances[goal_state] = 0
            frontier.append(goal_state)
        while frontier:
            state = frontier.popleft()
            next_distance = distances[state] + 1
            for predecessor in reverse[state]:
                if predecessor not in distances:
                    distances[predecessor] = next_distance
                    frontier.append(predecessor)

        finite_max = max(distances.values(), default=1)
        max_distance = max(finite_max, 1)
        complete_distances = {
            state: distances.get(state, max_distance)
            for state in states
        }
        optimal_actions = {}
        for state in states:
            if state[0] == goal:
                optimal_actions[state] = None
                continue
            optimal_actions[state] = min(
                ACTIONS,
                key=lambda action: (complete_distances[successors[(state, action)]], action),
            )
        return cls(complete_distances, optimal_actions, goal, max_distance)

    @staticmethod
    def _passable(cell):
        return cell is None or cell.can_overlap()

    @staticmethod
    def _find_goal(grid):
        for x in range(grid.width):
            for y in range(grid.height):
                cell = grid.get(x, y)
                if cell is not None and cell.type == "goal":
                    return (x, y)
        raise RuntimeError("Goal cell not found")

    @staticmethod
    def _successor(state, action, positions):
        position, direction = state
        if action == LEFT:
            return position, (direction - 1) % 4
        if action == RIGHT:
            return position, (direction + 1) % 4
        dx, dy = DIR_TO_VEC[direction]
        forward = (position[0] + dx, position[1] + dy)
        return (forward, direction) if forward in positions else state

    def distance(self, position, direction):
        return self.distances[(tuple(position), int(direction))]

    def optimal_action(self, position, direction):
        return self.optimal_actions[(tuple(position), int(direction))]
