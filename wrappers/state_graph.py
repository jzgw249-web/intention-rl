from collections import defaultdict, deque


DIR_TO_VEC = ((1, 0), (0, 1), (-1, 0), (0, -1))
ACTIONS = (0, 1, 2)  # left, right, forward


def traversable_positions(env):
    grid = env.unwrapped.grid
    positions = []
    for x in range(grid.width):
        for y in range(grid.height):
            cell = grid.get(x, y)
            if cell is None or cell.can_overlap():
                positions.append((x, y))
    return positions


def transition(state, action, traversable):
    (x, y), direction = state
    if action == 0:
        return ((x, y), (direction - 1) % 4)
    if action == 1:
        return ((x, y), (direction + 1) % 4)
    dx, dy = DIR_TO_VEC[direction]
    next_position = (x + dx, y + dy)
    if next_position not in traversable:
        next_position = (x, y)
    return (next_position, direction)


def shortest_path_table(env, goal_position):
    """Return finite distances and optimal actions for (position, direction)."""

    positions = traversable_positions(env)
    traversable = set(positions)
    states = [(position, direction) for position in positions for direction in range(4)]
    predecessors = defaultdict(list)
    for state in states:
        for action in ACTIONS:
            successors = transition(state, action, traversable)
            predecessors[successors].append((state, action))

    distances = {}
    queue = deque()
    for direction in range(4):
        goal_state = (tuple(goal_position), direction)
        distances[goal_state] = 0
        queue.append(goal_state)

    while queue:
        state = queue.popleft()
        for predecessor, _ in predecessors[state]:
            if predecessor not in distances:
                distances[predecessor] = distances[state] + 1
                queue.append(predecessor)

    finite_max = max(distances.values(), default=1)
    for state in states:
        distances.setdefault(state, finite_max)

    optimal_actions = {}
    for state in states:
        if state[0] == tuple(goal_position):
            optimal_actions[state] = ()
            continue
        best = min(distances[transition(state, action, traversable)] for action in ACTIONS)
        optimal_actions[state] = tuple(
            action
            for action in ACTIONS
            if distances[transition(state, action, traversable)] == best
        )
    return distances, optimal_actions, max(distances.values(), default=1)
