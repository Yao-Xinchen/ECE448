import heapq

def best_first_search(starting_state):
    visited_states = {starting_state: (None, 0)}
    frontier = []
    heapq.heappush(frontier, starting_state)
    while frontier:
        current = heapq.heappop(frontier)
        if current.is_goal():
            return backtrack(visited_states, current)
        for next in current.get_neighbors():
            # dist_from_start is calculated in get_neighbors()
            if next not in visited_states or next.dist_from_start < visited_states[next][1]:
                visited_states[next] = (current, next.dist_from_start)
                heapq.heappush(frontier, next)
    return []

def backtrack(visited_states, goal_state):
    path = []
    while goal_state is not None:
        path.insert(0, goal_state)
        goal_state = visited_states[goal_state][0]
    return path
