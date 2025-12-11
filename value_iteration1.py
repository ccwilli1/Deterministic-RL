states = [(x, y) for y in range(4) for x in range(3)]
terminal_states = {(1, 1), (1, 2), (2, 1), (2, 3)}
rewards = {
    (1, 1): -10, (2, 1): -20, (1, 2): 10, (2, 3): 20,
    (0, 0): 0, (1, 0): 0, (2, 0): 0, (0, 1): 0,
    (0, 2): 0, (0, 3): 0, (1, 3): 0, (2, 2): 0
}

actions = {
    "up":    (0, -1),
    "down":  (0, 1),
    "left":  (-1, 0),
    "right": (1, 0)
}

def transition(state, action):
    if state in terminal_states:
        return state
    x, y = state
    dx, dy = actions[action]
    nx, ny = x + dx, y + dy
    # Check if new state is within grid bounds
    if 0 <= nx < 3 and 0 <= ny < 4:
        return (nx, ny)
    return state

def value_iteration(gamma):
    V = {s: 0 for s in states}
    while True:
        delta = 0
        newV = V.copy()
        for s in states:
            if s in terminal_states:
                newV[s] = rewards[s]
                continue
            vals = []
            for a in actions:
                ns = transition(s, a)
                vals.append(rewards.get(ns, 0) + gamma * V[ns])
            newV[s] = max(vals)
            delta = max(delta, abs(newV[s] - V[s]))
        V = newV
        if delta < 1e-6:
            break

    policy = {}
    for s in states:
        if s in terminal_states:
            policy[s] = None
            continue
        best = None
        best_val = -1e9
        for a in actions:
            ns = transition(s, a)
            v = rewards.get(ns, 0) + gamma * V[ns]
            if v > best_val:
                best_val = v
                best = a
        policy[s] = best

    return V, policy

if __name__ == "__main__":
    for gamma in [0.9, 0.5, 0.1]:
        V, policy = value_iteration(gamma)
        print(f"\n=== VALUE ITERATION γ={gamma} ===")
        print("Values:")
        for s in states:
            print(s, ":", round(V[s], 2))
        print("Policy:")
        print(policy)
