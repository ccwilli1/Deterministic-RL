import math
import random

# States: 3 x 4 grid
states = [(x, y) for y in range(4) for x in range(3)]

# Terminal states
terminal_states = {(1, 1), (1, 2), (2, 1), (2, 3)}

# Rewards for states (all others are 0)
rewards = {
    (1, 1): -10,
    (2, 1): -20,
    (1, 2): 10,
    (2, 3): 20,
    (0, 0): 0, (1, 0): 0, (2, 0): 0,
    (0, 1): 0,
    (0, 2): 0,
    (0, 3): 0, (1, 3): 0,
    (2, 2): 0,
}

actions = ["up", "down", "left", "right"]
action_delta = {
    "up":    (0, -1),
    "down":  (0, 1),
    "left":  (-1, 0),
    "right": (1, 0),
}


def transition(s, a):
    """Deterministic next state."""
    if s in terminal_states:
        return s

    x, y = s
    dx, dy = action_delta[a]
    ns = (x + dx, y + dy)
    if ns in rewards:
        return ns
    return s


def policy_iteration(gamma, theta=1e-6):
    """
    Run policy iteration.
    Returns (V, policy, iterations), where iterations is the number
    of policy improvement cycles until convergence.
    """
    # Random initial policy for non terminal states
    policy = {}
    for s in states:
        if s in terminal_states:
            policy[s] = None
        else:
            policy[s] = random.choice(actions)

    iterations = 0

    while True:
        iterations += 1

        # Policy evaluation
        V = {s: 0.0 for s in states}
        while True:
            delta = 0.0
            newV = V.copy()
            for s in states:
                if s in terminal_states:
                    newV[s] = rewards[s]
                    continue

                a = policy[s]
                ns = transition(s, a)
                r = rewards.get(ns, 0)
                newV[s] = r + gamma * V[ns]
                delta = max(delta, abs(newV[s] - V[s]))

            V = newV
            if delta < theta:
                break

        # Policy improvement
        policy_stable = True
        for s in states:
            if s in terminal_states:
                continue

            old_action = policy[s]
            best_a = None
            best_val = -math.inf
            for a in actions:
                ns = transition(s, a)
                r = rewards.get(ns, 0)
                val = r + gamma * V[ns]
                if val > best_val:
                    best_val = val
                    best_a = a
            policy[s] = best_a
            if best_a != old_action:
                policy_stable = False

        if policy_stable:
            return V, policy, iterations


if __name__ == "__main__":
    for gamma in [0.9, 0.5, 0.1]:
        V, policy, iters = policy_iteration(gamma)
        print(f"\n=== POLICY ITERATION, gamma = {gamma} ===")
        print(f"Iterations until convergence: {iters}")
        print("Values:")
        for s in sorted(states, key=lambda t: (t[1], t[0])):
            print(f"{s}: {V[s]:.2f}")
        print("\nPolicy:")
        for s in sorted(states, key=lambda t: (t[1], t[0])):
            print(f"{s}: {policy[s]}")
