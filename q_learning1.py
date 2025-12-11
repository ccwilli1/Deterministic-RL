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
    """Deterministic next state for Q learning."""
    if s in terminal_states:
        return s
    x, y = s
    dx, dy = action_delta[a]
    ns = (x + dx, y + dy)
    if ns in rewards:
        return ns
    return s


def q_learning(gamma, alpha=0.1, epsilon=0.1, episodes=5000, max_steps=50):
    """
    Simple tabular Q learning.
    Returns (Q, policy) after training.
    """
    Q = {(s, a): 0.0 for s in states for a in actions}

    for _ in range(episodes):
        # start from a random non terminal state
        s = random.choice([st for st in states if st not in terminal_states])

        for _ in range(max_steps):
            if s in terminal_states:
                break

            # epsilon greedy action selection
            if random.random() < epsilon:
                a = random.choice(actions)
            else:
                a = max(actions, key=lambda act: Q[(s, act)])

            ns = transition(s, a)
            r = rewards.get(ns, 0)

            if ns in terminal_states:
                max_next = 0.0
            else:
                max_next = max(Q[(ns, act)] for act in actions)

            Q[(s, a)] += alpha * (r + gamma * max_next - Q[(s, a)])
            s = ns

    # derive greedy policy from Q
    policy = {}
    for s in states:
        if s in terminal_states:
            policy[s] = None
        else:
            policy[s] = max(actions, key=lambda a: Q[(s, a)])

    return Q, policy


if __name__ == "__main__":
    for gamma in [0.9, 0.5, 0.1]:
        Q, policy = q_learning(gamma)
        print(f"\n=== Q LEARNING, gamma = {gamma} ===")
        print("Policy:")
        for s in sorted(states, key=lambda t: (t[1], t[0])):
            print(f"{s}: {policy[s]}")

        print("\nFinal Q values:")
        for s in sorted(states, key=lambda t: (t[1], t[0])):
            if s in terminal_states:
                continue
            print(f"State {s}:")
            for a in actions:
                print(f"  {a}: {Q[(s, a)]:.2f}")
