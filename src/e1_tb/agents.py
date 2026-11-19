import random
from collections import defaultdict

N_ACTIONS = 4


def argmax(xs):
    """Deterministic first-index argmax — for evaluation."""
    return xs.index(max(xs))


def argmax_random(xs, rng):
    """Random among ties — for training. This is what breaks symmetry."""
    best = max(xs)
    ties = [i for i, x in enumerate(xs) if x == best]
    return rng.choice(ties) if len(ties) > 1 else ties[0]


class QLearning:
    def __init__(self, alpha=0.1, gamma=0.95, init_value=0.0, rng=None):
        self.alpha      = alpha
        self.gamma      = gamma
        self.init_value = init_value
        self.rng        = rng if rng is not None else random.Random()
        self.Q          = defaultdict(lambda: [init_value] * N_ACTIONS)

    def act(self, s, eps):
        if self.rng.random() < eps:
            return self.rng.randrange(N_ACTIONS)
        return argmax_random(self.Q[s], self.rng)

    def update(self, s, a, r, s2, done):
        target = r + (0.0 if done else self.gamma * max(self.Q[s2]))
        self.Q[s][a] += self.alpha * (target - self.Q[s][a])


class SARSA(QLearning):
    def update(self, s, a, r, s2, a2, done):
        target = r + (0.0 if done else self.gamma * self.Q[s2][a2])
        self.Q[s][a] += self.alpha * (target - self.Q[s][a])


class DynaQ(QLearning):
    def __init__(self, alpha=0.1, gamma=0.95, n_plan=10,
                 init_value=0.0, rng=None):
        super().__init__(alpha, gamma, init_value, rng)
        self.n_plan     = n_plan
        self.model      = {}
        self.model_keys = []

    def update(self, s, a, r, s2, done):
        super().update(s, a, r, s2, done)
        if (s, a) not in self.model:
            self.model[(s, a)] = (r, s2, done)
            self.model_keys.append((s, a))
        if not self.model_keys:
            return
        for _ in range(self.n_plan):
            sp, ap         = self.rng.choice(self.model_keys)
            rp, s2p, donep = self.model[(sp, ap)]
            target = rp + (0.0 if donep else self.gamma * max(self.Q[s2p]))
            self.Q[sp][ap] += self.alpha * (target - self.Q[sp][ap])