"""Grid environment. Fire/goal distribution is biased toward a hotspot."""
import math
import random

GRID        = 7
STORE       = (GRID // 2, GRID // 2)
VIS_RADIUS  = 2
MAX_STEPS   = 60

ACTIONS       = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
ACTION_ARROWS = ["^", "v", "<", ">"]


def _visible(pos, goal, radius):
    return abs(goal[0] - pos[0]) + abs(goal[1] - pos[1]) <= radius


def _make_state(pos, goal, radius, last_dir, state_mode):
    if state_mode == "pos_goal_dir":
        if _visible(pos, goal, radius):
            return (pos, goal[0] - pos[0], goal[1] - pos[1], last_dir)
        return (pos, None, None, last_dir)
    if _visible(pos, goal, radius):
        return (pos, goal[0] - pos[0], goal[1] - pos[1])
    return (pos, None, None)


def _build_goal_table(grid, store, mu, sigma, hotspot_prob):
    """Mixture: uniform background + truncated 2D Gaussians at each hotspot.

    `mu` may be a single (r, c) tuple or a list of tuples.
    Each hotspot receives equal weight within `hotspot_prob`.
    """
    mus = [mu] if isinstance(mu[0], (int, float)) else list(mu)

    cells = [(r, c) for r in range(grid) for c in range(grid) if (r, c) != store]
    n = len(cells)
    uniform = 1.0 / n
    raw = []
    for (r, c) in cells:
        g = 0.0
        for (mr, mc) in mus:
            dr, dc = r - mr, c - mc
            g += math.exp(-(dr * dr + dc * dc) / (2.0 * sigma * sigma))
        g /= len(mus)
        raw.append((1.0 - hotspot_prob) * uniform + hotspot_prob * g)
    total = sum(raw)
    return cells, [w / total for w in raw]


class DeliveryEnv:
    def __init__(self, seed=0, grid=GRID, store=STORE, vis_radius=VIS_RADIUS,
                max_steps=MAX_STEPS, wall_penalty=2.0,
                state_mode="pos_goal",
                goal_mu=None, goal_sigma=1.5, goal_hotspot_prob=0.7,
                corners=None):
        self.rng          = random.Random(seed)
        self.grid         = grid
        self.store        = store
        self.vis_radius   = vis_radius
        self.max_steps    = max_steps
        self.wall_penalty = wall_penalty
        self.state_mode   = state_mode

        self.goal_mu           = goal_mu if goal_mu is not None else (grid - 1, grid - 1)
        self.goal_sigma        = goal_sigma
        self.goal_hotspot_prob = goal_hotspot_prob
        self.corners = corners or []
        self._goal_cells, self._goal_weights = _build_goal_table(
            grid, store, self.goal_mu, self.goal_sigma, self.goal_hotspot_prob)

    def reset(self, goal=None, seed=None):
        if seed is not None:
            self.rng = random.Random(seed)
        self.t        = 0
        self.pos      = self.store
        self.goal     = goal if goal is not None else self._spawn_goal()
        self.last_dir = 0
        self.searched = [False] * len(self.corners)
        self.done = False
        return self._state()

    def _spawn_goal(self):
        return self.rng.choices(self._goal_cells, weights=self._goal_weights, k=1)[0]

    def _state(self):
        base = _make_state(self.pos, self.goal, self.vis_radius,
                           self.last_dir, self.state_mode)
        if self.corners:
            return base + (tuple(self.searched),)
        return base

    def step(self, a):
        self.t += 1
        self.last_dir = a
        dr, dc = ACTIONS[a]
        nr, nc = self.pos[0] + dr, self.pos[1] + dc
        bump = not (0 <= nr < self.grid and 0 <= nc < self.grid)
        if not bump:
            self.pos = (nr, nc)

        r = -1.0
        if bump:
            r -= self.wall_penalty
        for i, corner in enumerate(self.corners):
            if self.pos == corner and not self.searched[i]:
                self.searched[i] = True
        if self.pos == self.goal:
            r += 10.0
            self.done = True
        elif self.t >= self.max_steps:
            self.done = True
        return self._state(), r, self.done


class MultiDeliveryEnv:
    def __init__(self, n_agents=2, seed=0, grid=GRID, store=STORE,
                 vis_radius=VIS_RADIUS, max_steps=MAX_STEPS,
                 wall_penalty=2.0, state_mode="pos_goal",
                 goal_mu=None, goal_sigma=1.5, goal_hotspot_prob=0.7):
        self.rng          = random.Random(seed)
        self.n_agents     = n_agents
        self.grid         = grid
        self.store        = store
        self.vis_radius   = vis_radius
        self.max_steps    = max_steps
        self.wall_penalty = wall_penalty
        self.state_mode   = state_mode

        self.goal_mu           = goal_mu if goal_mu is not None else (grid - 1, grid - 1)
        self.goal_sigma        = goal_sigma
        self.goal_hotspot_prob = goal_hotspot_prob
        self._goal_cells, self._goal_weights = _build_goal_table(
            grid, store, self.goal_mu, self.goal_sigma, self.goal_hotspot_prob)

    def reset(self, goal=None, seed=None):
        if seed is not None:
            self.rng = random.Random(seed)
        self.t         = 0
        self.positions = [self.store] * self.n_agents
        self.last_dirs = [0] * self.n_agents
        self.goal      = goal if goal is not None else self._spawn_goal()
        self.done      = False
        return [self._state(i) for i in range(self.n_agents)]

    def _spawn_goal(self):
        return self.rng.choices(self._goal_cells, weights=self._goal_weights, k=1)[0]

    def _state(self, i):
        return _make_state(self.positions[i], self.goal, self.vis_radius,
                           self.last_dirs[i], self.state_mode)

    def step(self, actions):
        self.t += 1
        rewards = [-1.0] * self.n_agents
        for i, a in enumerate(actions):
            self.last_dirs[i] = a
            dr, dc = ACTIONS[a]
            nr, nc = self.positions[i][0] + dr, self.positions[i][1] + dc
            if 0 <= nr < self.grid and 0 <= nc < self.grid:
                self.positions[i] = (nr, nc)
            else:
                rewards[i] -= self.wall_penalty
        winners = [i for i, p in enumerate(self.positions) if p == self.goal]
        if winners:
            rewards[winners[0]] += 10.0
            self.done = True
        elif self.t >= self.max_steps:
            self.done = True
        return [self._state(i) for i in range(self.n_agents)], rewards, self.done