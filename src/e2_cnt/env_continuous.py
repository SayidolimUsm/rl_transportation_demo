"""Continuous-position delivery environments.

Single and multi-agent. `track_corners` adds two searched-flags to the
observation, mirroring the tabular memory fix.
"""
import math
import random

GRID       = 7.0
STORE      = (3.5, 3.5)
MAX_STEPS  = 60
VIS_RADIUS = 2.0
STEP_SIZE  = 0.5
ACTIONS    = [(-STEP_SIZE, 0), (STEP_SIZE, 0),
              (0, -STEP_SIZE), (0, STEP_SIZE)]


def _make_state(pos, goal, vis_radius, searched, track_corners, grid):
    dx = goal[0] - pos[0]
    dy = goal[1] - pos[1]
    visible = math.hypot(dx, dy) <= vis_radius
    base = [pos[0] / grid, pos[1] / grid,
            (dx / grid) if visible else 0.0,
            (dy / grid) if visible else 0.0,
            1.0 if visible else 0.0]
    if track_corners:
        base += [1.0 if s else 0.0 for s in searched]
    return base


def _mark_searched(pos, corners, searched):
    for i, (cx, cy) in enumerate(corners):
        if abs(pos[0] - cx) < STEP_SIZE and abs(pos[1] - cy) < STEP_SIZE:
            searched[i] = True


def _spawn_goal(rng, grid, sigma, hotspot_prob):
    if rng.random() < hotspot_prob:
        cx, cy = rng.choice([(STEP_SIZE, STEP_SIZE),
                             (grid - STEP_SIZE, grid - STEP_SIZE)])
        gx = cx + rng.gauss(0, sigma)
        gy = cy + rng.gauss(0, sigma)
        return (max(STEP_SIZE, min(grid - STEP_SIZE, gx)),
                max(STEP_SIZE, min(grid - STEP_SIZE, gy)))
    return (rng.uniform(STEP_SIZE, grid - STEP_SIZE),
            rng.uniform(STEP_SIZE, grid - STEP_SIZE))


class ContinuousDeliveryEnv:
    def __init__(self, seed=0, grid=GRID, store=STORE,
                 vis_radius=VIS_RADIUS, max_steps=MAX_STEPS,
                 hotspot_prob=0.7, sigma=0.7, wall_penalty=0.0,
                 track_corners=False):
        self.rng           = random.Random(seed)
        self.grid          = grid
        self.store         = store
        self.vis_radius    = vis_radius
        self.max_steps     = max_steps
        self.hotspot_prob  = hotspot_prob
        self.sigma         = sigma
        self.wall_penalty  = wall_penalty
        self.track_corners = track_corners
        self.corners       = [(STEP_SIZE, STEP_SIZE),
                              (grid - STEP_SIZE, grid - STEP_SIZE)]

    @property
    def obs_dim(self):
        return 7 if self.track_corners else 5

    def reset(self, goal=None, seed=None):
        if seed is not None:
            self.rng = random.Random(seed)
        self.t        = 0
        self.pos      = self.store
        self.goal     = goal if goal is not None else _spawn_goal(
            self.rng, self.grid, self.sigma, self.hotspot_prob)
        self.searched = [False] * len(self.corners)
        self.done     = False
        return self._state()

    def _state(self):
        return _make_state(self.pos, self.goal, self.vis_radius,
                           self.searched, self.track_corners, self.grid)

    def step(self, a):
        self.t += 1
        dx, dy = ACTIONS[a]
        nx, ny = self.pos[0] + dx, self.pos[1] + dy
        bumped = not (STEP_SIZE <= nx <= self.grid - STEP_SIZE and
                      STEP_SIZE <= ny <= self.grid - STEP_SIZE)
        self.pos = (max(STEP_SIZE, min(self.grid - STEP_SIZE, nx)),
                    max(STEP_SIZE, min(self.grid - STEP_SIZE, ny)))

        if self.track_corners:
            _mark_searched(self.pos, self.corners, self.searched)

        dist = math.hypot(self.goal[0] - self.pos[0],
                          self.goal[1] - self.pos[1])
        r = -1.0
        if bumped:
            r -= self.wall_penalty
        if dist < STEP_SIZE:
            r += 10.0
            self.done = True
        elif self.t >= self.max_steps:
            self.done = True
        return self._state(), r, self.done


class MultiContinuousEnv:
    """n agents, one shared goal. Only the first arriver gets +10."""
    def __init__(self, n_agents=2, seed=0, grid=GRID, store=STORE,
                 vis_radius=VIS_RADIUS, max_steps=MAX_STEPS,
                 hotspot_prob=0.7, sigma=0.7, wall_penalty=0.0,
                 track_corners=False):
        self.rng           = random.Random(seed)
        self.n_agents      = n_agents
        self.grid          = grid
        self.store         = store
        self.vis_radius    = vis_radius
        self.max_steps     = max_steps
        self.hotspot_prob  = hotspot_prob
        self.sigma         = sigma
        self.wall_penalty  = wall_penalty
        self.track_corners = track_corners
        self.corners       = [(STEP_SIZE, STEP_SIZE),
                              (grid - STEP_SIZE, grid - STEP_SIZE)]

    @property
    def obs_dim(self):
        return 7 if self.track_corners else 5

    def reset(self, goal=None, seed=None):
        if seed is not None:
            self.rng = random.Random(seed)
        self.t        = 0
        self.positions = [self.store] * self.n_agents
        self.searched  = [[False] * len(self.corners) for _ in range(self.n_agents)]
        self.goal      = goal if goal is not None else _spawn_goal(
            self.rng, self.grid, self.sigma, self.hotspot_prob)
        self.done      = False
        return [self._state(i) for i in range(self.n_agents)]

    def _state(self, i):
        return _make_state(self.positions[i], self.goal, self.vis_radius,
                           self.searched[i], self.track_corners, self.grid)

    def step(self, actions):
        self.t += 1
        rewards = [-1.0] * self.n_agents
        for i, a in enumerate(actions):
            dx, dy = ACTIONS[a]
            nx = self.positions[i][0] + dx
            ny = self.positions[i][1] + dy
            bumped = not (STEP_SIZE <= nx <= self.grid - STEP_SIZE and
                          STEP_SIZE <= ny <= self.grid - STEP_SIZE)
            self.positions[i] = (
                max(STEP_SIZE, min(self.grid - STEP_SIZE, nx)),
                max(STEP_SIZE, min(self.grid - STEP_SIZE, ny)),
            )
            if self.track_corners:
                _mark_searched(self.positions[i], self.corners, self.searched[i])
            if bumped:
                rewards[i] -= self.wall_penalty

        winners = [i for i, p in enumerate(self.positions)
                   if math.hypot(self.goal[0] - p[0],
                                 self.goal[1] - p[1]) < STEP_SIZE]
        if winners:
            rewards[winners[0]] += 10.0
            self.done = True
        elif self.t >= self.max_steps:
            self.done = True
        return [self._state(i) for i in range(self.n_agents)], rewards, self.done