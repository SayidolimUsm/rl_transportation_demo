"""Grid-city task allocation: 2 drones, 3 deliveries, 6x6 grid, full observability.

The reward function is PURE: `_transition(state, joint_action)` mutates nothing
and returns (next_state, n_completions). `step()` calls it once for the actual
joint action and n_agents * n_actions times for counterfactuals, producing a
`per_alternative` table that the trainer uses for difference rewards.

No shaping, no wall penalty. Reward: -0.1 per step, +5 per delivery.
"""
import random

GRID         = 6
N_AGENTS     = 2
N_TASKS      = 3
MAX_STEPS    = 100
STEP_REWARD  = -0.1
DELIVERY_REW = 5.0
ACTIONS      = [(0, 1), (0, -1), (-1, 0), (1, 0)]   # up, down, left, right
DEPOT        = (0, 0)


class GridCityEnv:
    def __init__(self, seed=0, grid=GRID, n_agents=N_AGENTS,
                 n_tasks=N_TASKS, max_steps=MAX_STEPS):
        self.rng       = random.Random(seed)
        self.grid      = grid
        self.n_agents  = n_agents
        self.n_tasks   = n_tasks
        self.max_steps = max_steps
        self.n_actions = len(ACTIONS)
        self.obs_dim   = 2 + 3 * n_agents + 5 * n_tasks + 1
        self._reset_state()

    def _reset_state(self):
        self.t              = 0
        self.positions      = [DEPOT] * self.n_agents
        self.carrying       = [None] * self.n_agents
        self.task_delivered = [False] * self.n_tasks
        self.done           = False
        self.tasks          = [self._new_task() for _ in range(self.n_tasks)]

    def _new_task(self):
        while True:
            p = (self.rng.randrange(self.grid), self.rng.randrange(self.grid))
            d = (self.rng.randrange(self.grid), self.rng.randrange(self.grid))
            if p != d:
                return {"pickup": p, "dropoff": d}

    def reset(self, seed=None):
        if seed is not None:
            self.rng = random.Random(seed)
        self._reset_state()
        return self._obs_all()

    def step(self, joint_action):
        prev = self._snapshot()
        next_state, n_completions = self._transition(prev, joint_action)

        per_alt = [[0.0] * self.n_actions for _ in range(self.n_agents)]
        for i in range(self.n_agents):
            for a_alt in range(self.n_actions):
                joint_alt = list(joint_action)
                joint_alt[i] = a_alt
                _, n_comp_alt = self._transition(prev, joint_alt)
                per_alt[i][a_alt] = STEP_REWARD + DELIVERY_REW * n_comp_alt

        self._apply(next_state)
        team_reward = STEP_REWARD + DELIVERY_REW * n_completions
        return self._obs_all(), team_reward, self.done, {
            "per_alternative": per_alt,
            "team_reward":     team_reward,
            "n_delivered":     sum(self.task_delivered),
        }

    def _snapshot(self):
        return {
            "positions":      list(self.positions),
            "carrying":       list(self.carrying),
            "task_delivered": list(self.task_delivered),
            "tasks":          self.tasks,   # never mutated in place
        }

    def _transition(self, state, joint_action):
        positions      = list(state["positions"])
        carrying       = list(state["carrying"])
        task_delivered = list(state["task_delivered"])
        n_completions  = 0

        # 1) move (clamped at grid edges — "stay" via bumping a wall)
        for i, a in enumerate(joint_action):
            dx, dy = ACTIONS[a]
            positions[i] = (max(0, min(self.grid - 1, positions[i][0] + dx)),
                            max(0, min(self.grid - 1, positions[i][1] + dy)))

        # 2) deliver
        for i in range(self.n_agents):
            if carrying[i] is None:
                continue
            if positions[i] == state["tasks"][carrying[i]]["dropoff"]:
                task_delivered[carrying[i]] = True
                carrying[i] = None
                n_completions += 1

        # 3) claim (agent-order tie-break: agent 0 wins contested pickups)
        claimed = set()
        for i in range(self.n_agents):
            if carrying[i] is not None:
                continue
            for idx, task in enumerate(state["tasks"]):
                if task_delivered[idx] or idx in claimed:
                    continue
                if positions[i] == task["pickup"]:
                    carrying[i] = idx
                    claimed.add(idx)
                    break

        return {
            "positions":      positions,
            "carrying":       carrying,
            "task_delivered": task_delivered,
            "tasks":          state["tasks"],
        }, n_completions

    def _apply(self, next_state):
        self.positions      = next_state["positions"]
        self.carrying       = next_state["carrying"]
        self.task_delivered = next_state["task_delivered"]
        self.t             += 1
        if all(self.task_delivered) or self.t >= self.max_steps:
            self.done = True

    def _obs_all(self):
        return [self._obs_agent(i) for i in range(self.n_agents)]

    def _obs_agent(self, agent_id):
        g = self.grid - 1
        id_onehot = [1.0 if agent_id == j else 0.0
                     for j in range(self.n_agents)]
        drones = []
        for i in range(self.n_agents):
            x, y = self.positions[i]
            carry_code = 3 if self.carrying[i] is None else self.carrying[i]
            drones.extend([x / g, y / g, carry_code / 3.0])
        tasks = []
        for idx in range(self.n_tasks):
            t = self.tasks[idx]
            tasks.extend([t["pickup"][0] / g, t["pickup"][1] / g,
                          t["dropoff"][0] / g, t["dropoff"][1] / g,
                          1.0 if self.task_delivered[idx] else 0.0])
        return id_onehot + drones + tasks + [self.t / self.max_steps]