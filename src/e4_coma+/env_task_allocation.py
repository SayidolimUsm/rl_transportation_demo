"""Simple 3-drone task-allocation environment on a 6x6 grid.

Actions: 0=up, 1=down, 2=left, 3=right.
Each drone can carry at most one task. A task is completed when its assigned
drone reaches its drop-off cell after picking it up.

The transition function is pure: it does not mutate the environment. This
allows step() to evaluate counterfactual actions for difference rewards.
"""
import random

GRID = 6
N_AGENTS = 3
N_TASKS = 3
MAX_STEPS = 100
STEP_REWARD = -0.1
DELIVERY_REWARD = 5.0
ACTIONS = [(0, 1), (0, -1), (-1, 0), (1, 0)]  # up, down, left, right
DEPOT = (0, 0)


class GridCityEnv:
    def __init__(
        self,
        seed=0,
        grid=GRID,
        n_agents=N_AGENTS,
        n_tasks=N_TASKS,
        max_steps=MAX_STEPS,
    ):
        if grid < 2:
            raise ValueError("grid must be at least 2")
        if n_agents < 1 or n_tasks < 1:
            raise ValueError("n_agents and n_tasks must be positive")

        self.rng = random.Random(seed)
        self.grid = grid
        self.n_agents = n_agents
        self.n_tasks = n_tasks
        self.max_steps = max_steps
        self.n_actions = len(ACTIONS)

        # Observation = agent ID + all drone states + all task states + time.
        self.obs_dim = n_agents + 3 * n_agents + 5 * n_tasks + 1
        self._reset_state()

    def _new_task(self):
        while True:
            pickup = (
                self.rng.randrange(self.grid),
                self.rng.randrange(self.grid),
            )
            dropoff = (
                self.rng.randrange(self.grid),
                self.rng.randrange(self.grid),
            )
            if pickup != dropoff:
                return {"pickup": pickup, "dropoff": dropoff}

    def _reset_state(self):
        self.t = 0
        self.positions = [DEPOT] * self.n_agents
        self.carrying = [None] * self.n_agents
        self.task_delivered = [False] * self.n_tasks
        self.done = False
        self.tasks = [self._new_task() for _ in range(self.n_tasks)]

    def reset(self, seed=None):
        if seed is not None:
            self.rng = random.Random(seed)
        self._reset_state()
        return self._obs_all()

    def step(self, joint_action):
        if self.done:
            raise RuntimeError("Episode is finished; call reset() before step().")
        if len(joint_action) != self.n_agents:
            raise ValueError(f"Expected {self.n_agents} actions")
        if any(int(a) not in range(self.n_actions) for a in joint_action):
            raise ValueError("Each action must be an integer in [0, 3]")

        joint_action = [int(a) for a in joint_action]
        previous_state = self._snapshot()
        next_state, n_completions = self._transition(previous_state, joint_action)

        # For each drone and each possible action, hold the other drones'
        # sampled actions fixed and calculate the resulting team reward.
        per_alternative = [
            [0.0] * self.n_actions for _ in range(self.n_agents)
        ]
        for i in range(self.n_agents):
            for alternative_action in range(self.n_actions):
                alternative_joint_action = list(joint_action)
                alternative_joint_action[i] = alternative_action
                _, alternative_completions = self._transition(
                    previous_state, alternative_joint_action
                )
                per_alternative[i][alternative_action] = (
                    STEP_REWARD + DELIVERY_REWARD * alternative_completions
                )

        self._apply(next_state)
        team_reward = STEP_REWARD + DELIVERY_REWARD * n_completions

        info = {
            "per_alternative": per_alternative,
            "team_reward": team_reward,
            "n_delivered": sum(self.task_delivered),
            "deliveries_this_step": n_completions,
        }
        return self._obs_all(), team_reward, self.done, info

    def _snapshot(self):
        # Task definitions are immutable during an episode, so sharing this
        # list between snapshots is safe.
        return {
            "positions": list(self.positions),
            "carrying": list(self.carrying),
            "task_delivered": list(self.task_delivered),
            "tasks": self.tasks,
        }

    def _transition(self, state, joint_action):
        positions = list(state["positions"])
        carrying = list(state["carrying"])
        task_delivered = list(state["task_delivered"])
        n_completions = 0

        # 1. Move all drones. Grid boundaries act as walls.
        for i, action in enumerate(joint_action):
            dx, dy = ACTIONS[action]
            x, y = positions[i]
            positions[i] = (
                max(0, min(self.grid - 1, x + dx)),
                max(0, min(self.grid - 1, y + dy)),
            )

        # 2. Deliver carried tasks.
        for i in range(self.n_agents):
            task_id = carrying[i]
            if task_id is None:
                continue
            if positions[i] == state["tasks"][task_id]["dropoff"]:
                task_delivered[task_id] = True
                carrying[i] = None
                n_completions += 1

        # 3. Pick up available tasks. If multiple drones reach the same
        # pickup in one step, the lower-index drone gets priority.
        claimed_this_step = set()
        for i in range(self.n_agents):
            if carrying[i] is not None:
                continue
            for task_id, task in enumerate(state["tasks"]):
                if task_delivered[task_id] or task_id in claimed_this_step:
                    continue
                if positions[i] == task["pickup"]:
                    carrying[i] = task_id
                    claimed_this_step.add(task_id)
                    break

        return {
            "positions": positions,
            "carrying": carrying,
            "task_delivered": task_delivered,
            "tasks": state["tasks"],
        }, n_completions

    def _apply(self, next_state):
        self.positions = next_state["positions"]
        self.carrying = next_state["carrying"]
        self.task_delivered = next_state["task_delivered"]
        self.t += 1
        if all(self.task_delivered) or self.t >= self.max_steps:
            self.done = True

    def _obs_all(self):
        return [self._obs_agent(i) for i in range(self.n_agents)]

    def _obs_agent(self, agent_id):
        scale = float(self.grid - 1)
        agent_id_onehot = [
            1.0 if agent_id == j else 0.0 for j in range(self.n_agents)
        ]

        drone_features = []
        for i in range(self.n_agents):
            x, y = self.positions[i]
            carrying_code = (
                self.n_tasks if self.carrying[i] is None else self.carrying[i]
            )
            drone_features.extend(
                [x / scale, y / scale, carrying_code / max(1, self.n_tasks)]
            )

        task_features = []
        for task_id, task in enumerate(self.tasks):
            task_features.extend(
                [
                    task["pickup"][0] / scale,
                    task["pickup"][1] / scale,
                    task["dropoff"][0] / scale,
                    task["dropoff"][1] / scale,
                    1.0 if self.task_delivered[task_id] else 0.0,
                ]
            )

        return (
            agent_id_onehot
            + drone_features
            + task_features
            + [self.t / self.max_steps]
        )
