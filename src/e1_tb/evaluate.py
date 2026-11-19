"""Rollout, metrics, ASCII rendering, goal-distribution heatmap."""
import random
from env import ACTION_ARROWS
from agents import argmax


def evaluate_single(env, Q, n_episodes=300, seed=999):
    env.rng = random.Random(seed)
    reached, steps, returns = 0, [], []
    successful_steps = []
    for _ in range(n_episodes):
        s = env.reset()
        ep_r, ep_n = 0.0, 0
        while True:
            a = argmax(Q[s])
            s, r, done = env.step(a)
            ep_r += r
            ep_n += 1
            if done:
                break
        if env.pos == env.goal:
            reached += 1
            successful_steps.append(ep_n)
        steps.append(ep_n)
        returns.append(ep_r)
    n = n_episodes
    return dict(
        success=reached / n,
        mean_steps=sum(steps) / n,
        mean_successful_steps=(sum(successful_steps) / len(successful_steps)
                               if successful_steps else 0.0),
        mean_return=sum(returns) / n,
    )


def evaluate_multi(env, Qs, n_episodes=300, seed=999):
    env.rng = random.Random(seed)
    reached, first, total = 0, [], []
    for _ in range(n_episodes):
        states = env.reset()
        t, first_at = 0, None
        while True:
            actions = [argmax(Qs[i][states[i]]) for i in range(env.n_agents)]
            states, rewards, done = env.step(actions)
            t += 1
            if first_at is None and any(r > 0 for r in rewards):
                first_at = t
            if done:
                break
        if first_at is not None:
            reached += 1
        first.append(first_at or t)
        total.append(t * env.n_agents)
    n = n_episodes
    return dict(success=reached / n,
                mean_first_arrival=sum(first) / n,
                mean_total_agent_steps=sum(total) / n)


def render_path(path, actions, goal, store, grid_size):
    grid = [["." for _ in range(grid_size)] for _ in range(grid_size)]
    for i, a in enumerate(actions):
        r, c = path[i]
        if (r, c) != store:
            grid[r][c] = ACTION_ARROWS[a]
    grid[goal[0]][goal[1]] = "G"
    grid[store[0]][store[1]] = "S"
    return "\n".join("  ".join(row) for row in grid)


def show_episode(env, Q, seed, title):
    env.rng = random.Random(seed)
    s = env.reset()
    path, taken = [env.pos], []
    while True:
        a = argmax(Q[s])
        taken.append(a)
        s, _, done = env.step(a)
        path.append(env.pos)
        if done:
            break
    print(f"\n--- {title} ---")
    print(f"goal = {env.goal}, steps = {len(taken)}, reached = {env.pos == env.goal}")
    print(render_path(path, taken, env.goal, env.store, env.grid))
    print("actions:", " ".join(ACTION_ARROWS[a] for a in taken))


def plot_goal_distribution(env, n_samples=5000, seed=0):
    """Sample goal spawns and print an ASCII heatmap of the empirical density."""
    env.rng = random.Random(seed)
    counts  = [[0] * env.grid for _ in range(env.grid)]
    for _ in range(n_samples):
        g = env._spawn_goal()
        counts[g[0]][g[1]] += 1
    peak   = max(max(row) for row in counts)
    chars  = " .:-=+*#%@"
    print(f"\nGoal distribution over {n_samples} samples "
          f"(hotspot_mu={env.goal_mu}, sigma={env.goal_sigma}, "
          f"hotspot_prob={env.goal_hotspot_prob}, peak_cell={peak}):")
    for r in range(env.grid):
        row = []
        for c in range(env.grid):
            if (r, c) == env.store:
                row.append("  S ")
                continue
            level = int(counts[r][c] / peak * (len(chars) - 1)) if peak else 0
            row.append(f" {chars[level]}  ")
        print("".join(row))

def render_multi_path(paths, actions, goal, store, grid_size):
    """Overlay two agents: agent 0 = lowercase arrows, agent 1 = UPPERCASE.
    Overlap is marked X."""
    grid = [["." for _ in range(grid_size)] for _ in range(grid_size)]
    for i, (path, acts) in enumerate(zip(paths, actions)):
        marker_set = (ACTION_ARROWS if i == 0
                      else [a.upper() for a in ACTION_ARROWS])
        for j, a in enumerate(acts):
            r, c = path[j]
            if (r, c) in (store, goal):
                continue
            grid[r][c] = "X" if grid[r][c] != "." else marker_set[a]
    grid[goal[0]][goal[1]] = "G"
    grid[store[0]][store[1]] = "S"
    return "\n".join("  ".join(row) for row in grid)


def show_multi_episode(env, Qs, seed, title):
    env.rng = random.Random(seed)
    states = env.reset()
    paths = [[env.positions[i]] for i in range(env.n_agents)]
    taken = [[] for _ in range(env.n_agents)]
    steps = 0
    while True:
        actions = [argmax(Qs[i][states[i]]) for i in range(env.n_agents)]
        states, rewards, done = env.step(actions)
        steps += 1
        for i in range(env.n_agents):
            paths[i].append(env.positions[i])
            taken[i].append(actions[i])
        if done:
            break
    winners = [i for i, p in enumerate(env.positions) if p == env.goal]
    print(f"\n--- {title} ---")
    print(f"goal = {env.goal}, steps = {steps}, "
          f"reached_by = {winners if winners else 'nobody'}")
    print(render_multi_path(paths, taken, env.goal, env.store, env.grid))

def evaluate_single(env, Q, n_episodes=300, seed=999):
    env.rng = random.Random(seed)
    reached, steps, returns, succ_steps = 0, [], [], []
    for _ in range(n_episodes):
        s = env.reset()
        ep_r, ep_n = 0.0, 0
        while True:
            a = argmax(Q[s])
            s, r, done = env.step(a)
            ep_r += r
            ep_n += 1
            if done:
                break
        if env.pos == env.goal:
            reached += 1
            succ_steps.append(ep_n)
        steps.append(ep_n)
        returns.append(ep_r)
    n = n_episodes
    return dict(
        success=reached / n,
        mean_steps=sum(steps) / n,
        mean_successful_steps=(sum(succ_steps) / len(succ_steps)
                               if succ_steps else 0.0),
        mean_return=sum(returns) / n,
    )