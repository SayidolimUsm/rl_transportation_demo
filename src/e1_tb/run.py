"""Tabular agents on two-corner fire task. Tests:
   - 1-agent memory vs no-memory (state augmentation)
   - 2-agent consistency across seeds (per-agent RNG)
"""
import random

from env import DeliveryEnv, MultiDeliveryEnv, GRID, VIS_RADIUS
from agents import QLearning, SARSA, DynaQ
from evaluate import (evaluate_single, evaluate_multi,
                      show_episode, show_multi_episode,
                      plot_goal_distribution)

N_EPISODES   = 3000
ALPHA        = 0.1
GAMMA        = 0.95
INIT         = 5.0
STATE_MODE   = "pos_goal"
HOTSPOTS     = [(0, 0), (GRID - 1, GRID - 1)]
SIGMA        = 1.5
HOTSPOT_PROB = 0.7
WALL_PENALTY = 2.0


def make_single_env(seed, with_corners=False):
    return DeliveryEnv(
        seed=seed, grid=GRID, vis_radius=VIS_RADIUS,
        wall_penalty=WALL_PENALTY, state_mode=STATE_MODE,
        goal_mu=HOTSPOTS, goal_sigma=SIGMA, goal_hotspot_prob=HOTSPOT_PROB,
        corners=HOTSPOTS if with_corners else None,
    )


def make_multi_env(seed):
    return MultiDeliveryEnv(
        n_agents=2, seed=seed, grid=GRID, vis_radius=VIS_RADIUS,
        wall_penalty=WALL_PENALTY, state_mode=STATE_MODE,
        goal_mu=HOTSPOTS, goal_sigma=SIGMA, goal_hotspot_prob=HOTSPOT_PROB,
    )


def make_agent(kind, seed):
    rng = random.Random(seed)
    if kind == "qlearning":
        return QLearning(ALPHA, GAMMA, INIT, rng=rng)
    if kind == "sarsa":
        return SARSA(ALPHA, GAMMA, INIT, rng=rng)
    if kind == "dynaq":
        return DynaQ(ALPHA, GAMMA, n_plan=10, init_value=INIT, rng=rng)
    raise ValueError(kind)


def train_single(env, agent, n_episodes=N_EPISODES):
    is_sarsa = isinstance(agent, SARSA)
    for ep in range(n_episodes):
        s   = env.reset()
        eps = max(0.05, 1.0 * (1 - ep / n_episodes))
        if is_sarsa:
            a = agent.act(s, eps)
            while True:
                s2, r, done = env.step(a)
                a2 = agent.act(s2, eps)
                agent.update(s, a, r, s2, a2, done)
                s, a = s2, a2
                if done:
                    break
        else:
            while True:
                a = agent.act(s, eps)
                s2, r, done = env.step(a)
                agent.update(s, a, r, s2, done)
                s = s2
                if done:
                    break


def train_multi(env, agents, n_episodes=N_EPISODES):
    is_sarsa = isinstance(agents[0], SARSA)
    for ep in range(n_episodes):
        states = env.reset()
        eps    = max(0.05, 1.0 * (1 - ep / n_episodes))
        if is_sarsa:
            actions = [agents[i].act(states[i], eps) for i in range(env.n_agents)]
            while True:
                next_states, rewards, done = env.step(actions)
                next_actions = [agents[i].act(next_states[i], eps)
                                for i in range(env.n_agents)]
                for i in range(env.n_agents):
                    agents[i].update(states[i], actions[i], rewards[i],
                                     next_states[i], next_actions[i], done)
                states, actions = next_states, next_actions
                if done:
                    break
        else:
            while True:
                actions = [agents[i].act(states[i], eps) for i in range(env.n_agents)]
                next_states, rewards, done = env.step(actions)
                for i in range(env.n_agents):
                    agents[i].update(states[i], actions[i], rewards[i],
                                     next_states[i], done)
                states = next_states
                if done:
                    break


def run_all(seed, algos=("qlearning", "sarsa", "dynaq"), verbose=False):
    """Train every algo on 1-agent (memory + no-memory) and 2-agent setups.
    Returns a dict of metrics for each configuration."""
    out = {}
    for kind in algos:
        # 1-agent, no memory
        env = make_single_env(seed, with_corners=False)
        ag  = make_agent(kind, seed)
        train_single(env, ag)
        m1 = evaluate_single(env, ag.Q)
        out[f"{kind}_1_noMem"] = m1
        if verbose:
            print(f"  [{kind:9s}] 1-agent noMem    success={m1['success']:.2%}  "
                  f"steps={m1['mean_successful_steps']:.2f}")

        # 1-agent, with memory
        env_mem = make_single_env(seed, with_corners=True)
        ag_mem  = make_agent(kind, seed)
        train_single(env_mem, ag_mem)
        m1m = evaluate_single(env_mem, ag_mem.Q)
        out[f"{kind}_1_mem"] = m1m
        if verbose:
            print(f"  [{kind:9s}] 1-agent memory    success={m1m['success']:.2%}  "
                  f"steps={m1m['mean_successful_steps']:.2f}")

        # 2-agent, no memory
        menv = make_multi_env(seed)
        agents = [make_agent(kind, seed * 100 + i) for i in range(2)]
        train_multi(menv, agents)
        m2 = evaluate_multi(menv, [a.Q for a in agents])
        out[f"{kind}_2"] = m2
        if verbose:
            ratio = (m2['mean_total_agent_steps']
                     / (m1m['mean_successful_steps'] or 1.0))
            print(f"  [{kind:9s}] 2-agent           success={m2['success']:.2%}  "
                  f"first_arr={m2['mean_first_arrival']:.2f}  "
                  f"ratio_vs_mem={ratio:.2f}×")
    return out


def print_seed_table(all_results):
    print(f"\n{'seed':<5} {'algo':<10} {'1noMem':>8} {'1mem':>7} {'2agent':>8} "
          f"{'2/1mem':>7}")
    print("-" * 52)
    for seed, res in all_results.items():
        for kind in ("qlearning", "sarsa", "dynaq"):
            m1 = res[f"{kind}_1_noMem"]
            mm = res[f"{kind}_1_mem"]
            m2 = res[f"{kind}_2"]
            ratio = m2['mean_total_agent_steps'] / (mm['mean_successful_steps'] or 1.0)
            print(f"{seed:<5} {kind:<10} "
                  f"{m1['success']:>7.0%} {mm['success']:>7.0%} "
                  f"{m2['success']:>8.0%} {ratio:>6.2f}×")


def main():
    # -- probe + header (one source of truth for config)
    probe = make_single_env(0)
    plot_goal_distribution(probe, n_samples=5000, seed=0)
    print(f"\ngrid={probe.grid}×{probe.grid}, store={probe.store}, "
          f"vis_radius={probe.vis_radius}, episodes={N_EPISODES}, "
          f"wall_penalty={probe.wall_penalty}, hotspots={probe.goal_mu}, "
          f"sigma={probe.goal_sigma}, hotspot_prob={probe.goal_hotspot_prob}")

    # -- multi-seed sweep
    print("\n" + "=" * 52)
    print("MULTI-SEED SWEEP  (1noMem / 1mem / 2agent success, 2/1mem ratio)")
    print("=" * 52)
    all_results = {}
    for seed in range(10):
        all_results[seed] = run_all(seed)
    print_seed_table(all_results)

    # -- detailed output for seed 0 (paths + one example)
    print("\n" + "=" * 52)
    print("DETAILED  seed=0")
    print("=" * 52)
    env = make_single_env(0, with_corners=False)
    ag  = make_agent("qlearning", 0)
    train_single(env, ag)
    show_episode(env, ag.Q, seed=123, title="1-agent, no memory")

    env_mem = make_single_env(0, with_corners=True)
    ag_mem  = make_agent("qlearning", 0)
    train_single(env_mem, ag_mem)
    show_episode(env_mem, ag_mem.Q, seed=123, title="1-agent, with memory")

    menv   = make_multi_env(0)
    agents = [
        make_agent("qlearning", seed * 100 + 0),         # init=5.0
        QLearning(ALPHA, GAMMA, 5.5,
                rng=random.Random(seed * 100 + 1)),    # init=5.5
    ]
    train_multi(menv, agents)
    show_multi_episode(menv, [a.Q for a in agents], seed=123,
                       title="2-agent, independent (per-agent RNG)")


if __name__ == "__main__":
    main()