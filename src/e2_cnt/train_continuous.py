"""Two experiments:

  1. DQN with corner-searched memory (fix for the 13-53% local minimum).
  2. Two-agent DQN, independent learners, no memory (replicates the
     tabular duplication test at the neural scale).

Experiment 2 reuses Experiment 1's single-agent memory numbers as the
1-agent baseline for the 2/1mem ratio, so nothing is retrained twice.
"""
import random

import numpy as np
import torch

from env_continuous import (ContinuousDeliveryEnv, MultiContinuousEnv,
                            STEP_SIZE)
from dqn import DQNAgent

torch.set_num_threads(4)

N_EPISODES = 1000
EPS_END    = 0.15
SEEDS      = [0, 1, 2]


def train_single(seed, track_corners, n_episodes=N_EPISODES):
    random.seed(seed)
    np.random.seed(seed)
    env   = ContinuousDeliveryEnv(seed=seed, track_corners=track_corners)
    agent = DQNAgent(obs_dim=env.obs_dim, seed=seed)
    for ep in range(n_episodes):
        s    = env.reset()
        eps  = max(EPS_END, 1.0 * (1 - ep / n_episodes))
        done = False
        if ep % 500 == 0:
            print(f"  ep {ep:4d}  eps={eps:.2f}  buffer={len(agent.buffer)}")
        while not done:
            a = agent.act(s, eps)
            s2, r, done = env.step(a)
            agent.store(s, a, r, s2, done)
            agent.train_step()
            s = s2
    return env, agent


def train_multi(seed, n_agents=2, track_corners=False, n_episodes=N_EPISODES):
    random.seed(seed)
    np.random.seed(seed)
    env    = MultiContinuousEnv(n_agents=n_agents, seed=seed,
                                track_corners=track_corners)
    agents = [DQNAgent(obs_dim=env.obs_dim, seed=seed * 100 + i)
              for i in range(n_agents)]
    for ep in range(n_episodes):
        states = env.reset()
        eps    = max(EPS_END, 1.0 * (1 - ep / n_episodes))
        done   = False
        if ep % 500 == 0:
            for i in range(n_agents):
                print(f"  ep {ep:4d}  eps={eps:.2f}  buffer={len(agents[i].buffer)}")
        while not done:
            actions = [agents[i].act(states[i], eps) for i in range(n_agents)]
            states2, rewards, done = env.step(actions)
            for i in range(n_agents):
                agents[i].store(states[i], actions[i], rewards[i],
                                states2[i], done)
                agents[i].train_step()
            states = states2
    return env, agents


def evaluate_single(env, agent, n_episodes=300, seed=999):
    env.rng = random.Random(seed)
    reached, succ_steps = 0, []
    for _ in range(n_episodes):
        s = env.reset()
        n = 0
        while True:
            a = agent.act(s, eps=0.0)
            s, _, done = env.step(a)
            n += 1
            if done:
                break
        dist = ((env.pos[0] - env.goal[0]) ** 2 +
                (env.pos[1] - env.goal[1]) ** 2) ** 0.5
        if dist < STEP_SIZE:
            reached += 1
            succ_steps.append(n)
    return dict(success=reached / n_episodes,
                mean_successful_steps=(sum(succ_steps) / len(succ_steps)
                                       if succ_steps else 0.0))


def evaluate_multi(env, agents, n_episodes=300, seed=999):
    env.rng = random.Random(seed)
    reached, first_arrivals, total_steps = 0, [], []
    for _ in range(n_episodes):
        states   = env.reset()
        t        = 0
        winner_t = None
        while True:
            actions = [agents[i].act(states[i], eps=0.0)
                       for i in range(env.n_agents)]
            states, rewards, done = env.step(actions)
            t += 1
            if winner_t is None and any(r > 0 for r in rewards):
                winner_t = t
            if done:
                break
        if winner_t is not None:
            reached += 1
        first_arrivals.append(winner_t or t)
        total_steps.append(t * env.n_agents)
    return dict(success=reached / n_episodes,
                mean_first_arrival=sum(first_arrivals) / n_episodes,
                mean_total_agent_steps=sum(total_steps) / n_episodes)


def main():
    # ================================================================
    # Experiment 1 — DQN with corner-searched memory
    # ================================================================
    print("=" * 60)
    print(f"EXPERIMENT 1: DQN with memory (track_corners=True, eps_end={EPS_END})")
    print("=" * 60)
    print(f"{'seed':<5}{'success':>10}{'succ_steps':>12}")
    print("-" * 27)

    exp1_results = {}
    for seed in SEEDS:
        env, agent = train_single(seed, track_corners=True)
        m = evaluate_single(env, agent)
        exp1_results[seed] = m
        print(f"{seed:<5}{m['success']:>9.1%}{m['mean_successful_steps']:>12.2f}")

    mean_success = sum(r["success"] for r in exp1_results.values()) / len(SEEDS)
    print(f"\nmean success: {mean_success:.1%}")

    # ================================================================
    # Experiment 2 — two-agent DQN, independent learners, no memory
    # ================================================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: 2-agent DQN, independent learners, no memory")
    print("=" * 60)
    print(f"{'seed':<5}{'success':>10}{'first_arr':>12}{'total_steps':>14}"
          f"{'2/1mem':>10}")
    print("-" * 51)

    for seed in SEEDS:
        env, agents = train_multi(seed, n_agents=2, track_corners=False)
        m2 = evaluate_multi(env, agents)
        baseline = exp1_results[seed]["mean_successful_steps"] or 1.0
        ratio    = m2["mean_total_agent_steps"] / baseline
        print(f"{seed:<5}{m2['success']:>9.1%}{m2['mean_first_arrival']:>12.2f}"
              f"{m2['mean_total_agent_steps']:>14.2f}{ratio:>10.2f}x")

    print("\n'2/1mem' = 2-agent total steps / single-agent memory steps.")
    print("  ~1.2x → agents split (one won, one wasted a partial trip)")
    print("  ~5x   → both chased the same corner (duplication)")


if __name__ == "__main__":
    main()