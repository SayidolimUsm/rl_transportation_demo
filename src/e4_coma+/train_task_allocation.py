"""Train and compare shared-reward PPO vs difference-reward PPO.

Run:
    python train_task_allocation.py

This is a deliberately small prototype. It trains three independent drone
actors with centralized critics on the custom 6x6 GridCityEnv.
"""
import csv
import os
import random
import time

import torch
import torch.nn.functional as F

from env_task_allocation import GridCityEnv, N_AGENTS, N_TASKS, GRID
from ppo_multi import Actor, CentralizedCritic, RolloutBuffer

torch.set_num_threads(4)

# PPO hyperparameters (kept close to train_ppo_single.py).
T = 2048
GAMMA = 0.95
LAM = 0.95
CLIP = 0.2
K_EPOCHS = 4
MINIBATCH = 64
ENT_COEF = 0.01
VF_COEF = 0.5
LR = 3e-4
HIDDEN = 64
ITERATIONS = 100
SEEDS = [0, 1, 2]

EVAL_EPISODES = 100
EVAL_INTERVAL = 10
EVAL_SEED = 999
MODES = ("shared", "difference")
RESULTS_DIR = "results_task_allocation"


def collect_rollout(env, actors, buffer):
    """Collect complete episodes so GAE never crosses episode boundaries."""
    episode_returns = []
    episode_deliveries = []
    episode_lengths = []

    while buffer.ptr + env.max_steps <= buffer.T:
        observations = env.reset()
        episode_return = 0.0
        episode_steps = 0
        done = False

        while not done:
            actions = []
            logprobs = []
            probabilities = []

            with torch.no_grad():
                for i, actor in enumerate(actors):
                    obs_tensor = torch.as_tensor(
                        observations[i], dtype=torch.float32
                    )
                    action, logprob, _, probs = actor.act(obs_tensor)
                    actions.append(int(action.item()))
                    logprobs.append(float(logprob.item()))
                    probabilities.append(probs.detach().cpu())

            next_observations, reward, done, info = env.step(actions)
            buffer.add(
                observations,
                actions,
                logprobs,
                probabilities,
                reward,
                info["per_alternative"],
                done,
            )

            observations = next_observations
            episode_return += reward
            episode_steps += 1

        episode_returns.append(episode_return)
        episode_deliveries.append(info["n_delivered"])
        episode_lengths.append(episode_steps)

    return episode_returns, episode_deliveries, episode_lengths


def calculate_advantages(mode, buffer, critics):
    """Compute GAE using the critic(s) before any optimization update."""
    with torch.no_grad():
        joint_obs = buffer.joint_obs
        if mode == "shared":
            values = critics[0](joint_obs)
            buffer.compute_gae_shared(values, last_value=0.0)
        else:
            values = torch.stack(
                [critic(joint_obs) for critic in critics], dim=-1
            )
            buffer.compute_difference_rewards()
            buffer.compute_gae_difference(
                values, last_values_per_agent=[0.0] * N_AGENTS
            )


def update(mode, actors, actor_optimizers, critics, critic_optimizers, buffer):
    calculate_advantages(mode, buffer, critics)

    ptr = buffer.ptr
    observations = buffer.obs[:ptr]
    joint_observations = buffer.joint_obs
    actions = buffer.actions[:ptr]
    old_logprobs = buffer.logprobs[:ptr]
    advantages = buffer.adv[:ptr].clone()
    returns = buffer.ret[:ptr]

    # Normalize each drone's advantage stream independently.
    for i in range(N_AGENTS):
        advantages[:, i] = (
            advantages[:, i] - advantages[:, i].mean()
        ) / (advantages[:, i].std(unbiased=False) + 1e-8)

    policy_losses = []
    value_losses = []
    entropies = []

    for _ in range(K_EPOCHS):
        permutation = torch.randperm(ptr)
        for start in range(0, ptr, MINIBATCH):
            batch = permutation[start : start + MINIBATCH]
            actor_loss_sum = 0.0
            entropy_sum = 0.0

            for i, actor in enumerate(actors):
                new_logprob, entropy = actor.evaluate(
                    observations[batch, i], actions[batch, i]
                )
                ratio = (new_logprob - old_logprobs[batch, i]).exp()
                adv = advantages[batch, i]
                surrogate_1 = ratio * adv
                surrogate_2 = (
                    ratio.clamp(1.0 - CLIP, 1.0 + CLIP) * adv
                )
                actor_loss_sum = actor_loss_sum - torch.minimum(
                    surrogate_1, surrogate_2
                ).mean() / N_AGENTS
                entropy_sum = entropy_sum + entropy.mean() / N_AGENTS

            if mode == "shared":
                predicted_values = critics[0](joint_observations[batch])
                value_loss = F.mse_loss(
                    predicted_values, returns[batch, 0]
                )
            else:
                per_agent_value_losses = []
                for i, critic in enumerate(critics):
                    predicted_values = critic(joint_observations[batch])
                    per_agent_value_losses.append(
                        F.mse_loss(predicted_values, returns[batch, i])
                    )
                value_loss = torch.stack(per_agent_value_losses).mean()

            total_loss = (
                actor_loss_sum
                + VF_COEF * value_loss
                - ENT_COEF * entropy_sum
            )

            for optimizer in actor_optimizers + critic_optimizers:
                optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for model in actors + critics for p in model.parameters()],
                max_norm=0.5,
            )
            for optimizer in actor_optimizers + critic_optimizers:
                optimizer.step()

            policy_losses.append(float(actor_loss_sum.detach()))
            value_losses.append(float(value_loss.detach()))
            entropies.append(float(entropy_sum.detach()))

    return {
        "policy_loss": sum(policy_losses) / max(1, len(policy_losses)),
        "value_loss": sum(value_losses) / max(1, len(value_losses)),
        "entropy": sum(entropies) / max(1, len(entropies)),
    }


def evaluate(actors, seed, n_episodes=EVAL_EPISODES):
    """Greedy evaluation on the same deterministic task layouts per seed."""
    env = GridCityEnv(seed=seed, grid=GRID, n_agents=N_AGENTS, n_tasks=N_TASKS)
    successes = 0
    delivered_total = 0
    returns = []
    lengths = []

    for episode in range(n_episodes):
        observations = env.reset(seed=seed + episode)
        episode_return = 0.0
        steps = 0

        while True:
            actions = []
            with torch.no_grad():
                for i, actor in enumerate(actors):
                    logits = actor(
                        torch.as_tensor(observations[i], dtype=torch.float32)
                    )
                    actions.append(int(logits.argmax().item()))

            observations, reward, done, info = env.step(actions)
            episode_return += reward
            steps += 1
            if done:
                break

        delivered = info["n_delivered"]
        delivered_total += delivered
        successes += int(delivered == N_TASKS)
        returns.append(episode_return)
        lengths.append(steps)

    return {
        "success_rate": successes / n_episodes,
        "mean_delivered": delivered_total / n_episodes,
        "mean_return": sum(returns) / n_episodes,
        "mean_steps": sum(lengths) / n_episodes,
    }


def train_one(mode, seed, verbose=True):
    random.seed(seed)
    torch.manual_seed(seed)

    env = GridCityEnv(
        seed=seed, grid=GRID, n_agents=N_AGENTS, n_tasks=N_TASKS
    )
    actors = [
        Actor(env.obs_dim, env.n_actions, hidden=HIDDEN)
        for _ in range(N_AGENTS)
    ]
    actor_optimizers = [
        torch.optim.Adam(actor.parameters(), lr=LR) for actor in actors
    ]

    joint_obs_dim = N_AGENTS * env.obs_dim
    n_critics = 1 if mode == "shared" else N_AGENTS
    critics = [
        CentralizedCritic(joint_obs_dim, hidden=HIDDEN)
        for _ in range(n_critics)
    ]
    critic_optimizers = [
        torch.optim.Adam(critic.parameters(), lr=LR) for critic in critics
    ]

    buffer = RolloutBuffer(
        T=T,
        n_agents=N_AGENTS,
        obs_dim=env.obs_dim,
        n_actions=env.n_actions,
        gamma=GAMMA,
        lam=LAM,
    )

    history = []
    start_time = time.time()

    for iteration in range(1, ITERATIONS + 1):
        buffer.reset()
        ep_returns, ep_deliveries, ep_lengths = collect_rollout(
            env, actors, buffer
        )
        losses = update(
            mode,
            actors,
            actor_optimizers,
            critics,
            critic_optimizers,
            buffer,
        )

        if verbose:
            print(
                f"  {mode:10s} it {iteration:3d} | "
                f"train return {sum(ep_returns)/len(ep_returns):7.2f} | "
                f"delivered {sum(ep_deliveries)/len(ep_deliveries):4.2f}/{N_TASKS} | "
                f"steps {sum(ep_lengths)/len(ep_lengths):5.1f} | "
                f"pi {losses['policy_loss']:7.3f} "
                f"v {losses['value_loss']:7.3f}"
            )

        if iteration % EVAL_INTERVAL == 0 or iteration == 1:
            metrics = evaluate(
                actors,
                seed=EVAL_SEED + seed * 10000,
                n_episodes=EVAL_EPISODES,
            )
            history.append(
                {
                    "mode": mode,
                    "seed": seed,
                    "iteration": iteration,
                    **metrics,
                }
            )
            if verbose:
                print(
                    f"      eval | success {metrics['success_rate']:5.1%} | "
                    f"delivered {metrics['mean_delivered']:.2f}/{N_TASKS} | "
                    f"return {metrics['mean_return']:.2f}"
                )

    elapsed = time.time() - start_time
    return actors, history, elapsed


def save_history(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    columns = [
        "mode",
        "seed",
        "iteration",
        "success_rate",
        "mean_delivered",
        "mean_return",
        "mean_steps",
    ]
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    print("=" * 78)
    print("TASK ALLOCATION: SHARED REWARD vs DIFFERENCE REWARD")
    print(
        f"Grid={GRID}x{GRID} | drones={N_AGENTS} | tasks={N_TASKS} | "
        f"PPO iterations={ITERATIONS}"
    )
    print("=" * 78)

    all_history = []
    final_results = []

    for mode in MODES:
        for seed in SEEDS:
            print(f"\n--- mode={mode}, seed={seed} ---")
            actors, history, elapsed = train_one(mode, seed)
            all_history.extend(history)
            final_metrics = evaluate(
                actors,
                seed=EVAL_SEED + seed * 10000,
                n_episodes=EVAL_EPISODES,
            )
            final_results.append(
                {"mode": mode, "seed": seed, **final_metrics}
            )
            print(
                f"FINAL {mode:10s} seed={seed} | "
                f"success={final_metrics['success_rate']:.1%} | "
                f"delivered={final_metrics['mean_delivered']:.2f}/{N_TASKS} | "
                f"return={final_metrics['mean_return']:.2f} | "
                f"time={elapsed:.1f}s"
            )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_history(all_history, os.path.join(RESULTS_DIR, "learning_curve.csv"))
    save_history(
        [
            {
                "mode": row["mode"],
                "seed": row["seed"],
                "iteration": ITERATIONS,
                "success_rate": row["success_rate"],
                "mean_delivered": row["mean_delivered"],
                "mean_return": row["mean_return"],
                "mean_steps": row["mean_steps"],
            }
            for row in final_results
        ],
        os.path.join(RESULTS_DIR, "final_results.csv"),
    )

    print("\n" + "=" * 78)
    print("FINAL MEAN PERFORMANCE")
    for mode in MODES:
        rows = [r for r in final_results if r["mode"] == mode]
        mean_success = sum(r["success_rate"] for r in rows) / len(rows)
        mean_delivered = sum(r["mean_delivered"] for r in rows) / len(rows)
        mean_return = sum(r["mean_return"] for r in rows) / len(rows)
        print(
            f"{mode:10s} | success={mean_success:.1%} | "
            f"delivered={mean_delivered:.2f}/{N_TASKS} | "
            f"return={mean_return:.2f}"
        )
    print(f"\nLearning curves and final metrics saved in: {RESULTS_DIR}/")
    print("=" * 78)


if __name__ == "__main__":
    main()
