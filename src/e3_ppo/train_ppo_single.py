"""Experiment 3 — single-agent PPO on the discrete-action, corner-memory env.

Trains an ActorCritic on ContinuousDeliveryEnv(track_corners=True), logs
per-iteration metrics, and evaluates success rate / mean successful steps
the same way train_continuous.py does for DQN.

Run: python train_ppo_single.py
"""
import random
import time

import torch
import torch.nn.functional as F

from env_continuous import ContinuousDeliveryEnv, STEP_SIZE
from ppo import ActorCritic, RolloutBuffer

torch.set_num_threads(4)

# ---- rollout / optimisation hyperparameters ----
T          = 2048       # env steps per iteration
GAMMA      = 0.95
LAM        = 0.95
CLIP       = 0.2
K_EPOCHS   = 4
MINIBATCH  = 64
ENT_COEF   = 0.01
VF_COEF    = 0.5
LR         = 3e-4
HIDDEN     = 64
ITERATIONS = 100        # raise to 150–200 if the success curve is still rising
SEEDS      = [0, 1, 2]

EVAL_EPISODES = 300
EVAL_SEED     = 999


def collect_rollout(env, ac, buf):
    """Fill buf with complete episodes until a full episode no longer fits.
    Returns per-episode stats for logging."""
    ep_returns, ep_lengths, ep_successes = [], [], []
    while buf.ptr + env.max_steps <= buf.T:
        s = env.reset()
        ep_ret, ep_steps, done = 0.0, 0, False
        success = False
        while not done:
            with torch.no_grad():
                a, lp, v, _ = ac.act(torch.as_tensor(s, dtype=torch.float32))
            s2, r, done = env.step(int(a))
            buf.add(s, a, lp, r, done, v)
            ep_ret += r
            ep_steps += 1
            if r > 0:            # +10 goal reward outweighs the -1 step cost
                success = True
            s = s2
        ep_returns.append(ep_ret)
        ep_lengths.append(ep_steps)
        ep_successes.append(1.0 if success else 0.0)

    # Value of the state after the last stored transition. Every episode
    # terminates with done=True, so this is masked by (1-done) in GAE — but
    # passing it keeps the GAE loop uniform.
    with torch.no_grad():
        _, last_v = ac.forward(torch.as_tensor(s, dtype=torch.float32))
    buf.compute_gae(last_value=float(last_v))
    return ep_returns, ep_lengths, ep_successes


def update(ac, opt, buf):
    """K epochs of clipped-surrogate PPO over the filled buffer."""
    ptr        = buf.ptr
    obs        = buf.obs[:ptr].reshape(ptr, -1)
    actions    = buf.actions[:ptr].reshape(ptr)
    old_lp     = buf.logprobs[:ptr].reshape(ptr)
    adv        = buf.adv[:ptr]
    ret        = buf.ret[:ptr]
    # Advantage normalisation — standard; keeps the policy-gradient scale
    # stable across iterations.
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    pi_losses, v_losses, ents = [], [], []
    for _ in range(K_EPOCHS):
        perm = torch.randperm(ptr)
        for start in range(0, ptr, MINIBATCH):
            b = perm[start:start + MINIBATCH]
            lp, ent, v = ac.evaluate(obs[b], actions[b])
            ratio      = (lp - old_lp[b]).exp()
            surr1      = ratio * adv[b]
            surr2      = ratio.clamp(1.0 - CLIP, 1.0 + CLIP) * adv[b]
            pi_loss    = -torch.min(surr1, surr2).mean()
            v_loss     = F.mse_loss(v, ret[b])
            ent_loss   = -ent.mean()
            loss       = pi_loss + VF_COEF * v_loss + ENT_COEF * ent_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            pi_losses.append(pi_loss.item())
            v_losses.append(v_loss.item())
            ents.append(ent.mean().item())
    return (sum(pi_losses) / len(pi_losses),
            sum(v_losses) / len(v_losses),
            sum(ents) / len(ents))


def evaluate(env, ac, n_episodes=EVAL_EPISODES, seed=EVAL_SEED):
    """Greedy eval, same protocol as evaluate_single in train_continuous.py."""
    env.rng = random.Random(seed)
    reached, succ_steps = 0, []
    for _ in range(n_episodes):
        s = env.reset()
        n = 0
        while True:
            with torch.no_grad():
                logits, _ = ac.forward(torch.as_tensor(s, dtype=torch.float32))
            a = int(logits.argmax().item())
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


def train_one(seed, verbose=True):
    random.seed(seed)
    torch.manual_seed(seed)
    env = ContinuousDeliveryEnv(seed=seed, track_corners=True)
    ac  = ActorCritic(obs_dim=env.obs_dim, n_actions=4, hidden=HIDDEN)
    opt = torch.optim.Adam(ac.parameters(), lr=LR)
    buf = RolloutBuffer(T=T, n_agents=1, obs_dim=env.obs_dim,
                        gamma=GAMMA, lam=LAM)

    t0 = time.time()
    for it in range(1, ITERATIONS + 1):
        buf.reset()
        ep_returns, ep_lengths, ep_successes = collect_rollout(env, ac, buf)
        pi_l, v_l, ent = update(ac, opt, buf)
        if verbose:
            print(f"  it {it:3d} | eps {len(ep_returns):3d} | "
                  f"ret {sum(ep_returns)/len(ep_returns):7.2f} | "
                  f"succ {sum(ep_successes)/len(ep_successes):5.1%} | "
                  f"len {sum(ep_lengths)/len(ep_lengths):5.1f} | "
                  f"pi {pi_l:6.3f} v {v_l:6.3f} ent {ent:5.3f}")
    elapsed = time.time() - t0
    return env, ac, elapsed


def main():
    print("=" * 72)
    print("EXPERIMENT 3: single-agent PPO with corner memory")
    print(f"  T={T}  gamma={GAMMA}  lam={LAM}  clip={CLIP}  K={K_EPOCHS}  "
          f"mb={MINIBATCH}  iters={ITERATIONS}")
    print("=" * 72)
    print(f"{'seed':<5}{'success':>10}{'succ_steps':>12}{'train_s':>10}")
    print("-" * 37)

    results = {}
    for seed in SEEDS:
        print(f"\n--- seed {seed} ---")
        env, ac, elapsed = train_one(seed, verbose=True)
        m = evaluate(env, ac)
        results[seed] = m
        print(f"seed {seed}: success={m['success']:.1%}  "
              f"succ_steps={m['mean_successful_steps']:.2f}  "
              f"train_time={elapsed:.1f}s")

    mean_success = sum(r["success"] for r in results.values()) / len(SEEDS)
    print("\n" + "=" * 72)
    print(f"mean success across {len(SEEDS)} seeds: {mean_success:.1%}")
    print("=" * 72)


if __name__ == "__main__":
    main()