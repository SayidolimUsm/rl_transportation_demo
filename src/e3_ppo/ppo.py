"""PPO components for the low-altitude delivery prototype.

Single-agent:  ActorCritic (shared trunk, policy + value heads).
Multi-agent:   n separate Actors + one CentralizedCritic on concat(obs).
               This is the MAPPO-lite for Experiment 4.

The RolloutBuffer stores env steps with per-agent fields. GAE is computed on
the team reward (sum of per-agent rewards) and the resulting advantage is
shared by all agents — the cooperative objective. For n_agents=1 this reduces
exactly to standard PPO.

No training loop here; that lives in train_ppo_single.py / train_ppo_multi.py.
"""
import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    """Single-agent actor + critic with a shared trunk (hidden=64 default)."""
    def __init__(self, obs_dim, n_actions=4, hidden=64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.pi = nn.Linear(hidden, n_actions)
        self.v  = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.pi(h), self.v(h).squeeze(-1)

    def act(self, x):
        """x: (d,) or (B,d). Returns (a, logprob, value, entropy)."""
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        a = dist.sample()
        return a, dist.log_prob(a), value, dist.entropy()

    def evaluate(self, x, a):
        """x: (B,d), a: (B,). Returns (logprob, entropy, value)."""
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        return dist.log_prob(a), dist.entropy(), value


class Actor(nn.Module):
    """Policy only — one per agent in multi-agent (MAPPO-lite)."""
    def __init__(self, obs_dim, n_actions=4, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)

    def act(self, x):
        logits = self.forward(x)
        dist = Categorical(logits=logits)
        a = dist.sample()
        return a, dist.log_prob(a), dist.entropy()

    def evaluate(self, x, a):
        logits = self.forward(x)
        dist = Categorical(logits=logits)
        return dist.log_prob(a), dist.entropy()


class CentralizedCritic(nn.Module):
    """V(joint state). Input = concatenation of all agents' observations."""
    def __init__(self, joint_obs_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(joint_obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class RolloutBuffer:
    """On-policy buffer for T env steps, n_agents per step.

    Stores per-agent obs/actions/logprobs/rewards plus the shared done and
    the centralized-critic value at each step. GAE fills self.adv (T,) and
    self.ret (T,) from the team reward.
    """
    def __init__(self, T, n_agents, obs_dim, gamma=0.95, lam=0.95):
        self.T = T
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.gamma = gamma
        self.lam = lam
        self.reset()

    def reset(self):
        T, n, d = self.T, self.n_agents, self.obs_dim
        self.obs      = torch.zeros(T, n, d)
        self.actions  = torch.zeros(T, n, dtype=torch.long)
        self.logprobs = torch.zeros(T, n)
        self.rewards  = torch.zeros(T, n)
        self.dones    = torch.zeros(T)
        self.values   = torch.zeros(T)
        self.adv      = torch.zeros(T)
        self.ret      = torch.zeros(T)
        self.ptr = 0

    @staticmethod
    def _as_1d(x, n, dtype):
        t = torch.as_tensor(x, dtype=dtype)
        if t.dim() == 0:
            t = t.reshape(1)
        return t.reshape(n)

    def add(self, obs, actions, logprobs, rewards, done, value):
        """obs: (n,d) or (d,) for n=1. actions/logprobs/rewards: length n.
        done: shared scalar. value: scalar from centralized critic."""
        assert self.ptr < self.T, "buffer full"
        n = self.n_agents
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
        self.obs[self.ptr]      = obs_t
        self.actions[self.ptr]  = self._as_1d(actions, n, torch.long)
        self.logprobs[self.ptr] = self._as_1d(logprobs, n, torch.float32)
        self.rewards[self.ptr]  = self._as_1d(rewards, n, torch.float32)
        self.dones[self.ptr]    = float(done)
        self.values[self.ptr]   = float(value)
        self.ptr += 1

    def full(self):
        return self.ptr >= self.T

    def compute_gae(self, last_value=0.0):
        """Team-reward GAE. Fills self.adv and self.ret over [:ptr].

        team_r(t) = sum_i r_i(t).
        delta(t)  = team_r(t) + gamma * V(t+1) * (1 - done(t)) - V(t)
        adv(t)    = delta(t) + gamma * lam * (1 - done(t)) * adv(t+1)
        ret(t)    = adv(t) + V(t)

        last_value is V(s_T), the centralized value of the state after the
        last stored transition. If that transition was terminal, the (1-done)
        factor zeroes it, so passing V(s_T) is always safe.
        """
        T = self.ptr
        team_r = self.rewards[:T].sum(dim=1)
        adv = torch.zeros(T)
        last_adv = 0.0
        for t in reversed(range(T)):
            next_v = last_value if t == T - 1 else self.values[t + 1].item()
            nonterminal = 1.0 - self.dones[t].item()
            delta = (team_r[t].item()
                     + self.gamma * next_v * nonterminal
                     - self.values[t].item())
            last_adv = delta + self.gamma * self.lam * nonterminal * last_adv
            adv[t] = last_adv
        self.adv[:T] = adv
        self.ret[:T] = adv + self.values[:T]