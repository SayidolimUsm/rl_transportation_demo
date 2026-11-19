"""Multi-agent PPO components for the task-allocation subproject.

Three policy-gradient modes share this file:

  shared      — one centralized V(s), MAPPO-style team advantage.
  difference  — per-agent V_i(s), GAE on the difference reward D_i.
  coma        — joint-action Q(s, ·), counterfactual advantage (optional).

RolloutBuffer stores everything needed for all three. The training script
decides which GAE/advantage method to call and which critics to instantiate.
"""
import torch
import torch.nn as nn
from torch.distributions import Categorical


# ---------------------------------------------------------------------- actors
class Actor(nn.Module):
    """Decentralized per-agent policy. Each agent has its own instance.

    With full observability + agent-ID one-hot, all actors see 24-dim obs.
    Separate instances (not shared weights) because agent ID already breaks
    symmetry; shared weights would force them to be identical by construction,
    which is a different (and stronger) assumption than we want for the
    credit-assignment comparison.
    """
    def __init__(self, obs_dim, n_actions, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)

    def act(self, x):
        """x: (d,). Returns (a, logprob, entropy, all_probs)."""
        logits = self.forward(x)
        dist = Categorical(logits=logits)
        a = dist.sample()
        return a, dist.log_prob(a), dist.entropy(), dist.probs

    def evaluate(self, x, a):
        """x: (B, d), a: (B,). Returns (logprob, entropy)."""
        logits = self.forward(x)
        dist = Categorical(logits=logits)
        return dist.log_prob(a), dist.entropy()


# ---------------------------------------------------------------------- critics
class CentralizedCritic(nn.Module):
    """V(joint obs). One instance for shared mode; n_agents instances for
    difference mode (each critic i learns V_i(s) = E[return_i | s])."""
    def __init__(self, joint_obs_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(joint_obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class JointActionCritic(nn.Module):
    """Q(s, ·) for all n_actions^n_agents joint actions (COMA mode).

    Matches the miguelsuau/PyTorch-CMA reference: input is the joint
    observation only, output is a vector over joint actions. The counterfactual
    baseline marginalizes over agent i's actions by reshaping this vector.
    """
    def __init__(self, joint_obs_dim, n_agents, n_actions, hidden=64):
        super().__init__()
        self.n_agents  = n_agents
        self.n_actions = n_actions
        self.n_joint   = n_actions ** n_agents
        self.net = nn.Sequential(
            nn.Linear(joint_obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, self.n_joint),
        )

    def forward(self, joint_obs):
        return self.net(joint_obs)      # (B, n_joint)

    def joint_index(self, actions):
        """actions: (B, n_agents) → (B,) index into the n_joint vector."""
        idx = torch.zeros(actions.shape[0], dtype=torch.long)
        for i in range(self.n_agents):
            idx = idx * self.n_actions + actions[:, i]
        return idx


# --------------------------------------------------------------------- buffer
class RolloutBuffer:
    """On-policy buffer storing per-agent obs/actions/logprobs/all-probs,
    team reward, per-alternative counterfactual reward table, and done.

    `joint_obs` is a reshaped view of `obs` — no separate storage.
    """
    def __init__(self, T, n_agents, obs_dim, n_actions, gamma=0.95, lam=0.95):
        self.T         = T
        self.n_agents  = n_agents
        self.obs_dim   = obs_dim
        self.n_actions = n_actions
        self.gamma     = gamma
        self.lam       = lam
        self.reset()

    def reset(self):
        T, n, d, A = self.T, self.n_agents, self.obs_dim, self.n_actions
        self.obs            = torch.zeros(T, n, d)
        self.actions        = torch.zeros(T, n, dtype=torch.long)
        self.logprobs       = torch.zeros(T, n)
        self.all_probs      = torch.zeros(T, n, A)         # full π_i(·|o_i) at rollout
        self.team_rewards   = torch.zeros(T)
        self.diff_rewards   = torch.zeros(T, n)            # filled later
        self.per_alternative= torch.zeros(T, n, A)         # counterfactual table
        self.dones          = torch.zeros(T)
        self.adv            = torch.zeros(T, n)
        self.ret            = torch.zeros(T, n)
        self.ptr            = 0

    @property
    def joint_obs(self):
        """(ptr, n*d) view over the filled portion of the buffer."""
        return self.obs[:self.ptr].reshape(self.ptr, -1)

    def full(self):
        return self.ptr >= self.T

    def add(self, per_agent_obs, actions, logprobs, all_probs,
            team_reward, per_alt, done):
        """All arguments are per-step, already detached tensors/lists.

        per_agent_obs: (n, d)     actions: (n,)
        logprobs: (n,)            all_probs: (n, A)
        team_reward: scalar       per_alt: (n, A)   done: scalar
        """
        assert self.ptr < self.T, "buffer full"
        self.obs[self.ptr]             = torch.as_tensor(per_agent_obs, dtype=torch.float32)
        self.actions[self.ptr]         = torch.as_tensor(actions, dtype=torch.long)
        self.logprobs[self.ptr]        = torch.as_tensor(logprobs, dtype=torch.float32)
        self.all_probs[self.ptr]       = torch.as_tensor(all_probs, dtype=torch.float32)
        self.team_rewards[self.ptr]    = float(team_reward)
        self.per_alternative[self.ptr] = torch.as_tensor(per_alt, dtype=torch.float32)
        self.dones[self.ptr]           = float(done)
        self.ptr += 1

    # ----------------------------------------------------- difference rewards
    def compute_difference_rewards(self):
        """D_i = team_r - Σ_a π_i(a|o_i) × per_alt[i][a].

        Uses the OLD policy distribution stored at rollout time. This is the
        standard difference-reward estimator when the counterfactual is
        expectation-based (vs. a single default action).
        """
        expected = (self.all_probs[:self.ptr] *
                    self.per_alternative[:self.ptr]).sum(dim=-1)   # (ptr, n)
        self.diff_rewards[:self.ptr] = self.team_rewards[:self.ptr].unsqueeze(-1) - expected
        return self.diff_rewards[:self.ptr]

    # -------------------------------------------------------------- GAE paths
    def _gae_one(self, rewards, values, last_value, dones):
        """Standard GAE for a 1-D reward stream. Returns (adv, ret) of shape (T,)."""
        T = rewards.shape[0]
        adv = torch.zeros(T)
        last_adv = 0.0
        for t in reversed(range(T)):
            next_v = last_value if t == T - 1 else values[t + 1].item()
            nonterminal = 1.0 - dones[t].item()
            delta = (rewards[t].item()
                     + self.gamma * next_v * nonterminal
                     - values[t].item())
            last_adv = delta + self.gamma * self.lam * nonterminal * last_adv
            adv[t] = last_adv
        return adv, adv + values

    def compute_gae_shared(self, v_values, last_value=0.0):
        """GAE on team reward. Fills self.adv and self.ret (both T×n) with
        the SAME team advantage in every column (standard MAPPO).
        v_values: (ptr,) from the single centralized critic."""
        ptr = self.ptr
        adv, ret = self._gae_one(self.team_rewards[:ptr], v_values[:ptr],
                                 last_value, self.dones[:ptr])
        self.adv[:ptr] = adv.unsqueeze(-1).expand(-1, self.n_agents)
        self.ret[:ptr] = ret.unsqueeze(-1).expand(-1, self.n_agents)
        return self.adv[:ptr], self.ret[:ptr]

    def compute_gae_difference(self, v_values_per_agent, last_values_per_agent):
        """GAE on each agent's difference reward. Fills self.adv/ret (T×n)
        with agent-specific advantages.
        v_values_per_agent: (ptr, n). last_values_per_agent: list of n floats."""
        ptr = self.ptr
        assert self.diff_rewards[:ptr].abs().sum() > 0 or ptr == 0, \
            "call compute_difference_rewards() first"
        for i in range(self.n_agents):
            adv, ret = self._gae_one(self.diff_rewards[:ptr, i],
                                     v_values_per_agent[:ptr, i],
                                     last_values_per_agent[i],
                                     self.dones[:ptr])
            self.adv[:ptr, i] = adv
            self.ret[:ptr, i] = ret
        return self.adv[:ptr], self.ret[:ptr]