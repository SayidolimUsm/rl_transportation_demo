"""Neural-network and rollout-buffer components for multi-agent PPO.

The prototype uses:
- shared: one centralized value critic and the shared team reward;
- difference: one centralized value critic per drone and difference rewards.

JointActionCritic is included as a starting point for a later COMA extension.
"""
import torch
import torch.nn as nn
from torch.distributions import Categorical


class Actor(nn.Module):
    """One decentralized policy network for one drone."""

    def __init__(self, obs_dim, n_actions, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)

    def act(self, x):
        logits = self.forward(x)
        distribution = Categorical(logits=logits)
        action = distribution.sample()
        return (
            action,
            distribution.log_prob(action),
            distribution.entropy(),
            distribution.probs,
        )

    def evaluate(self, x, actions):
        distribution = Categorical(logits=self.forward(x))
        return distribution.log_prob(actions), distribution.entropy()


class CentralizedCritic(nn.Module):
    """V(joint observation)."""

    def __init__(self, joint_obs_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(joint_obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, joint_obs):
        return self.net(joint_obs).squeeze(-1)


class JointActionCritic(nn.Module):
    """Optional COMA critic: Q(s, joint_action) for every joint action."""

    def __init__(self, joint_obs_dim, n_agents, n_actions, hidden=64):
        super().__init__()
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.n_joint_actions = n_actions ** n_agents
        self.net = nn.Sequential(
            nn.Linear(joint_obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, self.n_joint_actions),
        )

    def forward(self, joint_obs):
        return self.net(joint_obs)

    def joint_index(self, actions):
        index = torch.zeros(actions.shape[0], dtype=torch.long)
        for i in range(self.n_agents):
            index = index * self.n_actions + actions[:, i]
        return index


class RolloutBuffer:
    """Stores on-policy transitions for all drones."""

    def __init__(
        self, T, n_agents, obs_dim, n_actions, gamma=0.95, lam=0.95
    ):
        self.T = T
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.lam = lam
        self.reset()

    def reset(self):
        T, n, d, n_actions = (
            self.T,
            self.n_agents,
            self.obs_dim,
            self.n_actions,
        )
        self.obs = torch.zeros(T, n, d)
        self.actions = torch.zeros(T, n, dtype=torch.long)
        self.logprobs = torch.zeros(T, n)
        self.all_probs = torch.zeros(T, n, n_actions)
        self.team_rewards = torch.zeros(T)
        self.diff_rewards = torch.zeros(T, n)
        self.per_alternative = torch.zeros(T, n, n_actions)
        self.dones = torch.zeros(T)
        self.adv = torch.zeros(T, n)
        self.ret = torch.zeros(T, n)
        self.ptr = 0

    @property
    def joint_obs(self):
        return self.obs[: self.ptr].reshape(self.ptr, -1)

    def add(
        self,
        per_agent_obs,
        actions,
        logprobs,
        all_probs,
        team_reward,
        per_alt,
        done,
    ):
        if self.ptr >= self.T:
            raise RuntimeError("Rollout buffer is full")

        self.obs[self.ptr] = torch.as_tensor(
            per_agent_obs, dtype=torch.float32
        )
        self.actions[self.ptr] = torch.as_tensor(
            actions, dtype=torch.long
        )
        self.logprobs[self.ptr] = torch.as_tensor(
            logprobs, dtype=torch.float32
        )
        if isinstance(all_probs, (list, tuple)) and all_probs and torch.is_tensor(all_probs[0]):
            probabilities_tensor = torch.stack(all_probs).to(dtype=torch.float32)
        else:
            probabilities_tensor = torch.as_tensor(all_probs, dtype=torch.float32)
        self.all_probs[self.ptr] = probabilities_tensor
        self.team_rewards[self.ptr] = float(team_reward)
        self.per_alternative[self.ptr] = torch.as_tensor(
            per_alt, dtype=torch.float32
        )
        self.dones[self.ptr] = float(done)
        self.ptr += 1

    def compute_difference_rewards(self):
        """D_i = r_team - E_{a_i~pi_i}[r_team(a_i, a_-i)].

        The expectation uses the policy probabilities saved during rollout;
        the other drones' sampled actions remain fixed in each counterfactual.
        """
        if self.ptr == 0:
            return self.diff_rewards[:0]

        expected_counterfactual = (
            self.all_probs[: self.ptr] * self.per_alternative[: self.ptr]
        ).sum(dim=-1)
        self.diff_rewards[: self.ptr] = (
            self.team_rewards[: self.ptr].unsqueeze(-1)
            - expected_counterfactual
        )
        return self.diff_rewards[: self.ptr]

    def _gae_one(self, rewards, values, last_value, dones):
        length = rewards.shape[0]
        advantages = torch.zeros(length)
        last_advantage = 0.0

        for t in reversed(range(length)):
            next_value = (
                float(last_value) if t == length - 1 else values[t + 1].item()
            )
            nonterminal = 1.0 - dones[t].item()
            delta = (
                rewards[t].item()
                + self.gamma * next_value * nonterminal
                - values[t].item()
            )
            last_advantage = (
                delta
                + self.gamma
                * self.lam
                * nonterminal
                * last_advantage
            )
            advantages[t] = last_advantage

        return advantages, advantages + values

    def compute_gae_shared(self, value_predictions, last_value=0.0):
        advantages, returns = self._gae_one(
            self.team_rewards[: self.ptr],
            value_predictions[: self.ptr],
            last_value,
            self.dones[: self.ptr],
        )
        self.adv[: self.ptr] = advantages.unsqueeze(-1).expand(
            -1, self.n_agents
        )
        self.ret[: self.ptr] = returns.unsqueeze(-1).expand(
            -1, self.n_agents
        )
        return self.adv[: self.ptr], self.ret[: self.ptr]

    def compute_gae_difference(
        self, value_predictions, last_values_per_agent
    ):
        if self.ptr:
            if value_predictions.shape != (self.ptr, self.n_agents):
                raise ValueError("value_predictions must have shape (ptr, n_agents)")
            if len(last_values_per_agent) != self.n_agents:
                raise ValueError("Need one bootstrap value per drone")

        for i in range(self.n_agents):
            advantages, returns = self._gae_one(
                self.diff_rewards[: self.ptr, i],
                value_predictions[: self.ptr, i],
                last_values_per_agent[i],
                self.dones[: self.ptr],
            )
            self.adv[: self.ptr, i] = advantages
            self.ret[: self.ptr, i] = returns

        return self.adv[: self.ptr], self.ret[: self.ptr]
