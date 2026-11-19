"""DQN with target network and replay buffer."""
import random
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_num_threads(1)   # tiny MLPs are faster single-threaded

class QNet(nn.Module):
    def __init__(self, obs_dim=5, n_actions=4, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, obs_dim=5, n_actions=4, lr=1e-3, gamma=0.95,
                 buffer_size=100000, batch_size=32, target_sync=200,
                 eps_start=1.0, eps_end=0.05, seed=0):
        self.rng          = random.Random(seed)
        torch.manual_seed(seed)
        self.n_actions    = n_actions
        self.gamma        = gamma
        self.batch_size   = batch_size
        self.target_sync  = target_sync
        self.eps_start    = eps_start
        self.eps_end      = eps_end
        self.buffer       = deque(maxlen=buffer_size)
        self.steps        = 0

        self.online = QNet(obs_dim, n_actions)
        self.target = QNet(obs_dim, n_actions)
        self.target.load_state_dict(self.online.state_dict())
        self.opt    = torch.optim.Adam(self.online.parameters(), lr=lr)

    def act(self, state, eps):
        if self.rng.random() < eps:
            return self.rng.randrange(self.n_actions)
        with torch.no_grad():
            q = self.online(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
        return int(q.argmax().item())

    def store(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, float(done)))

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return None
        batch = self.rng.sample(self.buffer, self.batch_size)
        s  = torch.tensor([b[0] for b in batch], dtype=torch.float32)
        a  = torch.tensor([b[1] for b in batch], dtype=torch.long)
        r  = torch.tensor([b[2] for b in batch], dtype=torch.float32)
        s2 = torch.tensor([b[3] for b in batch], dtype=torch.float32)
        d  = torch.tensor([b[4] for b in batch], dtype=torch.float32)

        q  = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            a_best = self.online(s2).argmax(1, keepdim=True)      # Double DQN selection
            q2     = self.target(s2).gather(1, a_best).squeeze(1) # target evaluation
            target = r + self.gamma * q2 * (1.0 - d)
        loss = F.mse_loss(q, target)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.steps += 1
        if self.steps % self.target_sync == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())