"""DQN agent components: Q-network, replay buffer, and the agent itself.
Training logic that uses these pieces lives in train.py."""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import N_ACTIONS, OBS_DIM


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int = OBS_DIM,
                 device: str | torch.device = "cpu"):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.device = torch.device(device)
        self._s = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._ns = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._a = np.zeros((capacity,), dtype=np.int64)
        self._r = np.zeros((capacity,), dtype=np.float32)
        self._d = np.zeros((capacity,), dtype=np.float32)
        self._size = 0
        self._idx = 0

    def push(self, s, a, r, ns, d) -> None:
        i = self._idx
        self._s[i] = s
        self._a[i] = a
        self._r[i] = r
        self._ns[i] = ns
        self._d[i] = float(d)
        self._idx = (self._idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self._size, size=batch_size)
        s = torch.from_numpy(self._s[idx]).to(self.device)
        a = torch.from_numpy(self._a[idx]).to(self.device)
        r = torch.from_numpy(self._r[idx]).to(self.device)
        ns = torch.from_numpy(self._ns[idx]).to(self.device)
        d = torch.from_numpy(self._d[idx]).to(self.device)
        return s, a, r, ns, d

    def __len__(self) -> int:
        return self._size


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, n_actions: int = N_ACTIONS,
                 hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.head(x)


class DQNAgent:
    def __init__(self, seed: int = 0, lr: float = 1e-3, gamma: float = 0.99,
                 batch_size: int = 64, buffer_capacity: int = 100_000,
                 device: str | torch.device = "cpu",
                 grad_clip: float = 10.0):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.device = torch.device(device)
        self.gamma = gamma
        self.batch_size = batch_size
        self.grad_clip = grad_clip
        self.epsilon = 1.0

        self.online_net = QNetwork().to(self.device)
        self.target_net = QNetwork().to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        for p in self.target_net.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(capacity=buffer_capacity, device=self.device)
        self._rng = random.Random(seed)

    def act(self, obs: np.ndarray, eval_mode: bool = False) -> int:
        if not eval_mode and self._rng.random() < self.epsilon:
            return self._rng.randint(0, N_ACTIONS - 1)
        with torch.no_grad():
            x = torch.from_numpy(
                np.asarray(obs, dtype=np.float32)
            ).unsqueeze(0).to(self.device)
            q = self.online_net(x)
            return int(q.argmax(dim=1).item())

    def remember(self, s, a, r, ns, d) -> None:
        self.buffer.push(s, a, r, ns, d)

    def learn_step(self) -> float | None:
        if len(self.buffer) < self.batch_size:
            return None
        s, a, r, ns, d = self.buffer.sample(self.batch_size)
        q_sa = self.online_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self.target_net(ns).max(dim=1).values
            target = r + self.gamma * q_next * (1.0 - d)
        loss = F.mse_loss(q_sa, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.grad_clip)
        self.optimizer.step()
        return float(loss.item())

    def sync_target(self) -> None:
        self.target_net.load_state_dict(self.online_net.state_dict())

    def save(self, path: str) -> None:
        torch.save({
            "online": self.online_net.state_dict(),
            "target": self.target_net.state_dict(),
            "epsilon": self.epsilon,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.online_net.load_state_dict(ckpt["online"])
        self.target_net.load_state_dict(ckpt["target"])
        self.epsilon = float(ckpt.get("epsilon", 0.0))
