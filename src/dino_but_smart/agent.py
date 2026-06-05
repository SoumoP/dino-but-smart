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
