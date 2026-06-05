import numpy as np
import torch

from dino_but_smart.agent import QNetwork, ReplayBuffer
from dino_but_smart.constants import N_ACTIONS, OBS_DIM


def test_buffer_len_grows_until_capacity():
    buf = ReplayBuffer(capacity=5, obs_dim=OBS_DIM)
    assert len(buf) == 0
    for i in range(3):
        buf.push(np.zeros(OBS_DIM), 0, 0.0, np.zeros(OBS_DIM), False)
    assert len(buf) == 3
    for i in range(10):
        buf.push(np.zeros(OBS_DIM), 0, 0.0, np.zeros(OBS_DIM), False)
    assert len(buf) == 5, "buffer should cap at capacity"


def test_buffer_sample_shapes_and_dtypes():
    buf = ReplayBuffer(capacity=100, obs_dim=OBS_DIM)
    for i in range(50):
        s = np.random.randn(OBS_DIM).astype(np.float32)
        ns = np.random.randn(OBS_DIM).astype(np.float32)
        buf.push(s, i % 3, float(i), ns, bool(i % 2))
    s, a, r, ns, d = buf.sample(16)
    assert s.shape == (16, OBS_DIM)
    assert a.shape == (16,)
    assert r.shape == (16,)
    assert ns.shape == (16, OBS_DIM)
    assert d.shape == (16,)
    assert s.dtype == torch.float32
    assert a.dtype == torch.long
    assert r.dtype == torch.float32
    assert d.dtype == torch.float32


def test_qnet_forward_returns_batch_q_values():
    net = QNetwork()
    x = torch.zeros((4, OBS_DIM))
    out = net(x)
    assert out.shape == (4, N_ACTIONS)


def test_qnet_eval_is_deterministic_for_same_input():
    net = QNetwork().eval()
    x = torch.randn((2, OBS_DIM))
    out1 = net(x)
    out2 = net(x)
    assert torch.allclose(out1, out2)
