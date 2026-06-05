import numpy as np
import pytest

from dino_but_smart.env import DinoEnv
from dino_but_smart.constants import (
    ACTION_NOOP, ACTION_JUMP, ACTION_DUCK, OBS_DIM, GROUND_Y,
)


def test_reset_returns_obs_with_correct_shape_and_dtype():
    env = DinoEnv(render=False, seed=0)
    obs = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    env.close()


def test_dino_starts_on_ground():
    env = DinoEnv(render=False, seed=0)
    env.reset()
    assert env.dino_y == GROUND_Y
    assert env.dino_vy == 0.0
    env.close()


def test_jump_action_lifts_dino_off_ground():
    env = DinoEnv(render=False, seed=0)
    env.reset()
    env.step(ACTION_JUMP)
    assert env.dino_y < GROUND_Y, "dino should have risen above ground after jumping"
    env.close()


def test_gravity_brings_dino_back_down():
    env = DinoEnv(render=False, seed=0)
    env.reset()
    env.step(ACTION_JUMP)
    for _ in range(60):  # plenty of steps to land
        env.step(ACTION_NOOP)
    assert env.dino_y == GROUND_Y, "dino should be back on the ground"
    env.close()


def test_duck_only_active_on_ground():
    env = DinoEnv(render=False, seed=0)
    env.reset()
    env.step(ACTION_DUCK)
    assert env.is_ducking is True
    env.close()
