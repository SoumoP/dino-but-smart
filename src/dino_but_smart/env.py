"""pygame-based Dino clone with a gym-like API. No rendering yet — just
physics and game logic. Obstacles are added in Task 4."""

from __future__ import annotations

import random

import numpy as np

from .constants import (
    ACTION_DUCK, ACTION_JUMP, ACTION_NOOP, DINO_DUCK_H, DINO_H, DINO_W, DINO_X,
    GRAVITY, GROUND_Y, JUMP_V, MAX_SPEED, MAX_VY, N_ACTIONS, OBS_DIM,
    REWARD_CLEAR, REWARD_DEATH, REWARD_STEP, SCREEN_H, SCREEN_W, SPEED_ACCEL,
    V0_SPEED,
)


class DinoEnv:
    def __init__(self, render: bool = False, seed: int = 0):
        self.render_mode = render
        self._rng = random.Random(seed)
        self._initialised_pygame = False  # rendering wired in Task 6
        self.reset()

    def reset(self) -> np.ndarray:
        self.dino_y = float(GROUND_Y)
        self.dino_vy = 0.0
        self.is_ducking = False
        self.speed = V0_SPEED
        self.steps = 0
        self.obstacles: list[dict] = []   # populated in Task 4
        self.done = False
        return self._build_observation()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        assert 0 <= action < N_ACTIONS, f"invalid action {action}"
        assert not self.done, "step called on a terminated episode"

        on_ground = self.dino_y >= GROUND_Y
        if action == ACTION_JUMP and on_ground:
            self.dino_vy = JUMP_V
            self.is_ducking = False
        elif action == ACTION_DUCK and on_ground:
            self.is_ducking = True
        else:
            if on_ground:
                self.is_ducking = False

        self.dino_vy += GRAVITY
        self.dino_vy = max(-MAX_VY, min(MAX_VY, self.dino_vy))
        self.dino_y += self.dino_vy
        if self.dino_y >= GROUND_Y:
            self.dino_y = float(GROUND_Y)
            self.dino_vy = 0.0

        self.speed = min(MAX_SPEED, self.speed + SPEED_ACCEL)
        self.steps += 1

        reward = REWARD_STEP
        obs = self._build_observation()
        return obs, reward, self.done, {"score": self.steps}

    def close(self) -> None:
        pass

    @property
    def dino_hitbox(self) -> tuple[float, float, float, float]:
        """Returns (x, y, w, h) AABB. y is top of hitbox."""
        h = DINO_DUCK_H if self.is_ducking else DINO_H
        return (DINO_X, self.dino_y - h, DINO_W, float(h))

    def _build_observation(self) -> np.ndarray:
        return np.array([
            1.0,
            0.0,
            0.0,
            0.0,
            self.dino_y / SCREEN_H,
            self.dino_vy / MAX_VY,
            self.speed / MAX_SPEED,
        ], dtype=np.float32)
