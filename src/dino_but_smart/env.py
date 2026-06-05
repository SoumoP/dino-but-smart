"""pygame-based Dino clone with a gym-like API. No rendering yet — just
physics and game logic. Obstacles are added in Task 4."""

from __future__ import annotations

import os
import random

import numpy as np
import pygame

from .constants import (
    ACTION_DUCK, ACTION_JUMP, ACTION_NOOP, DINO_DUCK_H, DINO_H, DINO_W, DINO_X,
    GRAVITY, GROUND_Y, JUMP_V, MAX_SPEED, MAX_VY, N_ACTIONS, OBS_DIM,
    REWARD_CLEAR, REWARD_DEATH, REWARD_STEP, SCREEN_H, SCREEN_W, SPEED_ACCEL,
    V0_SPEED,
)

# Obstacle catalogue. y_top is the y-coordinate of the obstacle's top.
OBSTACLE_TYPES = [
    {"w": 17, "h": 35, "y_top": GROUND_Y - 35, "kind": "cactus_small"},
    {"w": 25, "h": 50, "y_top": GROUND_Y - 50, "kind": "cactus_large"},
    {"w": 51, "h": 35, "y_top": GROUND_Y - 35, "kind": "cactus_cluster"},
    {"w": 46, "h": 40, "y_top": GROUND_Y - 40, "kind": "bird_low"},
    {"w": 46, "h": 40, "y_top": GROUND_Y - 75, "kind": "bird_mid"},
    {"w": 46, "h": 40, "y_top": GROUND_Y - 110, "kind": "bird_high"},
]

MIN_GAP_PX = 200
MAX_GAP_PX = 500


class DinoEnv:
    def __init__(self, render: bool = False, seed: int = 0):
        self.render_mode = render
        self._rng = random.Random(seed)
        self._initialised_pygame = False
        self.reset()
        if self.render_mode:
            os.environ.setdefault("SDL_VIDEO_CENTERED", "1")
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
            pygame.display.set_caption("dino-but-smart")
            self.clock = pygame.time.Clock()
            self._initialised_pygame = True

    def reset(self) -> np.ndarray:
        self.dino_y = float(GROUND_Y)
        self.dino_vy = 0.0
        self.is_ducking = False
        self.speed = V0_SPEED
        self.steps = 0
        self.obstacles: list[dict] = []
        self.done = False
        self._next_gap = float(self._rng.randint(MIN_GAP_PX, MAX_GAP_PX))
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

        for obs in self.obstacles:
            obs["x"] -= self.speed

        bonus = 0.0
        for o in self.obstacles:
            if not o["cleared"] and (o["x"] + o["w"]) < DINO_X:
                o["cleared"] = True
                bonus += REWARD_CLEAR

        self.obstacles = [o for o in self.obstacles if o["x"] + o["w"] > 0]

        self._maybe_spawn()

        self.speed = min(MAX_SPEED, self.speed + SPEED_ACCEL)
        self.steps += 1

        if self._check_collision():
            self.done = True
            reward = REWARD_DEATH
        else:
            reward = REWARD_STEP + bonus

        obs = self._build_observation()
        if self.render_mode:
            self.render_frame()
        return obs, reward, self.done, {"score": self.steps}

    def _maybe_spawn(self) -> None:
        rightmost_x = max(
            (o["x"] + o["w"] for o in self.obstacles), default=-MAX_GAP_PX,
        )
        gap = SCREEN_W - rightmost_x
        if gap < self._next_gap:
            return
        proto = self._rng.choice(OBSTACLE_TYPES)
        self.obstacles.append({
            "x": float(SCREEN_W),
            "y": float(proto["y_top"]),
            "w": float(proto["w"]),
            "h": float(proto["h"]),
            "kind": proto["kind"],
            "cleared": False,
        })
        self._next_gap = float(self._rng.randint(MIN_GAP_PX, MAX_GAP_PX))

    def close(self) -> None:
        if self._initialised_pygame:
            pygame.quit()
            self._initialised_pygame = False

    def render_frame(self) -> None:
        if not self.render_mode:
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                raise SystemExit

        self.screen.fill((247, 247, 247))
        pygame.draw.line(self.screen, (83, 83, 83),
                         (0, GROUND_Y), (SCREEN_W, GROUND_Y), 2)

        dx, dy, dw, dh = self.dino_hitbox
        pygame.draw.rect(self.screen, (50, 50, 50),
                         pygame.Rect(dx, dy, dw, dh))

        for o in self.obstacles:
            colour = (60, 120, 60) if o["kind"].startswith("cactus") else (120, 60, 60)
            pygame.draw.rect(self.screen, colour,
                             pygame.Rect(o["x"], o["y"], o["w"], o["h"]))

        font = pygame.font.SysFont(None, 22)
        score_surf = font.render(f"score: {self.steps}", True, (83, 83, 83))
        self.screen.blit(score_surf, (SCREEN_W - 120, 10))

        pygame.display.flip()
        self.clock.tick(60)

    @property
    def dino_hitbox(self) -> tuple[float, float, float, float]:
        """Returns (x, y, w, h) AABB. y is top of hitbox."""
        h = DINO_DUCK_H if self.is_ducking else DINO_H
        return (DINO_X, self.dino_y - h, DINO_W, float(h))

    def _next_obstacle(self) -> dict | None:
        for o in self.obstacles:
            if o["x"] + o["w"] > DINO_X:
                return o
        return None

    def _check_collision(self) -> bool:
        dx, dy, dw, dh = self.dino_hitbox
        for o in self.obstacles:
            ox, oy, ow, oh = o["x"], o["y"], o["w"], o["h"]
            if dx < ox + ow and dx + dw > ox and dy < oy + oh and dy + dh > oy:
                return True
        return False

    def _build_observation(self) -> np.ndarray:
        nxt = self._next_obstacle()
        if nxt is None:
            dist, w, h, y = 1.0, 0.0, 0.0, 0.0
        else:
            dist = max(0.0, (nxt["x"] - DINO_X)) / SCREEN_W
            w = nxt["w"] / SCREEN_W
            h = nxt["h"] / SCREEN_H
            y = nxt["y"] / SCREEN_H
        return np.array([
            dist, w, h, y,
            self.dino_y / SCREEN_H,
            self.dino_vy / MAX_VY,
            self.speed / MAX_SPEED,
        ], dtype=np.float32)
