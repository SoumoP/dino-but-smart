"""Gym-like environment wrapper around the real Chrome Dino game via Selenium.

Designed for fine-tuning a clone-pretrained DQN against the real game so the
policy adapts to Chrome's exact obstacle distribution + timing without
retraining from scratch (which would take hours at 30 Hz).
"""

from __future__ import annotations

import time

import numpy as np

from .bridge import ChromeDinoBridge
from .constants import OBS_DIM, REWARD_DEATH, REWARD_STEP


class ChromeEnv:
    """Gym-like API around ChromeDinoBridge.

    Reward shaping is intentionally simple — survival + death — because the
    agent is being fine-tuned, not trained from scratch. Tracking 'obstacle
    cleared' events without a stable obstacle id from Chrome's runtime would
    be brittle, and the clone-pretrained policy already knows roughly when
    to act.
    """

    def __init__(self,
                 url: str = "https://chromedino.com/",
                 step_hz: float = 30.0,
                 headless: bool = False,
                 post_crash_sleep_s: float = 0.6):
        self.bridge = ChromeDinoBridge(url=url, headless=headless)
        self.step_dt = 1.0 / step_hz
        self.post_crash_sleep_s = post_crash_sleep_s

    def reset(self) -> np.ndarray:
        # Wait for the game-over screen to clear before restarting, otherwise
        # SPACE may not register or may double-trigger.
        time.sleep(self.post_crash_sleep_s)
        self.bridge.start_game()
        # Let the game tick a couple frames so the first observation is
        # representative (speed has settled etc.)
        time.sleep(0.1)
        return self.bridge.get_observation()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        self.bridge.send_action(action)
        time.sleep(self.step_dt)
        state = self.bridge._pull_state()
        if state is None:
            # Page/driver lost — terminate without reward so the training
            # loop can attempt to recover via reset().
            return (np.zeros(OBS_DIM, dtype=np.float32), 0.0, True,
                    {"error": "no_state"})

        if state["crashed"]:
            return (np.zeros(OBS_DIM, dtype=np.float32), REWARD_DEATH, True,
                    {"score": int(state["distanceRan"] * 0.025)})

        obs = self.bridge._state_to_obs(state)
        return obs, REWARD_STEP, False, {
            "score": int(state["distanceRan"] * 0.025),
        }

    def close(self) -> None:
        self.bridge.close()
