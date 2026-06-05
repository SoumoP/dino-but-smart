"""Selenium adapter that drives the real Chrome Dino game. Game state is read
from window.Runner.instance_ via injected JS — same 7-dim observation as
DinoEnv so the same checkpoint plays both."""

from __future__ import annotations

import time

import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

from .constants import (
    ACTION_DUCK, ACTION_JUMP, ACTION_NOOP, DINO_X, MAX_SPEED, MAX_VY, OBS_DIM,
    SCREEN_H, SCREEN_W,
)


JS_PULL_STATE = """
const r = window.Runner && window.Runner.instance_;
if (!r) return null;
return {
  crashed: !!r.crashed,
  speed: r.currentSpeed || 0,
  distanceRan: r.distanceRan || 0,
  tRexY: r.tRex ? r.tRex.yPos : 0,
  tRexJumpVy: r.tRex ? (r.tRex.jumpVelocity || 0) : 0,
  obstacles: (r.horizon && r.horizon.obstacles ? r.horizon.obstacles : []).map(o => ({
    x: o.xPos, y: o.yPos, w: o.width,
    h: (o.typeConfig && o.typeConfig.height) ? o.typeConfig.height : 30,
    type: (o.typeConfig && o.typeConfig.type) ? o.typeConfig.type : 'unknown'
  }))
};
"""


class ChromeDinoBridge:
    def __init__(self, url: str = "https://chromedino.com/", headless: bool = False):
        opts = Options()
        if headless:
            opts.add_argument("--headless=new")
        opts.add_argument("--disable-infobars")
        opts.add_argument("--mute-audio")
        opts.add_argument("--window-size=1024,400")
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=opts)
        self.driver.get(url)
        WebDriverWait(self.driver, 15).until(
            EC.presence_of_element_located((By.TAG_NAME, "canvas"))
        )

    def start_game(self) -> None:
        body = self.driver.find_element(By.TAG_NAME, "body")
        body.click()
        ActionChains(self.driver).send_keys(Keys.SPACE).perform()
        for _ in range(50):
            time.sleep(0.1)
            state = self._pull_state()
            if state is not None and not state["crashed"]:
                return
        raise RuntimeError("could not start the dino game (Runner not ready)")

    def send_action(self, action: int) -> None:
        if action == ACTION_JUMP:
            ActionChains(self.driver).send_keys(Keys.SPACE).perform()
        elif action == ACTION_DUCK:
            ActionChains(self.driver).key_down(Keys.ARROW_DOWN).perform()
            ActionChains(self.driver).key_up(Keys.ARROW_DOWN).perform()
        # ACTION_NOOP: do nothing

    def get_observation(self) -> np.ndarray:
        state = self._pull_state()
        if state is None:
            return np.zeros(OBS_DIM, dtype=np.float32)
        return self._state_to_obs(state)

    def is_crashed(self) -> bool:
        state = self._pull_state()
        return bool(state and state["crashed"])

    def get_score(self) -> int:
        state = self._pull_state()
        if not state:
            return 0
        # Chrome Dino: displayed score ~ distanceRan * 0.025
        return int(state["distanceRan"] * 0.025)

    def _state_to_obs(self, state: dict) -> np.ndarray:
        obstacles = state.get("obstacles", [])
        nxt = None
        for o in obstacles:
            if o["x"] + o["w"] > DINO_X:
                nxt = o
                break
        if nxt is None:
            dist, w, h, y = 1.0, 0.0, 0.0, 0.0
        else:
            dist = max(0.0, (nxt["x"] - DINO_X)) / SCREEN_W
            w = nxt["w"] / SCREEN_W
            h = nxt["h"] / SCREEN_H
            y = nxt["y"] / SCREEN_H

        dino_y = max(0.0, min(SCREEN_H, state["tRexY"])) / SCREEN_H
        dino_vy = max(-MAX_VY, min(MAX_VY, state["tRexJumpVy"])) / MAX_VY
        speed = min(MAX_SPEED, state["speed"]) / MAX_SPEED

        return np.array([dist, w, h, y, dino_y, dino_vy, speed], dtype=np.float32)

    def close(self) -> None:
        try:
            self.driver.quit()
        except Exception:
            pass

    def _pull_state(self) -> dict | None:
        return self.driver.execute_script(JS_PULL_STATE)
