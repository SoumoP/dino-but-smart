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
    ACTION_DUCK, ACTION_JUMP, CHROME_SCORE_PER_DISTANCE, DINO_H, DINO_X,
    MAX_SPEED, MAX_VY, OBS_DIM, SCREEN_H, SCREEN_W,
)


JS_PULL_GEOMETRY = """
const r = window.Runner && window.Runner.instance_;
if (!r || !r.tRex || !r.dimensions) return null;
// r.dimensions is the LOGICAL game coordinate space (e.g. 600x150).
// The <canvas> element itself may be 2x larger for HiDPI; we want the
// logical dims because all obstacle/dino x,y are in logical coords.
return {
  canvas_w: r.dimensions.WIDTH,
  canvas_h: r.dimensions.HEIGHT,
  ground_y: r.tRex.groundYPos,
  dino_x: r.tRex.xPos
};
"""


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
        # Populated lazily on first observation via _query_geometry.
        self._chrome_geom: dict | None = None

    def start_game(self) -> None:
        body = self.driver.find_element(By.TAG_NAME, "body")
        body.click()
        ActionChains(self.driver).send_keys(Keys.SPACE).perform()
        for _ in range(50):
            time.sleep(0.1)
            state = self.pull_state()
            if state is not None and not state["crashed"]:
                self._query_geometry()
                return
        raise RuntimeError("could not start the dino game (Runner not ready)")

    def _query_geometry(self) -> None:
        """Read Chrome's logical canvas + dino geometry once."""
        geom = self.driver.execute_script(JS_PULL_GEOMETRY)
        if not geom:
            return
        self._chrome_geom = {
            "canvas_w": geom["canvas_w"],
            "canvas_h": geom["canvas_h"],
            "ground_y": geom["ground_y"],
            "dino_x": geom["dino_x"],
        }

    def send_action(self, action: int) -> None:
        if action == ACTION_JUMP:
            ActionChains(self.driver).send_keys(Keys.SPACE).perform()
        elif action == ACTION_DUCK:
            ActionChains(self.driver).key_down(Keys.ARROW_DOWN).perform()
            ActionChains(self.driver).key_up(Keys.ARROW_DOWN).perform()
        # ACTION_NOOP: do nothing

    def get_observation(self) -> np.ndarray:
        state = self.pull_state()
        if state is None:
            return np.zeros(OBS_DIM, dtype=np.float32)
        return self.state_to_obs(state)

    def is_crashed(self) -> bool:
        state = self.pull_state()
        return bool(state and state["crashed"])

    def get_score(self) -> int:
        state = self.pull_state()
        if not state:
            return 0
        return int(state["distanceRan"] * CHROME_SCORE_PER_DISTANCE)

    def state_to_obs(self, state: dict) -> np.ndarray:
        if self._chrome_geom is None:
            self._query_geometry()
        g = self._chrome_geom

        if g is None:
            cw, ch = float(SCREEN_W), float(SCREEN_H)
            chrome_dino_x = float(DINO_X)
        else:
            cw = float(g["canvas_w"])
            ch = float(g["canvas_h"])
            chrome_dino_x = float(g["dino_x"])

        obstacles = state.get("obstacles", [])
        nxt = None
        for o in obstacles:
            if o["x"] + o["w"] > chrome_dino_x:
                nxt = o
                break
        if nxt is None:
            dist, w, h, y = 1.0, 0.0, 0.0, 0.0
        else:
            dist = max(0.0, (nxt["x"] - chrome_dino_x)) / cw
            w = nxt["w"] / cw
            h = nxt["h"] / ch
            y = nxt["y"] / ch  # Chrome & clone both use top-of-obstacle here

        # Chrome's tRex.yPos is the TOP of the dino; clone's dino_y is the
        # BOTTOM. Convert top -> bottom by adding DINO_H so the agent sees the
        # same number it trained on when the dino is grounded.
        dino_y_bottom = state["tRexY"] + DINO_H
        dino_y = max(0.0, dino_y_bottom) / ch
        dino_vy = max(-MAX_VY, min(MAX_VY, state["tRexJumpVy"])) / MAX_VY
        speed = min(MAX_SPEED, state["speed"]) / MAX_SPEED

        return np.array([dist, w, h, y, dino_y, dino_vy, speed], dtype=np.float32)

    def close(self) -> None:
        try:
            self.driver.quit()
        except Exception as e:
            print(f"[bridge] driver.quit failed: {e}")

    def pull_state(self) -> dict | None:
        return self.driver.execute_script(JS_PULL_STATE)
