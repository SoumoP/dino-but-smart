# dino-but-smart

A Deep Q-Network agent that learns to play Chrome's offline Dino game.

The training pipeline is two stages: (1) **clone-pretrain** on a fast pygame
recreation of the Dino game that mirrors Chrome's exact canvas geometry and
physics; (2) **fine-tune on the real Chrome game** through a Selenium bridge
that reads game state directly from `window.Runner.instance_`. The same agent
weights serve both environments.

## Measured performance

15-episode eval on real Chrome at chromedino.com:

| stage | mean score | median | max | mean steps survived |
| ----- | ---------- | ------ | --- | -------------------- |
| clone-pretrained only | 60.2 | 50 | 128 | 137 |
| **+ 30 min fine-tune on real Chrome** | **344.1** | **387** | **580** | **611** |
| improvement | **5.7×** | **7.7×** | **4.5×** | **4.5×** |

Action distribution after fine-tuning: 71% noop / 20% jump / 10% duck — a
sensible policy on the real game (vs. the duck-heavy 50% policy of the
clone-only checkpoint, which suffered from observation-distribution drift on
Chrome's actual physics).

## Pipeline at a glance

```
+----------------------+   ~2 min CPU    +---------------------+    30 min      +----------------------+
| Chrome-aligned       |---------------->| clone-pretrained.pt |--------------->| chrome-tuned-final.pt|
| pygame Dino clone    |  vanilla DQN    | (mean Chrome: 60)   |  fine-tune     | (mean Chrome: 344)   |
| (env.py, agent.py)   |                 |                     |  on real game  |                      |
+----------------------+                 +---------------------+                +----------------------+
                                                                                          ^
                                                                            +-------------+
                                                                            | Selenium bridge that
                                                                            | reads Runner.instance_
                                                                            | via injected JS         (bridge.py + chrome_env.py)
                                                                            +-------------------------+
```

## Quickstart

```bash
# 1. install (Python 3.10–3.13; pygame has no 3.14 wheels yet)
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 2. clone-pretrain (a couple of minutes on CPU)
python -m dino_but_smart.train --episodes 6000 --seed 0 --lr 5e-4 \
    --target-sync 500 --eps-decay 100000
# produces checkpoints/best.pt

# 3. fine-tune on the real Chrome game (~30 minutes, a Chrome window opens)
cp checkpoints/best.pt checkpoints/clone-pretrained.pt
python -m dino_but_smart.train_chrome \
    --resume-from checkpoints/clone-pretrained.pt \
    --duration-min 30
# produces checkpoints/chrome-tuned-final.pt

# 4. eval against real Chrome (5 episodes)
python -m dino_but_smart.eval_real --checkpoint checkpoints/chrome-tuned-final.pt \
    --episodes 5

# (optional) watch the agent play the clone
python -m dino_but_smart.play --checkpoint checkpoints/chrome-tuned-final.pt
```

## Sim-to-real engineering: what made the gap close

The interesting parts of this project aren't the DQN code — that's standard.
The interesting parts are the bugs we found and fixed when getting the same
checkpoint to play both the clone and real Chrome:

1. **Canvas coordinate alignment.** Originally the clone used 800×300 while
   Chrome's logical canvas is 600×150 — the dino's normalised y was 0.83 in
   training but 0.29 on Chrome, far enough out of distribution that the
   trained policy collapsed to duck-spam. Fix: mirror Chrome's exact
   dimensions in `constants.py`.
2. **HiDPI canvas.** Chrome's `<canvas>.width` reports the device-pixel size
   (1200), but the game's coordinates live in the logical space (600). The
   bridge was dividing logical-x by device-w, halving every distance reading.
   Fix: query `Runner.dimensions.WIDTH`, not the DOM element.
3. **Top-vs-bottom dino-y semantic.** Chrome reports `tRex.yPos` as the top
   of the dino hitbox; the clone tracks the bottom. Both normalise by canvas
   height, so identical "grounded" states produced different normalised
   values. Fix: bridge adds `DINO_H` before normalising.
4. **Sim-to-real overfitting residual.** Even after the three coordinate
   fixes, mean Chrome score sat at 60 — the clone's obstacle spawn
   distribution and physics timing didn't perfectly match the real game.
   Fix: 30 min of direct fine-tuning on the real Chrome game via Selenium,
   resuming from the clone-pretrained checkpoint with low lr (1e-4) and low
   ε (0.10→0.02) so the policy polishes without forgetting clone behaviour.

## Project layout

- `src/dino_but_smart/env.py`        — pygame Dino clone, gym-like API
- `src/dino_but_smart/agent.py`      — Q-network, replay buffer, DQN agent
- `src/dino_but_smart/train.py`      — clone-pretrain CLI
- `src/dino_but_smart/bridge.py`     — Selenium adapter to chromedino.com
- `src/dino_but_smart/chrome_env.py` — gym-like wrapper around the bridge
- `src/dino_but_smart/train_chrome.py` — fine-tune CLI with crash-safe
  checkpointing and Selenium-session recovery
- `src/dino_but_smart/play.py`       — watch a checkpoint in the clone
- `src/dino_but_smart/eval_real.py`  — run a checkpoint on the real game
- `src/dino_but_smart/constants.py`  — Chrome-aligned geometry constants
  shared by env and bridge

## Requirements

- Python 3.10–3.13.
- Google Chrome installed (only needed for `eval_real.py` and
  `train_chrome.py`). `webdriver-manager` fetches the matching `chromedriver`
  automatically on first run.

## Tests

```bash
pytest
```

20 smoke + behaviour tests covering the env, replay buffer, Q-network,
agent, and save/load round-trip. Selenium-backed code is covered by manual
smoke tests rather than CI (would need a real browser).

## Known limitations / future work

- Vanilla DQN. Double DQN was implemented and tested but didn't improve
  scores at the budgets we tried (3-action space → small overestimation bias;
  DDQN's slower convergence didn't pay off in ~6k training episodes).
- 30-min Chrome fine-tune is the headline number; longer fine-tuning likely
  pushes mean score past 400.
- Reward shaping on Chrome is survival-only (clone has a +1/obstacle-cleared
  bonus). Adding an equivalent "made forward progress" signal on Chrome
  would likely accelerate fine-tuning.
- No pixel-based agent. The 7-dim feature observation is handed to the
  network rather than learned; a CNN over stacked frames would be the next
  ambition.

## How the Selenium bridge actually plays the real Chrome Dino

`bridge.py` opens `https://chromedino.com/`, sends `SPACE` to start, then
30 times a second:

1. Reads `window.Runner.instance_` via `driver.execute_script` to extract
   dino position, current speed, and the obstacle list.
2. Builds the 7-dim observation **in the clone's coordinate system** by
   projecting Chrome's values (logical canvas dims, top-of-dino y) into
   the equivalent clone numbers (bottom-of-dino y, same normalisation).
3. Feeds the projected observation to the trained Q-network and picks the
   argmax action.
4. Sends the corresponding key event (`SPACE` for jump, `ARROW DOWN` for
   duck, nothing for noop) via Selenium `ActionChains`.

No pixel scraping, no OpenCV. The game exposes its own runtime on
`window.Runner.instance_` — we just read it.
