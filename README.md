# dino-but-smart

A Deep Q-Network agent that learns to play Chrome's offline Dino game.

Training runs against a fast pygame clone; the same checkpoint then plays the
real Chrome Dino through a Selenium bridge that reads the game's internal state
from `window.Runner.instance_`.

## Measured performance

Numbers from the current `checkpoints/best.pt` (trained with the command shown
under [Train](#train)):

| metric | value |
| ------ | ----- |
| training time (6,000 episodes, CPU) | **3.5 min** |
| clone — mean steps survived (20 greedy episodes) | **3,282** |
| clone — mean shaped reward (20 greedy episodes) | **380** |
| real Chrome — mean game score (5 episodes) | **109** |
| real Chrome — mean steps survived (5 episodes) | **263** |
| real-game improvement after sim-to-real fix | **2.5×** |

The 2.5× improvement came from diagnosing a *canvas-coordinate sim-to-real gap*:
the pygame clone uses an 800×300 canvas with the ground line at y=250, while
Chrome's offline-Dino uses ~600×150 with the ground near y=93. The naïve bridge
divided Chrome's y values by 300, so `dino_y` when grounded normalised to ~0.29
in real Chrome vs ~0.83 in training — far enough out of distribution that the
agent defaulted to duck-spam. The bridge now queries Chrome's canvas geometry
once at `start_game()` and projects every observation into the clone's
coordinate system so the same checkpoint generalises across both environments.

## Quickstart

```bash
# 1. install (Python 3.10–3.13; pygame has no 3.14 wheels yet)
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 2. train (a few minutes on CPU)
python -m dino_but_smart.train --episodes 6000 --seed 0 --lr 5e-4 \
    --target-sync 500 --eps-decay 100000

# 3. watch the agent play the clone
python -m dino_but_smart.play --checkpoint checkpoints/best.pt

# 4. play the real Chrome Dino
python -m dino_but_smart.eval_real --checkpoint checkpoints/best.pt --episodes 5
```

### Train

The settings above (lr 5e-4, target_sync 500, eps_decay 100k) produced the
checkpoint behind the numbers in the [Measured performance](#measured-performance)
table. Defaults in `train.py` work but converge less reliably; vanilla DQN is
sensitive to these knobs.

## Project layout

- `src/dino_but_smart/env.py`       — pygame Dino clone, gym-like API
- `src/dino_but_smart/agent.py`     — Q-network, replay buffer, DQN agent
- `src/dino_but_smart/train.py`     — training loop / CLI
- `src/dino_but_smart/play.py`      — watch a checkpoint in the clone
- `src/dino_but_smart/bridge.py`    — Selenium adapter to chromedino.com
- `src/dino_but_smart/eval_real.py` — run a checkpoint on the real game
- `src/dino_but_smart/constants.py` — observation normalisation constants
  shared by env and bridge

## Requirements

- Python 3.10–3.13.
- Google Chrome installed (only needed for `eval_real.py`).
  `webdriver-manager` will fetch a matching `chromedriver` automatically.

## Tests

```bash
pytest
```

## How it plays the real Chrome Dino

`bridge.py` opens `https://chromedino.com/`, sends `SPACE` to start the game,
then 30 times a second:

1. Reads `window.Runner.instance_` via `driver.execute_script` to extract the
   dino's position, current speed, and obstacle list.
2. Projects Chrome's pixel-space values into the clone's coordinate system
   (the canvas dimensions and dino's ground-y are queried once at game start),
   so the trained agent sees the same observation distribution it learned on.
3. Feeds the projected 7-dim observation to the trained Q-network and picks
   the argmax action.
4. Sends the corresponding key event (`SPACE` for jump, `ARROW DOWN` for duck,
   nothing for noop) via Selenium `ActionChains`.

No screen capture, no OpenCV. The game exposes its own runtime — we just read
it directly.

## Known limitations

- Vanilla DQN; no Double-DQN / Dueling / Rainbow improvements yet. Convergence
  is noisy and best-episode performance can collapse mid-training.
- Real-Chrome eval still scores noticeably below clone performance, indicating
  residual sim-to-real gap beyond the coordinate fix (likely jump dynamics
  and obstacle-spacing distribution). See *Future work* below.

## Future work

- Switch the agent to **Double DQN** + soft target updates for stable
  convergence.
- Add a small **idle-action penalty** so the agent doesn't waste jumps/ducks
  on an empty field.
- Domain randomisation: vary clone canvas dimensions / jump physics during
  training so the policy is invariant to Chrome's exact constants.
- Pixel-based encoder (small CNN over stacked frames) as an alternative to
  the 7-dim feature observation.
