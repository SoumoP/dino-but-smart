# dino-but-smart

A Deep Q-Network agent that learns to play Chrome's offline Dino game.

Training runs against a fast pygame clone; the same checkpoint then plays the
real Chrome Dino through a Selenium bridge that reads the game's internal state
from `window.Runner.instance_`.

## Quickstart

```bash
# 1. install (Python 3.10–3.13; pygame has no 3.14 wheels yet)
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 2. train (a few minutes on CPU)
python -m dino_but_smart.train --episodes 2000

# 3. watch the agent play the clone
python -m dino_but_smart.play --checkpoint checkpoints/best.pt

# 4. play the real Chrome Dino
python -m dino_but_smart.eval_real --checkpoint checkpoints/best.pt --episodes 5
```

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
2. Builds the same 7-dim observation the agent was trained on (normalisation
   constants live in `constants.py`, shared with the training env).
3. Feeds the observation to the trained Q-network and picks the argmax action.
4. Sends the corresponding key event (`SPACE` for jump, `ARROW DOWN` for duck,
   nothing for noop) via Selenium `ActionChains`.

No screen capture, no OpenCV. The game exposes its own runtime — we just read
it directly.
