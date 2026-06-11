"""Constants shared between the pygame clone (env.py) and the Selenium bridge
(bridge.py). The contract is: the same checkpoint must work in both, so both
must build the 7-dim observation using the same normalisation."""

# Screen geometry — matched to Chrome offline-Dino's logical canvas
# so the same checkpoint plays both environments with no sim-to-real
# observation drift.
SCREEN_W = 600          # Chrome's r.dimensions.WIDTH
SCREEN_H = 150          # Chrome's r.dimensions.HEIGHT
GROUND_Y = 140          # bottom-of-dino at rest (Chrome's groundYPos 93 + DINO_H 47)

# Dino physics — mirrors Chrome's r.tRex.config + r.config
DINO_X = 50             # Chrome's tRex.config.START_X_POS
DINO_W = 44             # Chrome's tRex.config.WIDTH
DINO_H = 47             # Chrome's tRex.config.HEIGHT
DINO_DUCK_H = 25        # Chrome's tRex.config.HEIGHT_DUCK
GRAVITY = 0.6           # Chrome's r.config.GRAVITY
JUMP_V = -10.0          # Chrome's tRex.config.INIITAL_JUMP_VELOCITY (sic)
MAX_VY = 15.0           # tighter range for normalisation now that JUMP_V is smaller

# Game speed (horizontal scroll rate, pixels per step) — already matched
V0_SPEED = 6.0
MAX_SPEED = 13.0
SPEED_ACCEL = 0.001  # per env step

# Actions
ACTION_NOOP = 0
ACTION_JUMP = 1
ACTION_DUCK = 2
N_ACTIONS = 3

# Observation
OBS_DIM = 7

# Reward shaping
REWARD_STEP = 0.1
REWARD_CLEAR = 1.0
REWARD_DEATH = -10.0

# Small per-action costs so noop is strictly preferred in safe states.
JUMP_COST = 0.02
DUCK_COST = 0.02

# Chrome Dino displayed score = distanceRan * this factor
CHROME_SCORE_PER_DISTANCE = 0.025
