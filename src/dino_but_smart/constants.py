"""Constants shared between the pygame clone (env.py) and the Selenium bridge
(bridge.py). The contract is: the same checkpoint must work in both, so both
must build the 7-dim observation using the same normalisation."""

# Screen geometry (matches a typical Chrome Dino aspect ratio)
SCREEN_W = 800
SCREEN_H = 300
GROUND_Y = 250  # y-coordinate of the ground (where the dino's feet rest)

# Dino physics
DINO_X = 50
DINO_W = 44
DINO_H = 47
DINO_DUCK_H = 26
GRAVITY = 1.0
JUMP_V = -16.0  # initial upward velocity on jump (pygame y axis points down)
MAX_VY = 20.0   # max absolute vertical velocity for observation normalisation

# Game speed (horizontal scroll rate, pixels per step)
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
