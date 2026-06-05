"""Load a trained checkpoint and watch the agent play the pygame clone."""

from __future__ import annotations

import argparse

from .agent import DQNAgent
from .env import DinoEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch a trained DQN agent")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = DinoEnv(render=True, seed=args.seed)
    agent = DQNAgent(seed=args.seed)
    agent.load(args.checkpoint)
    agent.epsilon = 0.0

    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        score = 0.0
        steps = 0
        while not done:
            action = agent.act(obs, eval_mode=True)
            obs, r, done, _ = env.step(action)
            score += r
            steps += 1
        print(f"ep {ep}  steps {steps}  score {score:.2f}")
    env.close()


if __name__ == "__main__":
    main()
