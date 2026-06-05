"""Run a trained checkpoint against the real Chrome Dino game via the Selenium
bridge. Prints per-episode score and the mean."""

from __future__ import annotations

import argparse
import time

from .agent import DQNAgent
from .bridge import ChromeDinoBridge


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval trained DQN on real Chrome Dino")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--url", type=str, default="https://chromedino.com/")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--step-hz", type=float, default=30.0,
                        help="how many observation/action cycles per second")
    args = parser.parse_args()

    agent = DQNAgent(seed=0)
    agent.load(args.checkpoint)
    agent.epsilon = 0.0

    bridge = ChromeDinoBridge(url=args.url, headless=args.headless)
    step_dt = 1.0 / args.step_hz
    scores: list[int] = []

    try:
        for ep in range(args.episodes):
            bridge.start_game()
            while not bridge.is_crashed():
                obs = bridge.get_observation()
                action = agent.act(obs, eval_mode=True)
                bridge.send_action(action)
                time.sleep(step_dt)
            score = bridge.get_score()
            scores.append(score)
            print(f"ep {ep}  score {score}")
            time.sleep(1.0)
    finally:
        bridge.close()

    if scores:
        print(f"mean score over {len(scores)} eps: {sum(scores) / len(scores):.1f}")


if __name__ == "__main__":
    main()
