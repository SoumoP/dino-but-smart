"""Training entrypoint. Runs the DQN agent against DinoEnv (the pygame clone),
logs episode metrics to CSV, and saves checkpoints."""

from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from .agent import DQNAgent
from .env import DinoEnv

CHECKPOINT_BEST = "best.pt"
CHECKPOINT_FINAL = "final.pt"


@dataclass
class EpisodeResult:
    steps: int = 0
    score: float = 0.0
    losses: list[float] = field(default_factory=list)

    @property
    def mean_loss(self) -> float:
        return sum(self.losses) / len(self.losses) if self.losses else 0.0


def anneal_epsilon(step: int, eps_start: float, eps_end: float,
                   decay_steps: int) -> float:
    if step >= decay_steps:
        return eps_end
    frac = step / decay_steps
    return eps_start + (eps_end - eps_start) * frac


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DQN on Dino clone")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--learn-every", type=int, default=4)
    parser.add_argument("--target-sync", type=int, default=1000)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay", type=int, default=50_000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-capacity", type=int, default=100_000)
    return parser.parse_args()


def _run_episode(
        env: DinoEnv, agent: DQNAgent,
        args: argparse.Namespace, global_step: int,
) -> tuple[EpisodeResult, int]:
    obs = env.reset()
    result = EpisodeResult()
    done = False
    while not done:
        agent.epsilon = anneal_epsilon(
            global_step, args.eps_start, args.eps_end, args.eps_decay,
        )
        action = agent.act(obs, eval_mode=False)
        next_obs, reward, done, _ = env.step(action)
        agent.remember(obs, action, reward, next_obs, done)
        obs = next_obs
        result.steps += 1
        result.score += reward
        global_step += 1

        if global_step > args.warmup and global_step % args.learn_every == 0:
            loss = agent.learn_step()
            if loss is not None:
                result.losses.append(loss)
        if global_step % args.target_sync == 0:
            agent.sync_target()
    return result, global_step


def main() -> None:
    args = _parse_args()
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_dir) / f"train-{int(time.time())}.csv"

    env = DinoEnv(render=args.render, seed=args.seed)
    agent = DQNAgent(
        seed=args.seed, lr=args.lr, gamma=args.gamma,
        batch_size=args.batch_size, buffer_capacity=args.buffer_capacity,
    )

    global_step = 0
    best_score = -float("inf")

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "steps", "score", "epsilon", "mean_loss", "wall_time"])

        t0 = time.time()
        for ep in range(args.episodes):
            result, global_step = _run_episode(env, agent, args, global_step)
            writer.writerow([
                ep, result.steps, f"{result.score:.2f}",
                f"{agent.epsilon:.3f}", f"{result.mean_loss:.4f}",
                f"{time.time() - t0:.1f}",
            ])
            f.flush()
            print(f"ep {ep:4d}  steps {result.steps:4d}  score {result.score:7.2f}  "
                  f"eps {agent.epsilon:.3f}  loss {result.mean_loss:.4f}")

            if result.score > best_score:
                best_score = result.score
                agent.save(os.path.join(args.checkpoint_dir, CHECKPOINT_BEST))

        agent.save(os.path.join(args.checkpoint_dir, CHECKPOINT_FINAL))
        print(f"saved best ({best_score:.2f}) and final checkpoints")

    env.close()


if __name__ == "__main__":
    main()
