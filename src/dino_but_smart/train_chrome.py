"""Fine-tune a clone-pretrained DQN against the real Chrome Dino game.

Designed for short runs (15-60 min). Loads an existing checkpoint, continues
training at a small learning rate and low epsilon so the policy polishes
rather than restarts. Saves to checkpoints/chrome-tuned-best.pt whenever a
new best episode reward is achieved, and to chrome-tuned-periodic.pt every
N seconds so a Selenium crash mid-run loses at most that much progress.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .agent import DQNAgent
from .chrome_env import ChromeEnv

CHECKPOINT_BEST = "chrome-tuned-best.pt"
CHECKPOINT_PERIODIC = "chrome-tuned-periodic.pt"
CHECKPOINT_FINAL = "chrome-tuned-final.pt"
SESSION_DEAD_MARKERS = (
    "no such window", "session deleted", "invalid session id",
    "chrome not reachable", "disconnected", "target closed",
    "session not created", "tab crashed",
)

EnvFactory = Callable[[], ChromeEnv]


@dataclass
class EpisodeResult:
    steps: int = 0
    reward: float = 0.0
    final_score: int = 0
    losses: list[float] = field(default_factory=list)
    error: str | None = None

    @property
    def mean_loss(self) -> float:
        return sum(self.losses) / len(self.losses) if self.losses else 0.0


def anneal(step: int, start: float, end: float, decay_steps: int) -> float:
    if step >= decay_steps:
        return end
    return start + (end - start) * (step / decay_steps)


def _is_session_dead(err_msg: str) -> bool:
    m = err_msg.lower()
    return any(s in m for s in SESSION_DEAD_MARKERS)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune DQN on real Chrome Dino")
    p.add_argument("--resume-from", type=str, required=True,
                   help="Path to clone-pretrained checkpoint (.pt)")
    p.add_argument("--duration-min", type=float, default=30.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--target-sync", type=int, default=200)
    p.add_argument("--eps-start", type=float, default=0.10)
    p.add_argument("--eps-end", type=float, default=0.02)
    p.add_argument("--eps-decay-steps", type=int, default=10_000)
    p.add_argument("--learn-every", type=int, default=4)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--buffer-capacity", type=int, default=20_000)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--checkpoint-every-sec", type=int, default=180)
    p.add_argument("--log-dir", type=str, default="logs")
    p.add_argument("--step-hz", type=float, default=30.0)
    p.add_argument("--url", type=str, default="https://chromedino.com/")
    return p.parse_args()


def _recreate_env(current: ChromeEnv, make_env: EnvFactory) -> ChromeEnv:
    try:
        current.close()
    except Exception as e:
        print(f"[recovery] current.close failed: {e}")
    last_err: Exception | None = None
    for attempt in range(5):
        try:
            fresh = make_env()
            print(f"[recovery] new browser session ready (attempt {attempt + 1})")
            return fresh
        except Exception as e:
            last_err = e
            print(f"[recovery] make_env failed ({e}); retrying in 10s")
            time.sleep(10)
    raise RuntimeError("could not recreate env after 5 attempts") from last_err


def _safe_reset(
        env: ChromeEnv, make_env: EnvFactory,
) -> tuple[ChromeEnv, np.ndarray | None]:
    """Reset env, recreating it on a dead session. Returns the (possibly new)
    env and the observation, or None if the caller should retry the loop."""
    try:
        return env, env.reset()
    except Exception as e:
        msg = str(e).splitlines()[0]
        if _is_session_dead(msg):
            print(f"[recovery] reset hit dead session: {msg}")
            return _recreate_env(env, make_env), None
        print(f"reset failed: {msg}; retrying in 5s")
        time.sleep(5)
        return env, None


def _run_episode(
        env: ChromeEnv, agent: DQNAgent, obs: np.ndarray,
        args: argparse.Namespace, global_step: int, deadline: float,
) -> tuple[EpisodeResult, int]:
    result = EpisodeResult()
    done = False
    while not done and time.time() < deadline:
        agent.epsilon = anneal(global_step, args.eps_start,
                               args.eps_end, args.eps_decay_steps)
        action = agent.act(obs, eval_mode=False)
        next_obs, reward, done, info = env.step(action)
        agent.remember(obs, action, reward, next_obs, done)
        obs = next_obs
        result.steps += 1
        result.reward += reward
        global_step += 1
        result.final_score = info.get("score", result.final_score)
        if "error" in info:
            result.error = info["error"]

        if global_step > args.warmup and global_step % args.learn_every == 0:
            loss = agent.learn_step()
            if loss is not None:
                result.losses.append(loss)
        if global_step % args.target_sync == 0:
            agent.sync_target()
    return result, global_step


def _log_episode(
        writer: Any, f: Any, ep: int, result: EpisodeResult,
        epsilon: float, elapsed: float,
) -> None:
    writer.writerow([
        ep, result.steps, f"{result.reward:.2f}", result.final_score,
        f"{epsilon:.3f}", f"{result.mean_loss:.4f}", f"{elapsed:.1f}",
    ])
    f.flush()
    print(f"ep {ep:3d}  steps {result.steps:4d}  score {result.final_score:4d}  "
          f"ep_reward {result.reward:7.2f}  eps {epsilon:.3f}  "
          f"loss {result.mean_loss:.4f}  elapsed {elapsed:.0f}s")


def main() -> None:
    args = _parse_args()
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_dir) / f"chrome-finetune-{int(time.time())}.csv"

    def make_env() -> ChromeEnv:
        return ChromeEnv(url=args.url, step_hz=args.step_hz, headless=False)

    env = make_env()
    agent = DQNAgent(seed=0, lr=args.lr,
                     batch_size=args.batch_size,
                     buffer_capacity=args.buffer_capacity)
    agent.load(args.resume_from)
    agent.epsilon = args.eps_start
    print(f"loaded {args.resume_from}; starting eps={agent.epsilon}, lr={args.lr}")

    global_step = 0
    best_ep_reward = -float("inf")
    last_periodic_save = time.time()
    t0 = time.time()
    deadline = t0 + args.duration_min * 60.0
    ep = 0

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode", "steps", "ep_reward", "game_score",
            "epsilon", "mean_loss", "elapsed_s",
        ])

        try:
            while time.time() < deadline:
                env, obs = _safe_reset(env, make_env)
                if obs is None:
                    continue

                result, global_step = _run_episode(
                    env, agent, obs, args, global_step, deadline,
                )

                # If the episode ended due to a dead session, swap in a fresh
                # browser before the next reset() instead of repeatedly poking
                # the corpse.
                if result.error and _is_session_dead(result.error):
                    print(f"[recovery] episode ended with dead session: {result.error}")
                    env = _recreate_env(env, make_env)

                _log_episode(writer, f, ep, result, agent.epsilon,
                             time.time() - t0)

                if result.reward > best_ep_reward:
                    best_ep_reward = result.reward
                    agent.save(os.path.join(args.checkpoint_dir, CHECKPOINT_BEST))

                if time.time() - last_periodic_save > args.checkpoint_every_sec:
                    agent.save(os.path.join(args.checkpoint_dir, CHECKPOINT_PERIODIC))
                    last_periodic_save = time.time()

                ep += 1
        except KeyboardInterrupt:
            print("interrupted by user — saving and exiting")
        except Exception:
            print("training crashed — saving and exiting")
            traceback.print_exc()
        finally:
            agent.save(os.path.join(args.checkpoint_dir, CHECKPOINT_FINAL))
            env.close()
            print(f"\ntotal episodes: {ep}, best ep reward: {best_ep_reward:.2f}")
            print(f"saved {CHECKPOINT_BEST} and {CHECKPOINT_FINAL}")


if __name__ == "__main__":
    main()
