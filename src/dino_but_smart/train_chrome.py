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
from pathlib import Path

from .agent import DQNAgent
from .chrome_env import ChromeEnv


def anneal(step: int, start: float, end: float, decay_steps: int) -> float:
    if step >= decay_steps:
        return end
    return start + (end - start) * (step / decay_steps)


def main() -> None:
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
    args = p.parse_args()

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

    SESSION_DEAD_MARKERS = (
        "no such window", "session deleted", "invalid session id",
        "chrome not reachable", "disconnected", "target closed",
        "session not created", "tab crashed",
    )

    def is_session_dead(err_msg: str) -> bool:
        m = err_msg.lower()
        return any(s in m for s in SESSION_DEAD_MARKERS)

    def recreate_env(current: ChromeEnv) -> ChromeEnv:
        try:
            current.close()
        except Exception:
            pass
        for attempt in range(5):
            try:
                fresh = make_env()
                print(f"[recovery] new browser session ready (attempt {attempt+1})")
                return fresh
            except Exception as e:
                print(f"[recovery] make_env failed ({e}); retrying in 10s")
                time.sleep(10)
        raise RuntimeError("could not recreate env after 5 attempts")

    global_step = 0
    best_ep_reward = -float("inf")
    last_periodic_save = time.time()
    t0 = time.time()
    deadline = t0 + args.duration_min * 60.0

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode", "steps", "ep_reward", "game_score",
            "epsilon", "mean_loss", "elapsed_s",
        ])

        ep = 0
        try:
            while time.time() < deadline:
                try:
                    obs = env.reset()
                except Exception as e:
                    msg = str(e).splitlines()[0]
                    if is_session_dead(msg):
                        print(f"[recovery] reset hit dead session: {msg}")
                        env = recreate_env(env)
                        continue
                    print(f"reset failed: {msg}; retrying in 5s")
                    time.sleep(5)
                    continue

                done = False
                ep_steps = 0
                ep_reward = 0.0
                final_score = 0
                losses: list[float] = []
                ep_error = None

                while not done and time.time() < deadline:
                    agent.epsilon = anneal(global_step, args.eps_start,
                                           args.eps_end, args.eps_decay_steps)
                    action = agent.act(obs, eval_mode=False)
                    next_obs, reward, done, info = env.step(action)
                    agent.remember(obs, action, reward, next_obs, done)
                    obs = next_obs
                    ep_steps += 1
                    ep_reward += reward
                    global_step += 1
                    if "score" in info:
                        final_score = info["score"]
                    if "error" in info:
                        ep_error = info["error"]

                    if (global_step > args.warmup
                            and global_step % args.learn_every == 0):
                        loss = agent.learn_step()
                        if loss is not None:
                            losses.append(loss)
                    if global_step % args.target_sync == 0:
                        agent.sync_target()

                # If the episode ended due to a dead session, swap in a fresh
                # browser before the next reset() instead of repeatedly poking
                # the corpse.
                if ep_error and is_session_dead(ep_error):
                    print(f"[recovery] episode ended with dead session: {ep_error}")
                    env = recreate_env(env)

                mean_loss = sum(losses) / len(losses) if losses else 0.0
                elapsed = time.time() - t0
                writer.writerow([
                    ep, ep_steps, f"{ep_reward:.2f}", final_score,
                    f"{agent.epsilon:.3f}", f"{mean_loss:.4f}", f"{elapsed:.1f}",
                ])
                f.flush()
                print(f"ep {ep:3d}  steps {ep_steps:4d}  score {final_score:4d}  "
                      f"ep_reward {ep_reward:7.2f}  eps {agent.epsilon:.3f}  "
                      f"loss {mean_loss:.4f}  elapsed {elapsed:.0f}s")

                if ep_reward > best_ep_reward:
                    best_ep_reward = ep_reward
                    agent.save(os.path.join(args.checkpoint_dir,
                                            "chrome-tuned-best.pt"))

                if time.time() - last_periodic_save > args.checkpoint_every_sec:
                    agent.save(os.path.join(args.checkpoint_dir,
                                            "chrome-tuned-periodic.pt"))
                    last_periodic_save = time.time()

                ep += 1
        except KeyboardInterrupt:
            print("interrupted by user — saving and exiting")
        except Exception:
            print("training crashed — saving and exiting")
            traceback.print_exc()
        finally:
            agent.save(os.path.join(args.checkpoint_dir,
                                    "chrome-tuned-final.pt"))
            env.close()
            print(f"\ntotal episodes: {ep}, best ep reward: {best_ep_reward:.2f}")
            print(f"saved chrome-tuned-best.pt and chrome-tuned-final.pt")


if __name__ == "__main__":
    main()
