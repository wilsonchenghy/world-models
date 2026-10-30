"""Super Mario Bros rollouts for the world model: biased-random play, frame skip 4 (15 fps), 2x downsampled to 120x128.

Each step stores the frame seen before the action. An action is held for 1-8 steps;
half the time it is right / right+A / right+A+B, otherwise uniform over SIMPLE_MOVEMENT (incl. NOOP and left).
Levels: world 1 (1-1 at 40%, 1-2..1-4 at 20% each). An episode (3 lives) lasts until game over, the flag, or MAX_STEPS.
Every 20th episode goes to eval/.

  python -m envs.mario.collect <out dir> <total frames> [workers]
"""
import multiprocessing as mp
import os
import sys
import warnings

import cv2
import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)  # old gym API notices
SKIP, MAX_STEPS = 4, 2000
STAGES, STAGE_P = ["1-1", "1-2", "1-3", "1-4"], [0.4, 0.2, 0.2, 0.2]
FORWARD = [1, 2, 4]  # right, right+A, right+A+B


def worker(args):
    wid, out, quota = args
    import gym_super_mario_bros
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
    from nes_py.wrappers import JoypadSpace

    rng = np.random.default_rng(wid)
    envs = {s: JoypadSpace(gym_super_mario_bros.make(f"SuperMarioBros-{s}-v0"), SIMPLE_MOVEMENT) for s in STAGES}
    done_frames, ep = 0, 0
    while done_frames < quota:
        stage = STAGES[rng.choice(len(STAGES), p=STAGE_P)]
        env = envs[stage]
        obs = env.reset()
        frames, actions, lives, xs = [], [], [], []
        hold, a, done, info = 0, 0, False, {"life": 2, "x_pos": 40}
        while not done and len(frames) < MAX_STEPS:
            if hold <= 0:
                a = int(rng.choice(FORWARD)) if rng.random() < 0.5 else int(rng.integers(len(SIMPLE_MOVEMENT)))
                hold = int(rng.integers(1, 9))
            hold -= 1
            frames.append(cv2.resize(obs, (128, 120), interpolation=cv2.INTER_AREA))
            actions.append(a); lives.append(info["life"]); xs.append(info["x_pos"])
            for _ in range(SKIP):
                obs, _, dying, info = env.step(a)
                done = info["life"] > 200 or info["flag_get"]  # game over (life wraps to 255) or the flag
                if done:
                    break
                if dying:  # the stage env flags every death as done; play on through the death and the respawn
                    env.unwrapped.done = False
        name = f"w{wid:02d}_e{ep:04d}_{stage}.npz"
        split = "eval" if ep % 20 == 19 else "train"
        np.savez_compressed(os.path.join(out, split, name), frames=np.stack(frames), actions=np.array(actions, np.int32),
                            life=np.array(lives, np.int8), x_pos=np.array(xs, np.int32), flag=bool(info["flag_get"]))
        done_frames += len(frames); ep += 1
    return wid, done_frames, ep


if __name__ == "__main__":
    out, total = sys.argv[1], int(sys.argv[2])
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 14
    for s in ("train", "eval"):
        os.makedirs(os.path.join(out, s), exist_ok=True)
    with mp.Pool(n) as pool:
        for wid, f, e in pool.imap_unordered(worker, [(w, out, total // n) for w in range(n)]):
            print(f"worker {wid}: {f} frames, {e} episodes", flush=True)
