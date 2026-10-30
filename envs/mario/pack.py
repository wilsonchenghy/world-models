"""Pack the collected episodes into flat arrays for training: frames (N, 120, 128, 3) uint8 as a .npy memmap, plus
actions (N,) int8 and episode ids (N,) int32, per split.

  python -m envs.mario.pack <raw dir> <out dir>
"""
import sys
from pathlib import Path

import numpy as np

raw, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
for split in ("train", "eval"):
    files = sorted((raw / split).glob("*.npz"))
    lengths = [np.load(f)["actions"].shape[0] for f in files]
    n = sum(lengths)
    frames = np.lib.format.open_memmap(out / f"{split}_frames.npy", mode="w+", dtype=np.uint8, shape=(n, 120, 128, 3))
    actions, episode, stage = np.zeros(n, np.int8), np.zeros(n, np.int32), []
    i = 0
    for k, (f, L) in enumerate(zip(files, lengths)):
        d = np.load(f)
        frames[i:i + L] = d["frames"]
        actions[i:i + L] = d["actions"]
        episode[i:i + L] = k
        stage.append(f.stem.split("_")[-1])
        i += L
    frames.flush()
    np.savez(out / f"{split}_meta.npz", actions=actions, episode=episode, stage=np.array(stage), files=np.array([f.name for f in files]))
    print(split, len(files), "episodes", n, "frames", flush=True)
