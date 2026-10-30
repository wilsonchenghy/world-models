"""Data loading, configs and checkpoint helpers shared by the Dreamer 4 training scripts.

Data format (any environment): <dir>/{split}_frames.npy uint8 (N, H, W, 3) and <dir>/{split}_meta.npz with
`actions` (N,) int and `episode` (N,) int, for both splits, train and eval.
"""
import math
import os
import time

import numpy as np
import torch

dev = "cuda"
N_ACTIONS = int(os.environ.get("ACTIONS", 7))


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_split(data, split):
    """frames memmap, actions and episode ids of a split"""
    f = np.load(data / f"{split}_frames.npy", mmap_mode="r")
    m = np.load(data / f"{split}_meta.npz")
    return f, m["actions"].astype(np.int64), m["episode"].astype(np.int64)


def starts(ep, T, keep=None):
    """indices i such that frames i..i+T-1 are in one episode (and all in `keep`, if given)"""
    ok = np.zeros(len(ep), bool)
    ok[:len(ep) - T + 1] = ep[:len(ep) - T + 1] == ep[T - 1:]
    if keep is not None:
        ok[:len(ep) - T + 1] &= keep[:len(ep) - T + 1] & keep[T - 1:]
    return np.nonzero(ok)[0]


def prev_actions(act, ep):
    """the action that led to each frame; N_ACTIONS (the extra index) for the first frame of an episode"""
    prev = np.concatenate([[N_ACTIONS], act[:-1]])
    prev[np.diff(ep, prepend=-1) != 0] = N_ACTIONS
    return prev


def tok_cfg():
    return dict(d=256, heads=8, enc_layers=8, dec_layers=8, channels=int(os.environ.get("TOK_CH", 32)), patch=8,
                bottleneck="tanh")


def dyn_cfg(latents, channels):
    return dict(d=int(os.environ.get("DYN_D", 512)), heads=8, layers=int(os.environ.get("DYN_LAYERS", 12)), latents=latents,
                channels=channels, registers=4, actions=N_ACTIONS, kmax=64)


def lr_at(i, steps, peak, warm=int(os.environ.get("WARM", 500))):
    """linear warmup, then cosine down to 10% of peak at `steps`"""
    return peak * min(1.0, (i + 1) / warm) * (0.5 * (1 + math.cos(math.pi * min(1.0, i / max(steps, 1)))) * 0.9 + 0.1)


def resume(model, opt, path, ema=None):
    """load model, optimizer (and EMA) from `path` and return its step; step 0 if there is no checkpoint yet"""
    if not path.exists():
        log(f"no checkpoint at {path}, starting from step 0")
        return 0
    ck = torch.load(path, map_location=dev)
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
    if ema is not None:
        ema.update(ck["ema"])
    log(f"resumed {path} at step {ck['step']}")
    return ck["step"]
