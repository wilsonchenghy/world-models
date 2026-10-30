"""Encode every frame to tokenizer latents: <run>/latents_{split}.npy (N, L, C) float16 + latents_{split}_meta.npz.

  python -m dreamer4.encode <data dir> <run dir>

Env: CLIP (8): frames are encoded in causal chunks of this length from each episode start, as in training.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from dreamer4.common import dev, log
from dreamer4.model import Tokenizer

data, run = Path(sys.argv[1]), Path(sys.argv[2])
ck = torch.load(run / "tok.pt", map_location=dev)
tok = Tokenizer(**ck["cfg"]).to(dev).eval()
tok.load_state_dict(ck["model"])
T = int(os.environ.get("CLIP", 8))

for split in ("train", "eval"):
    fr = np.load(data / f"{split}_frames.npy", mmap_mode="r")
    mt = np.load(data / f"{split}_meta.npz")
    act, ep = mt["actions"].astype(np.int64), mt["episode"].astype(np.int64)
    out = np.lib.format.open_memmap(run / f"latents_{split}.npy", "w+", np.float16, (len(fr), tok.L, ck["cfg"]["channels"]))
    bounds = np.nonzero(np.diff(ep, prepend=-1))[0].tolist() + [len(ep)]
    chunks = [(s, min(s + T, b1)) for b0, b1 in zip(bounds[:-1], bounds[1:]) for s in range(b0, b1, T)]
    t0 = time.time()
    for c in range(0, len(chunks), 64):
        grp = chunks[c:c + 64]
        for L in sorted({b - a for a, b in grp}):
            sel = [(a, b) for a, b in grp if b - a == L]
            x = torch.from_numpy(np.stack([fr[a:b] for a, b in sel])).to(dev).permute(0, 1, 4, 2, 3).float() / 255
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                z = tok.encode(x).float().cpu().numpy().astype(np.float16)
            for (a, b), zz in zip(sel, z):
                out[a:b] = zz
    out.flush()
    np.savez(run / f"latents_{split}_meta.npz", actions=act, episode=ep)
    log(f"{split}: encoded {len(fr)} frames in {time.time() - t0:.0f} s")
