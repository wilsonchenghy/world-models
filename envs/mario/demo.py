"""One-command demo: download the trained weights and a sample run, write a real vs imagined video.

  python -m envs.mario.demo [out.mp4] [seconds]

The model sees the first 8 real frames, then only the button presses. Needs a CUDA GPU.
"""
import json
import sys

import imageio
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import load_file

from dreamer4.model import Dynamics, Tokenizer, generate

REPO, CTX, CHUNK, FPS = "wilsonchenghy/world-models-mario", 8, 8, 15
out = sys.argv[1] if len(sys.argv) > 1 else "demo.mp4"
secs = float(sys.argv[2]) if len(sys.argv) > 2 else 30
if not torch.cuda.is_available():
    raise RuntimeError("the demo needs a CUDA GPU")
dev, get = "cuda", lambda name: hf_hub_download(REPO, name)

cfg = json.load(open(get("config.json")))
tok = Tokenizer(**cfg["tokenizer"]).to(dev).eval()
tok.load_state_dict(load_file(get("tokenizer.safetensors")))
dyn = Dynamics(**cfg["dynamics"]).to(dev).eval()
dyn.load_state_dict(load_file(get("dynamics.safetensors")))

sample = np.load(get("sample_1-1.npz"))
fr, act = sample["frames"], sample["actions"]
H = int(secs * FPS)
if H < 1 or CTX + H > len(fr):
    raise ValueError(f"seconds must give 1 to {len(fr) - CTX} frames, got {H}")
prev = torch.from_numpy(np.concatenate([[7], act[:-1]])).to(dev)  # the action that led to each frame; 7 = none

with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
    x = torch.from_numpy(fr[:CTX]).to(dev).permute(0, 3, 1, 2)[None].float() / 255
    g = generate(dyn, tok.encode(x).float(), prev[:CTX][None], prev[CTX:CTX + H][None], window=32)[0, CTX:].float()
    imagined = [tok.decode(g[None, a:a + CHUNK])[0].float().clamp(0, 1) for a in range(0, H, CHUNK)]
imagined = (torch.cat(imagined).permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)

big = lambda im: np.asarray(Image.fromarray(im).resize((384, 360), Image.NEAREST))
gap = np.zeros((360, 8, 3), np.uint8)
imageio.mimsave(out, [np.concatenate([big(a), gap, big(b)], 1) for a, b in zip(fr[CTX:CTX + H], imagined)],
                fps=FPS, quality=7, macro_block_size=8)
print(f"saved {out}: real (left) vs imagined (right), {H} frames")
