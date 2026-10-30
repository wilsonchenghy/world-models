"""Videos from a trained Dreamer 4 Mario model.

  python -m envs.mario.gen_video <data dir> <run dir> <out dir> [seconds]

1. play.mp4: a held-out episode, real game (left) vs the world model's imagination (right) driven by the same button
   presses, from 8 real context frames; the pressed buttons are drawn under the frames.
2. jump.png + numbers: from held-out moments, the model imagines 12 frames with A held vs nothing pressed; prints how
   much the two futures differ above Mario's row band (a jump changes the upper screen, standing still doesn't).
"""
import sys
from pathlib import Path

import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

from dreamer4.model import Dynamics, Tokenizer, generate

data, run, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
secs = float(sys.argv[4]) if len(sys.argv) > 4 else 30
out.mkdir(parents=True, exist_ok=True)
dev = "cuda"
BUTTONS = {0: [], 1: ["R"], 2: ["R", "A"], 3: ["R", "B"], 4: ["R", "A", "B"], 5: ["A"], 6: ["L"]}
CTX, CHUNK = 8, 8

ck = torch.load(run / "tok.pt", map_location=dev)
tok = Tokenizer(**ck["cfg"]).to(dev).eval(); tok.load_state_dict(ck["model"])
ck = torch.load(run / "dyn.pt", map_location=dev)
dyn = Dynamics(**ck["cfg"]).to(dev).eval(); dyn.load_state_dict(ck["ema"])

split = "eval" if (data / "eval_frames.npy").exists() else "train"
fr = np.load(data / f"{split}_frames.npy", mmap_mode="r")
m = np.load(data / f"{split}_meta.npz")
act, ep = m["actions"].astype(np.int64), m["episode"].astype(np.int64)
held = np.ones(len(ep), bool) if split == "eval" else ep % 20 == 19
prev = np.concatenate([[7], act[:-1]]); prev[np.diff(ep, prepend=-1) != 0] = 7


@torch.no_grad()
def enc(frames):  # (T,120,128,3) uint8 -> (T,L,C), in causal chunks like training
    zs = []
    for a in range(0, len(frames), CHUNK):
        x = torch.from_numpy(np.ascontiguousarray(frames[a:a + CHUNK])).to(dev).permute(0, 3, 1, 2)[None].float() / 255
        with torch.autocast("cuda", torch.bfloat16):
            zs.append(tok.encode(x)[0].float())
    return torch.cat(zs)


@torch.no_grad()
def dec(z):  # (T,L,C) -> (T,120,128,3) uint8
    ims = []
    for a in range(0, len(z), CHUNK):
        with torch.autocast("cuda", torch.bfloat16):
            y = tok.decode(z[None, a:a + CHUNK])[0].float()
        ims.append((y.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8))
    return np.concatenate(ims)


@torch.no_grad()
def imagine(i0, actions_after, horizon):
    """context = real frames i0..i0+CTX-1; then `horizon` frames driven by actions_after (actions leading to each)"""
    z = enc(fr[i0:i0 + CTX])[None]
    a = torch.from_numpy(prev[i0:i0 + CTX])[None].to(dev)
    with torch.autocast("cuda", torch.bfloat16):
        g = generate(dyn, z, a, torch.as_tensor(actions_after, device=dev)[None, :horizon], window=32)
    return dec(g[0].float())


def pad_buttons(im, pressed, text):
    W = im.shape[1]
    canvas = Image.new("RGB", (W, im.shape[0] + 40), (20, 22, 28))
    canvas.paste(Image.fromarray(im), (0, 0))
    d = ImageDraw.Draw(canvas)
    for j, b in enumerate(["L", "R", "B", "A"]):
        x, y = 10 + j * 46, im.shape[0] + 6
        on = b in pressed
        d.rounded_rectangle((x, y, x + 38, y + 28), 6, fill=(240, 190, 50) if on else (60, 64, 74))
        d.text((x + 19, y + 14), {"L": "<", "R": ">"}.get(b, b), fill=(20, 22, 28) if on else (220, 220, 220), anchor="mm")
    d.text((W - 10, im.shape[0] + 20), text, fill=(220, 220, 220), anchor="rm")
    return np.asarray(canvas)


def up(im, s=3):
    return np.asarray(Image.fromarray(im).resize((im.shape[1] * s, im.shape[0] * s), Image.NEAREST))


# 1. real vs imagined play
H = int(secs * 15)
eps = [e for e in np.unique(ep[held]) if (ep == e).sum() > CTX + H + 1]
e = eps[0]
i0 = int(np.nonzero(ep == e)[0][0]) + 30
real = np.asarray(fr[i0:i0 + CTX + H])
imag = imagine(i0, prev[i0 + CTX:i0 + CTX + H], H)
frames = []
for t in range(CTX + H):
    a = int(act[i0 + t - 1]) if t else 0
    pressed = BUTTONS[a]
    left = pad_buttons(up(real[t]), pressed, "REAL game")
    right = pad_buttons(up(imag[t]), pressed, "context (real)" if t < CTX else "IMAGINED by the world model")
    frames.append(np.concatenate([left, np.full((left.shape[0], 8, 3), 20, np.uint8), right], 1))
imageio.mimsave(out / "play.mp4", frames, fps=15, quality=8, macro_block_size=8)
err = [np.abs(real[t].astype(float) - imag[t]).mean() for t in (CTX, CTX + 15, CTX + 60, CTX + H - 1)]
print(f"play.mp4: episode {e}, {H} imagined frames; mean abs pixel error at +0/+1/+4 s/end: " + " ".join(f"{v:.1f}" for v in err))

# 2. press A vs nothing, from 16 held-out moments
idx = np.nonzero(held)[0]
pick = [i for i in idx[::max(1, len(idx) // 64)] if ep[i] == ep[min(i + CTX, len(ep) - 1)]][:16]
diffs, rows = [], []
for i in pick:
    ja = imagine(i, [5] * 12, 12)[CTX:]
    jn = imagine(i, [0] * 12, 12)[CTX:]
    top = slice(20, 80)  # rows above the ground where a jumping Mario shows up
    diffs.append((np.abs(ja[:, top].astype(float) - jn[:, top]).mean(), np.abs(ja.astype(float) - jn).mean()))
    if len(rows) < 3:
        rows.append(np.concatenate([np.concatenate([up(ja[k], 2) for k in (2, 5, 8, 11)], 1),
                                    np.concatenate([up(jn[k], 2) for k in (2, 5, 8, 11)], 1)], 0))
d = np.array(diffs)
print(f"jump check ({len(pick)} moments): A vs NOOP futures differ by {d[:, 0].mean():.2f} (upper band) / "
      f"{d[:, 1].mean():.2f} (whole frame) mean abs pixel; 0 would mean the model ignores the A button")
imageio.imwrite(out / "jump.png", np.concatenate(rows, 0))
print("jump.png: per moment, top row = A held, bottom row = nothing pressed, at +0.13/+0.33/+0.53/+0.73 s")
