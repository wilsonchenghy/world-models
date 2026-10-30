"""Train the Dreamer 4 tokenizer: masked autoencoding, MSE + LPIPS.

  python -m dreamer4.train_tok <data dir> <run dir> <steps>

Env: BATCH (8), CLIP (clip length, 8), LR (3e-4), MASK_MAX (0.9), LPIPS_W (0.2), TOK_CH (32).
Saves <run>/tok.pt every 2k steps (resumes from it) and <run>/tok_recon.png (top row real, bottom row recon).
"""
import os
import sys
import time
from pathlib import Path

import imageio
import lpips
import numpy as np
import torch
import torch.nn.functional as F

from dreamer4.common import dev, load_split, log, lr_at, resume, starts, tok_cfg
from dreamer4.model import Tokenizer

data, run, steps = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
run.mkdir(parents=True, exist_ok=True)
torch.backends.cuda.matmul.allow_tf32 = True
g = torch.Generator().manual_seed(0)

B, T = int(os.environ.get("BATCH", 8)), int(os.environ.get("CLIP", 8))
fr, _, ep, keep = load_split(data, "train")
st = starts(ep, keep, T)
efr, _, eep, ekeep = load_split(data, "eval")
est = starts(eep, ekeep, T)[::997][:16]
log(f"tokenizer: {len(st)} train clips, {len(est)} eval clips")
tok = Tokenizer(**tok_cfg()).to(dev)
perc = lpips.LPIPS(net="alex", verbose=False).to(dev).eval().requires_grad_(False)
opt = torch.optim.AdamW(tok.parameters(), lr=3e-4, weight_decay=0.01, betas=(0.9, 0.95))
log(f"tokenizer params {sum(p.numel() for p in tok.parameters()) / 1e6:.1f}M")
i0 = resume(tok, opt, run / "tok.pt")
peak = float(os.environ.get("LR", 3e-4))


def clip(frames, s):
    x = np.stack([frames[j:j + T] for j in s])
    return torch.from_numpy(x).to(dev).permute(0, 1, 4, 2, 3).float() / 255


def eval_recon():
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        xe = clip(efr, est)
        ye = tok.decode(tok.encode(xe)).float()
    return xe, ye


t0, hist = time.time(), []
for i in range(i0, steps):
    for pg in opt.param_groups:
        pg["lr"] = lr_at(i, steps, peak)
    x = clip(fr, st[torch.randint(len(st), (B,), generator=g).numpy()])
    with torch.autocast("cuda", torch.bfloat16):
        mp = torch.rand(B, T, device=dev) * float(os.environ.get("MASK_MAX", 0.9))
        y = tok.decode(tok.encode(x, mp))
        mse = F.mse_loss(y.float(), x)
        lp = perc(y.flatten(0, 1).float() * 2 - 1, x.flatten(0, 1) * 2 - 1).mean()
        loss = mse + float(os.environ.get("LPIPS_W", 0.2)) * lp
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(tok.parameters(), 1.0)
    opt.step()
    hist.append((mse.item(), lp.item()))
    if (i + 1) % 200 == 0:
        m = np.mean(hist, 0); hist = []
        xe, ye = eval_recon()
        emse = F.mse_loss(ye, xe).item()
        epsnr = (10 * torch.log10(1 / ((ye - xe) ** 2).mean((1, 2, 3, 4)))).mean().item()
        log(f"step {i + 1}/{steps} mse {m[0]:.5f} lpips {m[1]:.4f} | eval (no mask) mse {emse:.5f} psnr {epsnr:.2f} dB "
            f"| {200 / (time.time() - t0):.2f} it/s")
        t0 = time.time()
    if (i + 1) % 2000 == 0 or i + 1 == steps:
        torch.save({"model": tok.state_dict(), "opt": opt.state_dict(), "step": i + 1, "cfg": tok_cfg()}, run / "tok.pt")
        xe, ye = eval_recon()
        n = min(4, len(xe))
        im = torch.cat([xe[:n, ::2].flatten(0, 1), ye[:n, ::2].flatten(0, 1)], 0)
        H, W = im.shape[-2:]
        grid = im.reshape(2, -1, 3, H, W).permute(0, 3, 1, 4, 2).reshape(2 * H, -1, 3)
        imageio.imwrite(run / "tok_recon.png", (grid.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
log("TOK_DONE")
