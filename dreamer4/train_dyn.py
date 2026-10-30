"""Train the Dreamer 4 dynamics model with shortcut forcing on tokenizer latents (run dreamer4.encode first).

  python -m dreamer4.train_dyn <data dir> <run dir> <steps>

Env: BATCH (16), SEQ (sequence length, 32), LR (3e-4), DYN_D (512), DYN_LAYERS (12), ACTIONS (7).
Every 2k steps: eval rollout of the last 8 frames of held-out sequences (latent MSE vs copy-last-frame), and a
checkpoint <run>/dyn.pt (model, EMA, optimizer; resumes from it).
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from dreamer4.common import dev, dyn_cfg, log, lr_at, prev_actions, resume, starts
from dreamer4.model import Dynamics, generate, shortcut_forcing_loss

data, run, steps = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
torch.backends.cuda.matmul.allow_tf32 = True
g = torch.Generator().manual_seed(0)

B, T = int(os.environ.get("BATCH", 16)), int(os.environ.get("SEQ", 32))
# train and eval latents in one tensor; eval episode ids are offset so no sequence spans both
lat = torch.from_numpy(np.concatenate([np.load(run / f"latents_{sp}.npy") for sp in ("train", "eval")]).astype(np.float32))
ms = [np.load(run / f"latents_{sp}_meta.npz") for sp in ("train", "eval")]
act = np.concatenate([m["actions"] for m in ms]).astype(np.int64)
ep = np.concatenate([m["episode"] + 10 ** 6 * j for j, m in enumerate(ms)]).astype(np.int64)
is_train = np.arange(len(ep)) < len(ms[0]["episode"])
st, est = starts(ep, T, is_train), starts(ep, T, ~is_train)[::499][:32]
prev = torch.from_numpy(prev_actions(act, ep))
log(f"dynamics: {len(st)} train sequences, {len(est)} eval; latent std {lat[:100000].std():.3f}")
cfg = dyn_cfg(lat.shape[1], lat.shape[2])
dyn = Dynamics(**cfg).to(dev)
opt = torch.optim.AdamW(dyn.parameters(), lr=3e-4, weight_decay=0.01, betas=(0.9, 0.95))
log(f"dynamics params {sum(p.numel() for p in dyn.parameters()) / 1e6:.1f}M")
ema = {k: v.detach().clone() for k, v in dyn.state_dict().items()}
i0 = resume(dyn, opt, run / "dyn.pt", ema)
peak = float(os.environ.get("LR", 3e-4))


def seqs(s):
    idx = torch.from_numpy(s)[:, None] + torch.arange(T)
    return lat[idx].to(dev), prev[idx].to(dev)


t0, hist = time.time(), []
for i in range(i0, steps):
    for pg in opt.param_groups:
        pg["lr"] = lr_at(i, steps, peak)
    z, a = seqs(st[torch.randint(len(st), (B,), generator=g).numpy()])
    with torch.autocast("cuda", torch.bfloat16):
        loss, flow = shortcut_forcing_loss(dyn, z, a)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(dyn.parameters(), 1.0)
    opt.step()
    with torch.no_grad():
        for k, v in dyn.state_dict().items():
            ema[k].lerp_(v, 0.001) if v.dtype.is_floating_point else ema[k].copy_(v)
    hist.append((loss.item(), flow.item()))
    if (i + 1) % 200 == 0:
        mh = np.mean(hist, 0); hist = []
        log(f"step {i + 1}/{steps} loss {mh[0]:.4f} flow(x-mse) {mh[1]:.4f} | {200 / (time.time() - t0):.2f} it/s")
        t0 = time.time()
    if (i + 1) % 2000 == 0 or i + 1 == steps:
        # 4-step generation of the last 8 frames of held-out sequences from their first T-8, vs copy-last-frame
        cur = {k: v.clone() for k, v in dyn.state_dict().items()}
        dyn.load_state_dict(ema); dyn.eval()
        z, a = seqs(est)
        with torch.autocast("cuda", torch.bfloat16):
            gz = generate(dyn, z[:, :T - 8], a[:, :T - 8], a[:, T - 8:], window=T)
        err = (gz[:, T - 8:].float() - z[:, T - 8:]).pow(2).mean().item()
        copy = (z[:, T - 9:T - 8] - z[:, T - 8:]).pow(2).mean().item()
        log(f"eval rollout 8 frames: latent mse {err:.4f} vs copy-last-frame {copy:.4f}")
        dyn.load_state_dict(cur); dyn.train()
        torch.save({"model": dyn.state_dict(), "ema": ema, "opt": opt.state_dict(), "step": i + 1, "cfg": cfg},
                   run / "dyn.pt")
log("DYN_DONE")
