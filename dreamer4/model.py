"""Dreamer 4 world model (Hafner, Yan, Lillicrap 2025, "Training Agents Inside of Scalable World Models",
arXiv 2509.24527), re-implemented from the paper in PyTorch: a causal tokenizer (masked autoencoding, tanh bottleneck)
and an interactive dynamics model trained with shortcut forcing (x-prediction, per-frame signal levels and step sizes,
bootstrap targets, ramp loss weight). Both use the paper's 2D block-causal transformer: space attention inside each
frame, causal time attention every 4th layer, pre-RMSNorm, QK-norm, SwiGLU, RoPE over time.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],), self.g, 1e-6)


def rope(x, pos):
    """rotary embedding over the last dim of x (..., L, hd) at integer positions pos (L,)"""
    hd = x.shape[-1]
    f = 1.0 / (10000 ** (torch.arange(0, hd, 2, device=x.device).float() / hd))
    a = pos.float()[:, None] * f[None]
    c, s = a.cos().to(x.dtype), a.sin().to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).flatten(-2)


class Block(nn.Module):
    """one transformer layer over x (B, T, N, D): attention across N (space) or causally across T (time)"""

    def __init__(self, d, heads, time):
        super().__init__()
        self.h, self.time = heads, time
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d, bias=False), nn.Linear(d, d, bias=False)
        self.qn, self.kn = RMSNorm(d // heads), RMSNorm(d // heads)
        # with QK-norm, q.k / sqrt(hd) is at most sqrt(hd) (~5.7): too flat to attend sharply over hundreds of tokens,
        # so the logit scale is learned per layer, starting at 1 (q.k unscaled)
        self.scale = nn.Parameter(torch.tensor(1.0))
        hid = int(d * 8 / 3) // 64 * 64
        self.w12, self.w3 = nn.Linear(d, 2 * hid, bias=False), nn.Linear(hid, d, bias=False)

    def forward(self, x, space_mask=None, time_mask=None):
        B, T, N, D = x.shape
        y = self.n1(x)
        if self.time:
            y = y.transpose(1, 2).reshape(B * N, T, D)
        else:
            y = y.reshape(B * T, N, D)
        q, k, v = self.qkv(y).unflatten(-1, (3, self.h, D // self.h)).permute(2, 0, 3, 1, 4)
        q, k = self.qn(q) * self.scale, self.kn(k)
        if self.time:
            pos = torch.arange(T, device=x.device)
            q, k = rope(q, pos), rope(k, pos)
            m = time_mask  # (B, T, T) bool, True = may attend
            m = None if m is None else m[:, None].expand(B, N, T, T).reshape(B * N, 1, T, T)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=m, is_causal=m is None, scale=1.0)
            y = y.transpose(1, 2).reshape(B, N, T, D).transpose(1, 2)
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=space_mask, scale=1.0)
            y = y.transpose(1, 2).reshape(B, T, N, D)
        x = x + self.o(y)
        a, b = self.w12(self.n2(x)).chunk(2, -1)
        return x + self.w3(F.silu(a) * b)


class Stack(nn.Module):
    def __init__(self, d, heads, layers, time_every=4):
        super().__init__()
        self.blocks = nn.ModuleList([Block(d, heads, (i + 1) % time_every == 0) for i in range(layers)])
        self.norm = RMSNorm(d)

    def forward(self, x, space_mask=None, time_mask=None):
        for b in self.blocks:
            x = b(x, space_mask, time_mask)
        return self.norm(x)


# ---------------------------------------------------------------------------------------------------- tokenizer
class Tokenizer(nn.Module):
    """frames (B, T, 3, 120, 128) in [0, 1] -> latents (B, T, L, C) in [-1, 1] -> frames. Causal in time.
    Masked autoencoding (patches dropped at a per-frame rate) and a tanh bottleneck as in the paper. Deviation: the
    latents are tied to the image grid (each one summarises a 2x2 block of patches) instead of learned latent tokens
    that read the patches through attention. With our compute, learned latent tokens did not get past a blurry level
    average (16 dB PSNR) while the same transformer without them reached 29 dB."""

    def __init__(self, d=256, heads=8, enc_layers=8, dec_layers=8, channels=16, patch=8, group=2, hw=(128, 128),
                 bottleneck="tanh"):
        super().__init__()
        self.p, self.hw, self.g, self.bottleneck = patch, hw, group, bottleneck
        self.gh, self.gw = hw[0] // patch, hw[1] // patch
        self.np = self.gh * self.gw
        self.L = self.np // group ** 2
        pd = 3 * patch * patch
        self.patch_in = nn.Linear(pd, d)
        self.mask_tok = nn.Parameter(torch.randn(d) * 0.02)
        self.pos_p = nn.Parameter(torch.randn(self.np, d) * 0.02)
        self.enc = Stack(d, heads, enc_layers)
        self.to_c = nn.Linear(group * group * d, channels)
        self.from_c = nn.Linear(channels, group * group * d)
        self.pos_d = nn.Parameter(torch.randn(self.np, d) * 0.02)
        self.dec = Stack(d, heads, dec_layers)
        self.patch_out = nn.Linear(d, pd)

    def pad(self, x):  # 120x128 -> 128x128 (4 rows top and bottom)
        return F.pad(x, (0, 0, 4, 4))

    def patchify(self, x):
        B, T = x.shape[:2]
        p = self.p
        x = x.reshape(B, T, 3, self.gh, p, self.gw, p)
        return x.permute(0, 1, 3, 5, 2, 4, 6).reshape(B, T, self.np, 3 * p * p)

    def unpatchify(self, x):
        B, T = x.shape[:2]
        p = self.p
        x = x.reshape(B, T, self.gh, self.gw, 3, p, p).permute(0, 1, 4, 2, 5, 3, 6)
        return x.reshape(B, T, 3, self.hw[0], self.hw[1])[..., 4:-4, :]

    def group(self, h):  # (B,T,np,d) -> (B,T,L,g*g*d), 2x2 blocks of patches
        B, T, _, D = h.shape
        g = self.g
        h = h.reshape(B, T, self.gh // g, g, self.gw // g, g, D).permute(0, 1, 2, 4, 3, 5, 6)
        return h.reshape(B, T, self.L, g * g * D)

    def ungroup(self, h):
        B, T, _, GD = h.shape
        g = self.g
        h = h.reshape(B, T, self.gh // g, self.gw // g, g, g, GD // (g * g)).permute(0, 1, 2, 4, 3, 5, 6)
        return h.reshape(B, T, self.np, GD // (g * g))

    def encode(self, x, mask_p=None):
        h = self.patch_in(self.patchify(self.pad(x)))
        if mask_p is not None:  # masked autoencoding: drop patches with a per-frame rate
            drop = torch.rand(h.shape[:3], device=h.device) < mask_p[..., None]
            h = torch.where(drop[..., None], self.mask_tok.to(h.dtype), h)
        h = self.enc(h + self.pos_p)
        z = self.to_c(self.group(h))
        return torch.tanh(z) if self.bottleneck == "tanh" else z

    def decode(self, z):
        h = self.dec(self.ungroup(self.from_c(z)) + self.pos_d)
        return torch.sigmoid(self.unpatchify(self.patch_out(h)))


# ---------------------------------------------------------------------------------------------------- dynamics
class Dynamics(nn.Module):
    """predicts clean latents z1 from corrupted latents, per-frame signal level tau, step size d and the action that
    led to each frame. Tokens per frame: L latent tokens, R registers, 1 (tau, d) token, 1 action token."""

    def __init__(self, d=512, heads=8, layers=12, latents=32, channels=16, registers=4, actions=7, kmax=64):
        super().__init__()
        self.kmax, self.nsteps = kmax, int(math.log2(kmax)) + 1
        self.z_in = nn.Linear(channels, d)
        self.pos_z = nn.Parameter(torch.randn(latents, d) * 0.02)
        self.reg = nn.Parameter(torch.randn(registers, d) * 0.02)
        self.tau_emb = nn.Embedding(kmax + 1, d)  # tau on the grid of the smallest step, 0..kmax
        self.d_emb = nn.Embedding(self.nsteps, d)  # d = 2^-k, k = 0..log2(kmax)
        self.act_emb = nn.Embedding(actions + 1, d)  # last index: no action (first frame)
        self.act_base = nn.Parameter(torch.randn(d) * 0.02)
        self.tf = Stack(d, heads, layers)
        self.out = nn.Linear(d, channels)
        self.L = latents

    def forward(self, z, tau_i, k, act, time_mask=None):
        """z (B,T,L,C) corrupted; tau_i (B,T) long in [0,kmax]; k (B,T) long step-size exponent; act (B,T) long"""
        B, T = z.shape[:2]
        h = torch.cat([self.z_in(z) + self.pos_z, self.reg.expand(B, T, -1, -1),
                       (self.tau_emb(tau_i) + self.d_emb(k))[:, :, None],
                       (self.act_emb(act) + self.act_base)[:, :, None]], 2)
        return self.out(self.tf(h, None, time_mask)[:, :, :self.L])


def shortcut_forcing_loss(model, z1, act, boot_frac=0.25, single_frac=0.3):
    """Shortcut forcing on clean latents z1 (B,T,L,C). Signal level tau: 0 = noise, 1 = clean.
    Most sequences train at the smallest step (x-space flow loss); a fraction gets random step sizes per frame with
    bootstrap targets from two half steps (v-space, scaled by (1-tau)^2). Ramp weight 0.9 tau + 0.1. A fraction of
    sequences sees no history (each frame attends only to itself) so the model can also generate without context."""
    B, T = z1.shape[:2]
    dev, K = z1.device, model.kmax
    kmin = model.nsteps - 1
    boot = torch.rand(B, device=dev) < boot_frac
    k = torch.where(boot[:, None], torch.randint(0, model.nsteps, (B, T), device=dev), torch.full((B, T), kmin, device=dev))
    step = 2 ** (kmin - k)  # d on the grid of 1/K
    tau_i = (torch.rand(B, T, device=dev) * (K // step)).long() * step  # multiples of d in [0, 1 - d]
    tau = (tau_i.float() / K)[..., None, None]
    z0 = torch.randn_like(z1)
    zt = (1 - tau) * z0 + tau * z1
    tm = torch.ones(T, T, dtype=torch.bool, device=dev).tril().expand(B, T, T).clone()
    single = torch.rand(B, device=dev) < single_frac
    tm[single] = torch.eye(T, dtype=torch.bool, device=dev)
    pred = model(zt, tau_i, k, act, tm)
    w = (0.9 * tau + 0.1)
    flow = ((pred - z1) ** 2).mean((-1, -2), keepdim=True)
    is_boot = (k < kmin)[..., None, None]
    loss = torch.where(is_boot, 0.0, w * flow)
    if is_boot.any():
        with torch.no_grad():
            # frames that are not bootstrapped keep their input and step, so every frame's history matches the main pass
            kb = k < kmin
            kh = torch.where(kb, k + 1, k)  # half step
            half = torch.where(kb, step // 2, 0)
            p1 = model(zt, tau_i, kh, act, tm)
            b1 = (p1 - zt) / (1 - tau)
            z_mid = torch.where(is_boot, zt + b1 * (half.float() / K)[..., None, None], zt)
            tau_mid_i = tau_i + half
            p2 = model(z_mid, tau_mid_i, kh, act, tm)
            b2 = (p2 - z_mid) / (1 - (tau_mid_i.float() / K)[..., None, None]).clamp(min=1.0 / K)
            target = (b1 + b2) / 2
        v = (pred - zt) / (1 - tau)
        bl = ((1 - tau) ** 2 * (v - target) ** 2).mean((-1, -2), keepdim=True)
        loss = torch.where(is_boot, w * bl, loss)
    return loss.mean(), flow[~is_boot].mean()


@torch.no_grad()
def generate(model, ctx_z, ctx_act, actions, steps=4, ctx_signal=0.9, window=32):
    """autoregressive rollout: ctx_z (B,T0,L,C) clean latents of real frames, ctx_act (B,T0) the action index that led
    to each context frame, actions (B,H) the actions taken from the last context frame on. Each new frame starts from
    noise and is denoised in `steps` shortcut steps (d = 1/steps); history is corrupted to signal ctx_signal."""
    K, kmin = model.kmax, model.nsteps - 1
    kd = int(math.log2(steps))
    ci = round(ctx_signal * K)
    zs, acts = list(ctx_z.unbind(1)), list(ctx_act.unbind(1))
    hist = [(1 - ci / K) * torch.randn_like(z) + ci / K * z for z in zs]  # corrupted once, then kept fixed
    B, dev = ctx_z.shape[0], ctx_z.device
    for h in range(actions.shape[1]):
        acts.append(actions[:, h])
        x = torch.randn_like(zs[0])
        for s in range(steps):
            ti = s * K // steps
            Z = torch.stack(hist[-(window - 1):] + [x], 1)
            A = torch.stack(acts[-Z.shape[1]:], 1)
            Tn = Z.shape[1]
            tau_i = torch.full((B, Tn), ci, device=dev, dtype=torch.long); tau_i[:, -1] = ti
            k = torch.full((B, Tn), kmin, device=dev, dtype=torch.long); k[:, -1] = kd
            p = model(Z, tau_i, k, A)[:, -1].float()
            x = x + (p - x) / (1 - ti / K) * (1 / steps)
        zs.append(x)
        hist.append((1 - ci / K) * torch.randn_like(x) + ci / K * x)
    return torch.stack(zs, 1)
