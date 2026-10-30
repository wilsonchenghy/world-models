# world-models

A from-scratch PyTorch implementation of the Dreamer 4 world model
([Hafner, Yan, Lillicrap 2025](https://arxiv.org/abs/2509.24527)), with Super Mario Bros as the first environment.

From a few real frames and a stream of controller actions, the model imagines the frames that follow.

## Layout

```
dreamer4/            environment-agnostic model and training
  model.py           block-causal transformer, Tokenizer, Dynamics, shortcut_forcing_loss, generate
  common.py          data loading, configs, LR schedule, checkpoints
  train_tok.py       tokenizer: masked autoencoding, MSE + LPIPS
  encode.py          frames -> latents
  train_dyn.py       dynamics: shortcut forcing on the latents
envs/mario/          Super Mario Bros data and evaluation
  collect.py         biased-random play, frame skip 4, 120x128
  pack.py            episodes -> packed train/eval arrays
  gen_video.py       real vs imagined video + A-vs-NOOP jump check
```

`dreamer4/` reads one packed format, so a new environment only has to write it:
`{split}_frames.npy` uint8 (N, H, W, 3) and `{split}_meta.npz` with `actions` (N,) and `episode` (N,).

## Model

- **Tokenizer** (12.4M): 8x8 patches, an 8-layer encoder and decoder, 64 latents x 32 channels per frame with a tanh
  bottleneck, causal in time. Masked autoencoding drops 0-90% of patches per frame.
  Deviation from the paper: each latent summarises a 2x2 block of patches instead of being a free learned token.
- **Dynamics** (37.5M): 12 layers, d=512. Per frame: 64 latent tokens, 4 registers, a (signal level, step size) token
  and an action token. Space attention within a frame, causal time attention every 4th layer.
  Trained with shortcut forcing (x-prediction, per-frame signal levels, bootstrap targets for large steps), so a
  frame is generated in 4 steps.

## Run (Mario)

Run everything from the repo root.

```bash
# data (needs gym==0.23.1, gym-super-mario-bros, numpy<2): ~1M frames in ~10 min with 14 workers
python -m envs.mario.collect data/raw 1000000 14
python -m envs.mario.pack data/raw data/packed

# model (needs torch with CUDA, lpips, imageio)
BATCH=16 python -m dreamer4.train_tok data/packed runs/mario 40000
python -m dreamer4.encode data/packed runs/mario
BATCH=8 SEQ=32 python -m dreamer4.train_dyn data/packed runs/mario 148000

# evaluation
python -m envs.mario.gen_video data/packed runs/mario runs/mario/vid 30
```

Training resumes from the last checkpoint if you rerun the same command.

## Results so far (RTX 5070 Ti, 16 GB)

| | |
| --- | --- |
| Data | 1,003,384 frames, 2,571 episodes (levels 1-1 to 1-4), 5% held out |
| Tokenizer | 38.25 dB PSNR on held-out frames after 40k steps (5.3 it/s, batch 16) |
| Dynamics, 56k steps | 8-frame rollout latent MSE 0.017 vs 0.072 for copying the last frame (4.1 it/s, batch 8) |
| Generation | 4 steps per frame, faster than real time |

Rollouts track the real game for about 2-4 s, then continue as a plausible but different part of the level.
