# Vendored from DMD (Dense Minutia Descriptor)

`resnet.py`, `units.py`, `inception.py`, `model_zoo.py` in this directory are copied
verbatim from the official implementation of:

> Zhiyu Pan, Yongjie Duan, Xiongjun Guan, Jianjiang Feng, Jie Zhou.
> "Latent Fingerprint Matching via Dense Minutia Descriptor." IJCB 2024.
> https://arxiv.org/abs/2405.01199 (extended journal version: arXiv:2507.15297)

Licensed under Apache License 2.0. No modifications were made to these four files.
The pretrained checkpoint (`weights/dmd.pt`, trained on NIST SD14, epoch 126) is a
copy of the authors' officially released `DMD.tar`.

`src/losses/identity_loss.py::DMDEmbeddingExtractor` is original code in this repo
that wraps `model_zoo.DMD` (a dense per-patch descriptor + foreground-mask network,
built for correlation-based matching) to satisfy the single global, L2-normalized
embedding interface `IdentityCosineLoss` expects: mask-weighted global-average-pool
of both descriptor branches, concatenated, L2-normalized. This is a simplification
of DMD's intended use (dense correspondence) — adequate as a differentiable identity
signal for `L_Identity`, not a substitute for DMD's own matcher.
