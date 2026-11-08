"""Build the vendored CodeFormer net for face restoration.

The face detection / alignment / paste-back flow lives in `faces.py` (shared
with GFPGAN); this module only builds the network. The whole reason CodeFormer
is here is its fidelity knob: `w` (0..1) blends the encoder's identity features
into the decoder. w=0 lets the codebook prior run free (max quality / most
"restored", least faithful); w=1 stays closest to the input face. GFPGAN has no
equivalent control.

Called on aligned 512px crops as RGB float tensors in [-1, 1]:
`model(crop, w=fidelity, adain=True)[0]`.
"""
from __future__ import annotations

import torch

from .codeformer_arch import CodeFormer


def build_codeformer_model(model_path: str, device) -> CodeFormer:
    """Build CodeFormer with the released-model config and load weights.

    `codeformer.pth` is a `{'params_ema': state_dict}` checkpoint for the
    dim_embd=512, n_layers=9, codebook_size=1024 model. Loaded strictly so a
    mismatched checkpoint raises here (the caller degrades to a no-op) rather
    than silently producing garbage.
    """
    model = CodeFormer(
        dim_embd=512,
        codebook_size=1024,
        n_head=8,
        n_layers=9,
        connect_list=["32", "64", "128", "256"],
    ).to(device)
    # weights_only=False: trusted release artifact (fetched by download_weights.py);
    # silences torch's pickle FutureWarning.
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "params_ema" in ckpt:
        state_dict = ckpt["params_ema"]
    elif isinstance(ckpt, dict) and "params" in ckpt:
        state_dict = ckpt["params"]
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
