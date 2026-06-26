#!/usr/bin/env python
"""Fetch model weights into the weights/ directory.

Run once on the host (or as a compose init step) before enabling the deep
backends. Until weights are present, the pipeline uses classical fallbacks.

    python scripts/download_weights.py
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

WEIGHTS = {
    # SCUNet blind real-image denoising (colour PSNR model).
    "scunet_color_real_psnr.pth": "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth",
    # Real-ESRGAN x4 general model.
    "RealESRGAN_x4plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    # GFPGAN v1.4 (used by both GFPGAN and as a face helper for CodeFormer).
    "GFPGANv1.4.pth": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth",
    # CodeFormer.
    "codeformer.pth": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
    # DDColor colorization (modelscope "large" model, ~912MB).
    "ddcolor_modelscope.pth": "https://huggingface.co/piddnad/DDColor-models/resolve/main/ddcolor_modelscope.pth",
}


def main() -> int:
    out = Path(__file__).resolve().parent.parent / "weights"
    out.mkdir(exist_ok=True)
    for name, url in WEIGHTS.items():
        dest = out / name
        if dest.exists():
            print(f"✓ {name} already present")
            continue
        print(f"↓ {name} ...")
        urllib.request.urlretrieve(url, dest)  # noqa: S310 - trusted release URLs
        print(f"✓ {name} -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
