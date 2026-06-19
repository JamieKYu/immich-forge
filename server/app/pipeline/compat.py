"""Compatibility shims for the deep-learning model libs.

`basicsr` (a transitive dep of GFPGAN/CodeFormer/Real-ESRGAN) does:

    from torchvision.transforms.functional_tensor import rgb_to_grayscale

but `functional_tensor` was removed in torchvision 0.17. Rather than pin Torch
back several minor versions, alias the one symbol basicsr needs onto the current
`functional` module. Must run BEFORE importing basicsr/gfpgan/realesrgan.
"""
from __future__ import annotations

import sys
import types


def patch_basicsr_torchvision() -> None:
    name = "torchvision.transforms.functional_tensor"
    if name in sys.modules:
        return
    try:
        import torchvision.transforms.functional as F
    except ImportError:
        return  # torchvision not installed; nothing to patch
    if hasattr(F, "functional_tensor"):
        return
    shim = types.ModuleType(name)
    # basicsr only imports rgb_to_grayscale from here.
    shim.rgb_to_grayscale = F.rgb_to_grayscale  # type: ignore[attr-defined]
    sys.modules[name] = shim
