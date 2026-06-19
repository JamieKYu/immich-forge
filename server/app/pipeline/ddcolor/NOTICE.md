# Vendored: DDColor

The code in this directory is vendored from **DDColor**
(<https://github.com/piddnad/DDColor>), the official implementation of
"DDColor: Towards Photo-Realistic Image Colorization via Dual Decoders"
(ICCV 2023), licensed under **Apache License 2.0**.

Files (`model.py`, `pipeline.py`, `arch_utils/*`) are copied with one change:
`model.py`'s imports were rewritten from `basicsr.archs.ddcolor_arch_utils.*` to
the local `.arch_utils.*` so the model is self-contained (no `basicsr`/`timm`
dependency, avoiding a collision with the `basicsr` used by Real-ESRGAN).

Weights (`ddcolor_modelscope.pth`) are downloaded separately from
<https://huggingface.co/piddnad/DDColor-models> and are not committed.
