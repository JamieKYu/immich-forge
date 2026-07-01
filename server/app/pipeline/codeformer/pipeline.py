"""Build + run CodeFormer for face restoration.

Unlike GFPGAN (driven end-to-end by `gfpgan.GFPGANer`), CodeFormer has no
turnkey runner in our dependency set, so this module wires it up the same way
upstream's `inference_codeformer.py` does:

  * `facexlib.FaceRestoreHelper` handles face *detection*, alignment to a 512px
    crop, and pasting the restored crop back into the full image;
  * the vendored `CodeFormer` net restores each aligned crop.

The whole reason CodeFormer exists here is its fidelity knob: `w` (0..1) blends
the encoder's identity features into the decoder. w=0 lets the codebook prior
run free (max quality / most "restored", least faithful); w=1 stays closest to
the input face. GFPGAN has no equivalent control — hence this backend.

All I/O is BGR uint8 (the pipeline convention).
"""
from __future__ import annotations

import logging

import numpy as np
import torch
from basicsr.utils import img2tensor, tensor2img
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from torchvision.transforms.functional import normalize

from .codeformer_arch import CodeFormer

log = logging.getLogger("forge.face")


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


class CodeFormerPipeline:
    def __init__(self, model_path: str, device, model_rootpath: str | None = None):
        self.device = device
        self.model = build_codeformer_model(model_path, device)
        # face_size=512 matches CodeFormer's training crop. use_parse=True uses a
        # face-parsing mask so only face pixels are pasted back (no hard seams).
        # On first run facexlib downloads its detection + parsing weights.
        self.face_helper = FaceRestoreHelper(
            1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model="retinaface_resnet50",
            save_ext="png",
            use_parse=True,
            device=device,
            model_rootpath=model_rootpath,
        )

    def restore(self, img_bgr: np.ndarray, fidelity: float) -> np.ndarray:
        """Restore every detected face in-place. Returns the original image
        unchanged if no face is found."""
        self.face_helper.clean_all()
        self.face_helper.read_image(img_bgr)
        num_faces = self.face_helper.get_face_landmarks_5(
            only_center_face=False, resize=640, eye_dist_threshold=5
        )
        if num_faces == 0:
            return img_bgr
        self.face_helper.align_warp_face()

        for cropped_face in self.face_helper.cropped_faces:
            # BGR uint8 crop -> RGB float tensor in [-1, 1] (CodeFormer's convention).
            cropped_face_t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)
            try:
                # no_grad (not inference_mode): tensor2img does an in-place clamp_,
                # which errors on inference-mode tensors.
                with torch.no_grad():
                    output = self.model(cropped_face_t, w=fidelity, adain=True)[0]
                restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
            except Exception as exc:  # noqa: BLE001 - OOM / bad crop: keep the input face
                log.warning("CodeFormer restore failed on a face (%r); keeping original", exc)
                restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))
            # add_restored_face gained an optional second `input_face` arg in
            # later facexlib; older releases (what gfpgan pulls) take only the
            # restored face. Pass one positional arg for cross-version safety.
            self.face_helper.add_restored_face(restored_face.astype("uint8"))

        self.face_helper.get_inverse_affine(None)
        # draw_box / face_upsampler kwargs were added in later facexlib; call
        # with only the long-standing `upsample_img` for cross-version safety.
        restored_img = self.face_helper.paste_faces_to_input_image(upsample_img=None)
        return restored_img if restored_img is not None else img_bgr
