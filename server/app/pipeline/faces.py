"""Face detection / alignment / paste-back, shared by every face backend.

`facexlib.FaceRestoreHelper` detects faces (RetinaFace), warps each to an
aligned 512px crop, and pastes restored crops back with a face-parsing mask
(no hard seams). The face *models* (GFPGAN, CodeFormer, or both chained) only
ever see those crops, via the `restore_crop` callable passed to `restore()`.
This is the flow `GFPGANer` and upstream CodeFormer's inference script both
use; owning it here means no `gfpgan`/`basicsr` dependency.

facexlib fetches its detection + parsing weights into `model_rootpath` (the
weights volume) on first use; `download_weights.py` pre-fetches them.

All image I/O is BGR uint8 (the pipeline convention). Crops are handed to
`restore_crop` as RGB float (1,3,512,512) tensors in [-1, 1] — the range both
GFPGAN and CodeFormer were trained on — and it returns the same.
"""
from __future__ import annotations

import logging
from typing import Callable

import cv2
import numpy as np
import torch
from facexlib.utils.face_restoration_helper import FaceRestoreHelper

log = logging.getLogger("forge.face")

FACE_SIZE = 512

RestoreCrop = Callable[[torch.Tensor], torch.Tensor]


def crop_to_tensor(crop_bgr: np.ndarray, device) -> torch.Tensor:
    """BGR uint8 HxWx3 -> RGB float (1,3,H,W) in [-1, 1] on `device`."""
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float()
    return t.div_(127.5).sub_(1.0).to(device)


def tensor_to_crop(t: torch.Tensor) -> np.ndarray:
    """RGB float (1,3,H,W) in [-1, 1] -> BGR uint8 HxWx3."""
    rgb = t.detach().float().clamp(-1.0, 1.0).add(1.0).mul(127.5).round()
    rgb = rgb.squeeze(0).permute(1, 2, 0).byte().cpu().numpy()
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


class FacePipeline:
    def __init__(self, device, model_rootpath: str | None = None):
        self.device = device
        # face_size=512 matches GFPGAN's and CodeFormer's training crop.
        # use_parse=True pastes back through a face-parsing mask (no seams).
        self.helper = FaceRestoreHelper(
            1,
            face_size=FACE_SIZE,
            crop_ratio=(1, 1),
            det_model="retinaface_resnet50",
            save_ext="png",
            use_parse=True,
            device=device,
            model_rootpath=model_rootpath,
        )

    def restore(
        self,
        img_bgr: np.ndarray,
        restore_crop: RestoreCrop,
        factor: int = 1,
        bg_img: np.ndarray | None = None,
    ) -> np.ndarray:
        """Restore every detected face and paste it back.

        Faces are always detected/restored on `img_bgr` at the source
        resolution (a 512px aligned crop each). `factor` scales the paste: the
        restored crops are warped onto a `factor`x canvas via the helper's
        inverse affines. Pass `bg_img` as that upscaled canvas (e.g. the
        Real-ESRGAN output) so the general upscaler only ever touches the
        background — the restored faces are pasted on top of it last. With
        `factor=1` / `bg_img=None` this is a plain in-place restore.

        Returns `bg_img` (or the input) unchanged if no face is found, so a
        combined upscale still yields the upscaled image."""
        h = self.helper
        h.clean_all()
        # The helper was built at upscale_factor=1; override per call so the
        # inverse affines and paste target scale to the (upscaled) background.
        h.upscale_factor = factor
        h.read_image(img_bgr)
        num_faces = h.get_face_landmarks_5(
            only_center_face=False, resize=640, eye_dist_threshold=5
        )
        if num_faces == 0:
            return bg_img if bg_img is not None else img_bgr
        h.align_warp_face()

        for cropped_face in h.cropped_faces:
            try:
                with torch.inference_mode():
                    out = restore_crop(crop_to_tensor(cropped_face, self.device))
                restored_face = tensor_to_crop(out)
            except Exception as exc:  # noqa: BLE001 - OOM / bad crop: keep the input face
                log.warning("face restore failed on a face (%r); keeping original", exc)
                restored_face = cropped_face
            # add_restored_face gained an optional second `input_face` arg in
            # later facexlib; pass one positional arg for cross-version safety.
            h.add_restored_face(restored_face)

        h.get_inverse_affine(None)
        # draw_box / face_upsampler kwargs were added in later facexlib; call
        # with only the long-standing `upsample_img` for cross-version safety.
        # `bg_img` is the upscaled background (already factor x); the helper
        # resizes it to factor x if needed and warps the restored faces onto it.
        restored_img = h.paste_faces_to_input_image(upsample_img=bg_img)
        if restored_img is not None:
            return restored_img
        return bg_img if bg_img is not None else img_bgr
