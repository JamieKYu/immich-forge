"""Pydantic request/response models for the Forge API."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    done = "done"
    error = "error"


class ForgeOperations(BaseModel):
    """Which enhancement stages to run, and their parameters.

    Stages run in a fixed sensible order: denoise -> colorize ->
    (face_restore + upscale). Denoise is first so later stages (and the
    upscaler especially) don't amplify sensor noise. When both face_restore and
    upscale are on they run as one stage: the background is upscaled and the
    restored faces are pasted onto it last, so the general upscaler never runs
    over face pixels (which turns them waxy/"un-human").
    """

    # Denoise / low-light. Runs first. `denoise_strength` blends the denoised
    # result back toward the original (1 = fully denoised, 0 = original) so the
    # model's smoothing can be dialled down. `low_light` adds a classical
    # CLAHE + gamma brightening pass (SCUNet/NAFNet only denoise, they don't
    # brighten), applied after denoising.
    denoise: bool = False
    denoise_strength: float = Field(1.0, ge=0.0, le=1.0)
    low_light: bool = False

    colorize: bool = False

    upscale: bool = True
    upscale_factor: int = Field(4, ge=2, le=4)  # Real-ESRGAN supports x2/x4

    face_restore: bool = False
    # CodeFormer fidelity<->quality knob (0 = max quality, 1 = max fidelity).
    # Default leans strongly toward fidelity: lower values let CodeFormer
    # regenerate facial detail from its codebook prior, and the two eyes can be
    # reinvented independently (mismatched iris colour/shape). 0.95 keeps the
    # output anchored to the real input face — both eyes track the source — while
    # still cleaning it up. Drop it toward 0.5 only for heavily damaged photos
    # that need more aggressive reconstruction.
    face_fidelity: float = Field(0.95, ge=0.0, le=1.0)


class ForgeRequest(BaseModel):
    asset_id: str = Field(..., description="Immich asset id to forge")
    operations: ForgeOperations = ForgeOperations()


class ForgeAnalysis(BaseModel):
    """Result of the pre-submit source-quality precheck (see pipeline/quality.py).

    `low_quality` is the only field the UI acts on — it drives a generic, soft,
    non-blocking "results may be limited" note. `metrics` carries the raw scores
    for debugging/tuning."""

    low_quality: bool
    metrics: dict[str, float] = {}


class JobInfo(BaseModel):
    job_id: str
    asset_id: str
    status: JobStatus
    progress: float = 0.0          # 0..1
    stage: str | None = None       # human-readable current stage
    error: str | None = None
    notes: list[str] = []          # user-facing adjustments (e.g. upscale clamped)
    # Populated once the forged asset is accepted into Immich.
    new_asset_id: str | None = None
    stack_id: str | None = None


class AcceptResponse(BaseModel):
    new_asset_id: str
    stack_id: str
