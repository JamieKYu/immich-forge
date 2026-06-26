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

    Stages run in a fixed sensible order: denoise -> colorize -> face_restore
    -> upscale. Denoise is first so later stages (and the upscaler especially)
    don't amplify sensor noise.
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
    face_fidelity: float = Field(0.5, ge=0.0, le=1.0)


class ForgeRequest(BaseModel):
    asset_id: str = Field(..., description="Immich asset id to forge")
    operations: ForgeOperations = ForgeOperations()


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
