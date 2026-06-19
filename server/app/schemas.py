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

    Stages run in a fixed sensible order: colorize -> upscale -> face_restore.
    """

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
    # Populated once the forged asset is accepted into Immich.
    new_asset_id: str | None = None
    stack_id: str | None = None


class AcceptResponse(BaseModel):
    new_asset_id: str
    stack_id: str
