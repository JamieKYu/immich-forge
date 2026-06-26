"""Runtime configuration, loaded from environment / .env."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")

    # --- Immich (the broker holds these; the browser never sees them) ---
    immich_base_url: str = Field("http://immich-server:2283", alias="IMMICH_BASE_URL")
    immich_api_key: str = Field("", alias="IMMICH_API_KEY")

    # --- Auth on the Forge API itself (so it's not an open GPU endpoint) ---
    # Required: when empty, the API refuses to serve (fail closed) so the Immich
    # library is never exposed through an unauthenticated proxy.
    forge_api_token: str = Field("", alias="FORGE_API_TOKEN")

    # Comma-separated CORS allow-list for the Forge API. Empty (default) adds no
    # permissive CORS headers — the extension reaches Forge via its granted host
    # permission, which is not subject to CORS, so this stays locked down and
    # arbitrary websites cannot read responses from the Forge server.
    cors_origins: str = Field("", alias="FORGE_CORS_ORIGINS")

    # --- GPU / pipeline ---
    device: str = Field("cuda", alias="FORGE_DEVICE")  # "cuda" | "cpu"
    weights_dir: Path = Field(Path("weights"), alias="FORGE_WEIGHTS_DIR")
    max_image_pixels: int = Field(40_000_000, alias="FORGE_MAX_IMAGE_PIXELS")
    # Cap on the upscaled output; the upscale factor is clamped down to fit. Stops
    # a large source + x4 from producing a ~300MP image that OOMs the GPU.
    max_output_pixels: int = Field(100_000_000, alias="FORGE_MAX_OUTPUT_PIXELS")
    tile_size: int = Field(512, alias="FORGE_TILE_SIZE")  # 0 disables tiling

    # Backend selection per stage. Each falls back to a classical-CV impl
    # when the deep model / weights are unavailable.
    denoise_backend: str = Field("scunet", alias="FORGE_DENOISE_BACKEND")      # scunet|nlm|none
    upscale_backend: str = Field("realesrgan", alias="FORGE_UPSCALE_BACKEND")  # realesrgan|lanczos
    face_backend: str = Field("gfpgan", alias="FORGE_FACE_BACKEND")            # gfpgan|codeformer|none
    colorize_backend: str = Field("ddcolor", alias="FORGE_COLORIZE_BACKEND")   # ddcolor|none

    # --- Jobs ---
    job_ttl_seconds: int = Field(3600, alias="FORGE_JOB_TTL_SECONDS")
    max_concurrent_gpu_jobs: int = Field(1, alias="FORGE_MAX_CONCURRENT_GPU_JOBS")

    @property
    def immich_headers(self) -> dict[str, str]:
        return {"x-api-key": self.immich_api_key, "Accept": "application/json"}

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
