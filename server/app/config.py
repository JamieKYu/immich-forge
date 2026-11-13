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
    dejpeg_backend: str = Field("fbcnn", alias="FORGE_DEJPEG_BACKEND")         # fbcnn|none
    denoise_backend: str = Field("scunet", alias="FORGE_DENOISE_BACKEND")      # scunet|nlm|none
    upscale_backend: str = Field("realesrgan", alias="FORGE_UPSCALE_BACKEND")  # realesrgan|lanczos
    # codeformer default: it honours the face_fidelity knob (GFPGAN ignores it).
    # gfpgan+codeformer runs both (GFPGAN for structure, then CodeFormer for
    # texture; face_fidelity drives the CodeFormer pass).
    # NOTE: CodeFormer is S-Lab License 1.0 (non-commercial); set gfpgan for
    # commercial use.
    face_backend: str = Field("codeformer", alias="FORGE_FACE_BACKEND")        # codeformer|gfpgan|gfpgan+codeformer|none
    colorize_backend: str = Field("ddcolor", alias="FORGE_COLORIZE_BACKEND")   # ddcolor|none
    low_light_backend: str = Field("hvicidnet", alias="FORGE_LOW_LIGHT_BACKEND")  # hvicidnet|classical
    # HVI-CIDNet runs on the whole image up to this size (~1.35GB VRAM per MP,
    # so the 4MP default needs ~5.3GB); larger images run on a downscale carried
    # back to full res. Lower it if low-light OOMs; raise it on a big GPU (e.g.
    # 12MP on 24GB) for slightly truer brightness on more photos.
    low_light_max_pixels: int = Field(4_000_000, alias="FORGE_LOW_LIGHT_MAX_PIXELS")

    # --- Source-quality precheck (heuristic; see pipeline/quality.py) ---
    # An image is flagged low-quality if it fails ANY bar. Defaults lean
    # conservative (bias toward warning): real libraries skew low-quality, and a
    # spurious note is harmless since the user is never blocked from forging.
    quality_sharpness_min: float = Field(100.0, alias="FORGE_QUALITY_SHARPNESS_MIN")
    quality_hf_ratio_min: float = Field(0.010, alias="FORGE_QUALITY_HF_RATIO_MIN")
    quality_blockiness_max: float = Field(0.40, alias="FORGE_QUALITY_BLOCKINESS_MAX")

    # --- Auto-mode suggestions (heuristic; see pipeline/suggest.py) ---
    # /analyze suggests settings per photo from these cutoffs. They're starting
    # points: /analyze returns the raw measurements in `metrics`, and
    # scripts/suggest_report.py prints them for a folder of photos, to tune by.
    # Enlarge: x4 when the long side is at most this, else x2 up to the next.
    auto_x4_max_side: int = Field(1200, alias="FORGE_AUTO_X4_MAX_SIDE")
    auto_x2_max_side: int = Field(2500, alias="FORGE_AUTO_X2_MAX_SIDE")
    # JPEG cleanup: estimated JPEG quality below this, or an 8x8 block-edge
    # step (grey levels) at or above the next — that catches a heavily
    # compressed photo later re-saved at high quality. Conservative: FBCNN
    # softens photos that are already clean. Damage from an earlier save
    # followed by a resize/crop isn't on the file's grid and can't be seen
    # (FBCNN's own quality estimate can't either), so such photos are missed
    # rather than risk softening clean ones saved at the same quality.
    auto_jpeg_quality_max: float = Field(60.0, alias="FORGE_AUTO_JPEG_QUALITY_MAX")
    auto_jpeg_blocking_min: float = Field(2.0, alias="FORGE_AUTO_JPEG_BLOCKING_MIN")
    # Reduce noise: estimated noise sigma (0-255 scale) at or above this, or a
    # camera ISO at or above the next. 8 keeps fine film grain (sigma ~6 on an
    # old scan) from counting as noise worth removing.
    auto_noise_sigma_min: float = Field(8.0, alias="FORGE_AUTO_NOISE_SIGMA_MIN")
    auto_iso_min: int = Field(3200, alias="FORGE_AUTO_ISO_MIN")
    # Brighten: median brightness (0-255) at or below this AND 95th percentile
    # at or below the next (so a silhouette against a bright sky isn't "dark").
    auto_dark_median_max: float = Field(60.0, alias="FORGE_AUTO_DARK_MEDIAN_MAX")
    auto_dark_p95_max: float = Field(190.0, alias="FORGE_AUTO_DARK_P95_MAX")
    # ...but not when at least this share of the photo is near-black and the
    # rest has a median brightness of at least the next: a flash- or torch-lit
    # subject against a black background, where the darkness is the scene.
    # (Calibrated: such photos read 58-74% black / 60-82 lit; dim rooms and
    # concerts that did want brightening read 28-47% black, or 93% black with
    # the lit parts at 27.)
    auto_black_background_min: float = Field(0.52, alias="FORGE_AUTO_BLACK_BACKGROUND_MIN")
    auto_lit_subject_min: float = Field(45.0, alias="FORGE_AUTO_LIT_SUBJECT_MIN")
    # Colorize: black-and-white when the 95th-percentile chroma (Lab) is at
    # most `gray`, or a single-hue tint (sepia) when chroma is at most `tint`
    # and the hue concentration (0-1) is at least the last. Conservative:
    # colorizing a color photo replaces its real colors.
    auto_gray_chroma_max: float = Field(6.0, alias="FORGE_AUTO_GRAY_CHROMA_MAX")
    # (A real sepia print measured chroma 11.7 / hue 0.99; a pale, mostly
    # white-and-blue colour photo measured 19.2 / 0.92, so keep a margin.)
    auto_tint_chroma_max: float = Field(15.0, alias="FORGE_AUTO_TINT_CHROMA_MAX")
    auto_tint_hue_min: float = Field(0.97, alias="FORGE_AUTO_TINT_HUE_MIN")
    # The tint must also be warm — its hue angle (Lab a/b, degrees) in this
    # range, the brown/yellow of sepia and toned prints. A neutral scene with a
    # cool white-balance cast (snow read 313°) is a colour photo. And neither
    # rule applies if at least this share of pixels is strongly coloured
    # (chroma > 25): real colour somewhere (a UI icon read 0.3%; true sepia and
    # black-and-white prints read 0).
    auto_tint_angle_min: float = Field(20.0, alias="FORGE_AUTO_TINT_ANGLE_MIN")
    auto_tint_angle_max: float = Field(110.0, alias="FORGE_AUTO_TINT_ANGLE_MAX")
    auto_strong_color_min: float = Field(0.001, alias="FORGE_AUTO_STRONG_COLOR_MIN")
    # Restore faces (judged per photo, see suggest.face_plan): faces narrower
    # than `min` (source px) are ignored as too small to restore well; restore
    # when a remaining face is narrower than `small`, or softer than the
    # sharpness bar (Laplacian variance of a 128px face crop) — unless the
    # largest face is itself large and sharp. Conservative: restoration can
    # change how a person looks.
    auto_face_min_px: int = Field(80, alias="FORGE_AUTO_FACE_MIN_PX")
    auto_face_small_px: int = Field(200, alias="FORGE_AUTO_FACE_SMALL_PX")
    auto_face_sharpness_min: float = Field(50.0, alias="FORGE_AUTO_FACE_SHARPNESS_MIN")
    # CPU face detection for the Restore faces suggestion. Needs the ML deps and
    # detection_Resnet50_Final.pth in the weights dir (download_weights.py).
    auto_face_detection: bool = Field(True, alias="FORGE_AUTO_FACE_DETECTION")

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
