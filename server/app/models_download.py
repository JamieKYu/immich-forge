"""Model weights: what Forge downloads, and a background downloader with progress.

The server starts straight away and fetches any missing weights in a background
thread, so the setup page can open (and show progress) during the first-boot
download. Forging waits until every download has finished (see main.py), so no
pipeline stage ever runs with a weight missing.

Each file downloads to `<name>.part` and is renamed only when complete. An
interrupted download therefore never leaves a truncated file that a later start
would mistake for "already present".

`scripts/download_weights.py` uses the same list and download code, for a
foreground fetch.
"""
from __future__ import annotations

import logging
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("forge.models")


@dataclass(frozen=True)
class Weight:
    name: str        # file name in the weights dir
    label: str       # what it's for, shown on the setup page
    url: str
    approx_bytes: int  # rough size, shown until the download reports the real one


# Download order: the face detector first (Auto mode uses it), then smallest to
# largest so most stages are ready early.
WEIGHTS: tuple[Weight, ...] = (
    Weight(
        "detection_Resnet50_Final.pth", "Face detection",
        "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth",
        109_497_761,
    ),
    Weight(
        "hvi_cidnet_generalization.safetensors", "Brighten (HVI-CIDNet)",
        "https://huggingface.co/Fediory/HVI-CIDNet-Generalization/resolve/main/model.safetensors",
        8_000_000,
    ),
    Weight(
        "RealESRGAN_x4plus.pth", "Enlarge (Real-ESRGAN)",
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        67_000_000,
    ),
    Weight(
        "scunet_color_real_psnr.pth", "Reduce noise (SCUNet)",
        "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth",
        72_000_000,
    ),
    Weight(
        "parsing_parsenet.pth", "Face blending",
        "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth",
        85_000_000,
    ),
    Weight(
        "fbcnn_color.pth", "JPEG cleanup (FBCNN)",
        "https://github.com/jiaxi-jiang/FBCNN/releases/download/v1.0/fbcnn_color.pth",
        287_755_111,
    ),
    Weight(
        "GFPGANv1.4.pth", "Restore faces (GFPGAN)",
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth",
        348_632_874,
    ),
    Weight(
        "codeformer.pth", "Restore faces (CodeFormer)",
        "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
        376_637_898,
    ),
    Weight(
        "ddcolor_modelscope.pth", "Colorize (DDColor)",
        "https://huggingface.co/piddnad/DDColor-models/resolve/main/ddcolor_modelscope.pth",
        912_000_000,
    ),
)

_CHUNK = 1 << 20  # 1 MiB


def download_file(url: str, dest: Path, progress=None) -> None:
    """Download `url` to `dest` via `dest.part`, renaming only once complete.
    `progress(done_bytes, total_bytes_or_None)` is called as data arrives."""
    part = dest.with_name(dest.name + ".part")
    part.unlink(missing_ok=True)
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 - pinned release URLs
        length = resp.headers.get("Content-Length")
        total = int(length) if length and length.isdigit() else None
        done = 0
        with open(part, "wb") as out:
            while chunk := resp.read(_CHUNK):
                out.write(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total)
    if total is not None and done != total:
        part.unlink(missing_ok=True)
        raise OSError(f"incomplete download ({done} of {total} bytes)")
    part.replace(dest)


@dataclass
class FileStatus:
    weight: Weight
    state: str = "waiting"  # waiting | downloading | done | failed
    done_bytes: int = 0
    total_bytes: int = 0
    error: str | None = None


class ModelDownloader:
    """Fetches missing weights in a background thread and reports progress.

    Overall state:
      * ``downloading`` — still working;
      * ``ready`` — every weight is present;
      * ``failed`` — finished, but some downloads failed (see `retry`);
      * ``skipped`` — downloads turned off (FORGE_SKIP_WEIGHT_DOWNLOAD=1).
    """

    def __init__(self, weights_dir: Path, weights: tuple[Weight, ...] = WEIGHTS, skip: bool = False):
        self.weights_dir = weights_dir
        self.skip = skip
        self._lock = threading.Lock()
        self._files = [FileStatus(w, total_bytes=w.approx_bytes) for w in weights]
        self._thread: threading.Thread | None = None
        for f in self._files:
            if (weights_dir / f.weight.name).exists():
                f.state = "done"
                f.done_bytes = f.total_bytes = (weights_dir / f.weight.name).stat().st_size

    def start(self) -> None:
        if self.skip:
            log.info("model download skipped (FORGE_SKIP_WEIGHT_DOWNLOAD=1)")
            return
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            todo = [f for f in self._files if f.state != "done"]
            if not todo:
                return
            for f in todo:
                f.state, f.error, f.done_bytes = "waiting", None, 0
            self._thread = threading.Thread(target=self._run, args=(todo,), daemon=True)
            self._thread.start()

    def retry(self) -> None:
        """Download anything that's still missing (after a failure)."""
        self.start()

    def _run(self, todo: list[FileStatus]) -> None:
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        for f in todo:
            dest = self.weights_dir / f.weight.name
            with self._lock:
                f.state = "downloading"
            log.info("downloading %s ...", f.weight.name)

            def progress(done: int, total: int | None, f=f) -> None:
                with self._lock:
                    f.done_bytes = done
                    if total:
                        f.total_bytes = total

            try:
                download_file(f.weight.url, dest, progress)
            except Exception as exc:  # noqa: BLE001 - network / disk; reported on the page
                log.warning("download of %s failed: %r", f.weight.name, exc)
                with self._lock:
                    f.state, f.error = "failed", _short_error(exc)
                continue
            with self._lock:
                f.state = "done"
                f.done_bytes = f.total_bytes = dest.stat().st_size
            log.info("downloaded %s", f.weight.name)

    @property
    def state(self) -> str:
        with self._lock:
            return self._state_locked()

    def _state_locked(self) -> str:
        if all(f.state == "done" for f in self._files):
            return "ready"
        if self.skip:
            return "skipped"
        if any(f.state in ("waiting", "downloading") for f in self._files):
            return "downloading"
        return "failed"

    @property
    def ready_to_forge(self) -> bool:
        """Forging waits for every download (skipped downloads don't block)."""
        return self.state in ("ready", "skipped")

    def status(self) -> dict:
        with self._lock:
            done = sum(f.done_bytes for f in self._files)
            total = sum(max(f.total_bytes, f.done_bytes) for f in self._files)
            return {
                "state": self._state_locked(),
                "done_bytes": done,
                "total_bytes": total,
                "files": [
                    {
                        "name": f.weight.name,
                        "label": f.weight.label,
                        "state": f.state,
                        "done_bytes": f.done_bytes,
                        "total_bytes": max(f.total_bytes, f.done_bytes),
                        "error": f.error,
                    }
                    for f in self._files
                ],
            }

    def wait(self, timeout: float | None = None) -> None:
        """Block until the current download run ends (scripts and tests)."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while self._thread and self._thread.is_alive():
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            self._thread.join(remaining)
            if deadline is not None and time.monotonic() >= deadline:
                break


def _short_error(exc: BaseException) -> str:
    text = str(exc) or type(exc).__name__
    return text if len(text) <= 160 else text[:157] + "…"
