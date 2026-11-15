"""Tests for the background model downloader (models_download.py), using local
file:// URLs so nothing touches the network."""
from __future__ import annotations

from pathlib import Path

from app.models_download import ModelDownloader, Weight, download_file


def _source(tmp_path: Path, name: str, size: int) -> str:
    src = tmp_path / "src" / name
    src.parent.mkdir(exist_ok=True)
    src.write_bytes(b"x" * size)
    return src.as_uri()


def test_download_file_renames_only_when_complete(tmp_path):
    dest = tmp_path / "out" / "m.pth"
    dest.parent.mkdir()
    seen = []
    download_file(_source(tmp_path, "m.pth", 3_000_000), dest, lambda d, t: seen.append((d, t)))
    assert dest.stat().st_size == 3_000_000
    assert not dest.with_name("m.pth.part").exists()
    assert seen[-1] == (3_000_000, 3_000_000)


def test_downloads_missing_weights_in_the_background(tmp_path):
    weights = (
        Weight("a.pth", "Model A", _source(tmp_path, "a.pth", 1000), 900),
        Weight("b.pth", "Model B", _source(tmp_path, "b.pth", 2000), 2100),
    )
    out = tmp_path / "weights"
    d = ModelDownloader(out, weights)
    assert d.state == "downloading" and not d.ready_to_forge
    d.start()
    d.wait(10)
    status = d.status()
    assert status["state"] == "ready" and d.ready_to_forge
    assert status["done_bytes"] == status["total_bytes"] == 3000  # real sizes replace estimates
    assert [f["state"] for f in status["files"]] == ["done", "done"]
    assert (out / "a.pth").exists() and (out / "b.pth").exists()


def test_present_weights_are_not_downloaded_again(tmp_path):
    out = tmp_path / "weights"
    out.mkdir()
    (out / "a.pth").write_bytes(b"y" * 10)
    d = ModelDownloader(out, (Weight("a.pth", "A", "file:///nonexistent", 999),))
    assert d.state == "ready"
    d.start()  # nothing to do — and the bogus URL is never fetched
    assert d.state == "ready"


def test_failure_is_reported_and_retry_recovers(tmp_path):
    out = tmp_path / "weights"
    good = Weight("good.pth", "Good", _source(tmp_path, "good.pth", 100), 100)
    missing_src = tmp_path / "src" / "late.pth"
    late = Weight("late.pth", "Late", missing_src.as_uri(), 100)
    d = ModelDownloader(out, (good, late))
    d.start()
    d.wait(10)
    status = d.status()
    assert status["state"] == "failed" and not d.ready_to_forge
    failed = next(f for f in status["files"] if f["name"] == "late.pth")
    assert failed["state"] == "failed" and failed["error"]
    assert not (out / "late.pth.part").exists()

    missing_src.write_bytes(b"z" * 100)  # the source comes back
    d.retry()
    d.wait(10)
    assert d.state == "ready"


def test_skipped_downloads_do_not_block_forging(tmp_path):
    d = ModelDownloader(tmp_path, (Weight("a.pth", "A", "file:///nonexistent", 1),), skip=True)
    d.start()
    assert d.state == "skipped" and d.ready_to_forge
