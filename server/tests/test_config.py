"""Tests for Settings-derived properties (CORS lockdown, Immich headers)."""
from __future__ import annotations

from app.config import Settings


def _settings(**kw) -> Settings:
    base = dict(FORGE_API_TOKEN="t", IMMICH_API_KEY="k")
    base.update(kw)
    return Settings(**base)


def test_cors_empty_by_default_locks_down():
    assert _settings().cors_origin_list == []


def test_cors_parses_and_trims_comma_list():
    s = _settings(FORGE_CORS_ORIGINS=" https://a.example , https://b.example ")
    assert s.cors_origin_list == ["https://a.example", "https://b.example"]


def test_cors_ignores_blank_entries():
    s = _settings(FORGE_CORS_ORIGINS="https://a.example,, ,https://b.example")
    assert s.cors_origin_list == ["https://a.example", "https://b.example"]


def test_immich_headers_carry_api_key():
    s = _settings(IMMICH_API_KEY="abc123")
    assert s.immich_headers["x-api-key"] == "abc123"
    assert s.immich_headers["Accept"] == "application/json"
