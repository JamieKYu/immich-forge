"""Tests for the async Immich client, with the Immich REST API mocked via respx."""
from __future__ import annotations

import httpx
import pytest
import respx

from app.immich import ImmichClient, ImmichError

BASE = "http://immich-test:2283"


@pytest.fixture
async def client(settings):
    c = ImmichClient(settings)
    try:
        yield c
    finally:
        await c.aclose()


@respx.mock
async def test_ping_true_on_200(client):
    respx.get(f"{BASE}/api/server/ping").mock(return_value=httpx.Response(200))
    assert await client.ping() is True


@respx.mock
async def test_ping_false_on_non_200(client):
    respx.get(f"{BASE}/api/server/ping").mock(return_value=httpx.Response(500))
    assert await client.ping() is False


@respx.mock
async def test_get_asset_returns_json(client):
    respx.get(f"{BASE}/api/assets/abc").mock(
        return_value=httpx.Response(200, json={"id": "abc", "originalFileName": "p.jpg"})
    )
    asset = await client.get_asset("abc")
    assert asset["originalFileName"] == "p.jpg"


@respx.mock
async def test_get_asset_raises_on_error(client):
    respx.get(f"{BASE}/api/assets/missing").mock(
        return_value=httpx.Response(404, text="not found")
    )
    with pytest.raises(ImmichError):
        await client.get_asset("missing")


@respx.mock
async def test_download_original_returns_bytes(client):
    respx.get(f"{BASE}/api/assets/abc/original").mock(
        return_value=httpx.Response(200, content=b"\xff\xd8jpegbytes")
    )
    assert await client.download_original("abc") == b"\xff\xd8jpegbytes"


@respx.mock
async def test_upload_asset_new(client):
    route = respx.post(f"{BASE}/api/assets").mock(
        return_value=httpx.Response(201, json={"id": "new-id", "status": "created"})
    )
    asset_id, is_dup = await client.upload_asset(b"data", "forged-x.jpg")
    assert (asset_id, is_dup) == ("new-id", False)
    assert route.called


@respx.mock
async def test_upload_asset_flags_duplicate(client):
    respx.post(f"{BASE}/api/assets").mock(
        return_value=httpx.Response(200, json={"id": "orig-id", "status": "duplicate"})
    )
    asset_id, is_dup = await client.upload_asset(b"data", "forged-x.jpg")
    assert (asset_id, is_dup) == ("orig-id", True)


@respx.mock
async def test_upload_asset_raises_on_error(client):
    respx.post(f"{BASE}/api/assets").mock(
        return_value=httpx.Response(500, text="boom")
    )
    with pytest.raises(ImmichError):
        await client.upload_asset(b"data", "forged-x.jpg")


@respx.mock
async def test_create_stack_returns_id_with_primary_first(client):
    captured = {}

    def _capture(request):
        import json

        captured.update(json.loads(request.content))
        return httpx.Response(201, json={"id": "stack-1"})

    respx.post(f"{BASE}/api/stacks").mock(side_effect=_capture)

    stack_id = await client.create_stack("primary", ["primary", "child"])
    assert stack_id == "stack-1"
    assert captured["primaryAssetId"] == "primary"
    assert "primary" in captured["assetIds"]


@respx.mock
async def test_create_stack_raises_on_error(client):
    respx.post(f"{BASE}/api/stacks").mock(return_value=httpx.Response(400, text="bad"))
    with pytest.raises(ImmichError):
        await client.create_stack("primary", ["primary", "child"])
