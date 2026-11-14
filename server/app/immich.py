"""Thin async client for the Immich REST API (broker side).

Only the endpoints Forge needs: read asset metadata, pull the original bytes,
upload the forged image, and stack it so the forged version becomes primary.
"""
from __future__ import annotations

import mimetypes
from datetime import datetime, timezone

import httpx



class ImmichError(RuntimeError):
    pass


class ImmichClient:
    DEVICE_ID = "immich-forge"

    def __init__(self, conn):
        # `conn`: anything with `immich_base_url` and `immich_headers` (Settings,
        # or setup_config.Connection).
        self._base = conn.immich_base_url.rstrip("/")
        self._headers = conn.immich_headers
        self._client = httpx.AsyncClient(base_url=self._base, timeout=60.0)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def ping(self) -> bool:
        r = await self._client.get("/api/server/ping", headers=self._headers)
        return r.status_code == 200

    async def get_asset(self, asset_id: str) -> dict:
        r = await self._client.get(f"/api/assets/{asset_id}", headers=self._headers)
        if r.status_code != 200:
            raise ImmichError(f"get_asset {asset_id} failed: {r.status_code} {r.text}")
        return r.json()

    async def download_original(self, asset_id: str) -> bytes:
        r = await self._client.get(
            f"/api/assets/{asset_id}/original", headers=self._headers
        )
        if r.status_code != 200:
            raise ImmichError(
                f"download_original {asset_id} failed: {r.status_code} {r.text}"
            )
        return r.content

    async def upload_asset(
        self,
        data: bytes,
        filename: str,
        *,
        file_created_at: str | None = None,
        is_favorite: bool = False,
    ) -> tuple[str, bool]:
        """Upload bytes as a new asset. Returns (asset_id, is_duplicate)."""
        now = datetime.now(timezone.utc).isoformat()
        created = file_created_at or now
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"

        # deviceAssetId must be unique or Immich treats it as a duplicate of an
        # earlier upload; key it to the filename which already carries the source id.
        form = {
            "deviceAssetId": f"{self.DEVICE_ID}-{filename}",
            "deviceId": self.DEVICE_ID,
            "fileCreatedAt": created,
            "fileModifiedAt": now,
            "isFavorite": str(is_favorite).lower(),
        }
        files = {"assetData": (filename, data, content_type)}
        # Note: this endpoint expects multipart/form-data, NOT application/json.
        headers = {"x-api-key": self._headers["x-api-key"], "Accept": "application/json"}
        r = await self._client.post(
            "/api/assets", headers=headers, data=form, files=files
        )
        if r.status_code not in (200, 201):
            raise ImmichError(f"upload_asset failed: {r.status_code} {r.text}")
        body = r.json()
        return body["id"], body.get("status") == "duplicate"

    async def create_stack(self, primary_asset_id: str, asset_ids: list[str]) -> str:
        """Create a stack; `primary_asset_id` becomes the cover/primary asset.

        `asset_ids` must include the primary id. See
        https://api.immich.app/endpoints/stacks
        """
        payload = {"primaryAssetId": primary_asset_id, "assetIds": asset_ids}
        r = await self._client.post(
            "/api/stacks",
            headers={**self._headers, "Content-Type": "application/json"},
            json=payload,
        )
        if r.status_code not in (200, 201):
            raise ImmichError(f"create_stack failed: {r.status_code} {r.text}")
        return r.json()["id"]


async def check_connection(base_url: str, api_key: str) -> dict:
    """Test an Immich address and API key for the setup page.

    Returns {"ok": True, "user": name} or {"ok": False, "kind": ..., "message":
    ...}, telling apart a bad address, something that isn't Immich, and a key
    that's wrong or lacks permission."""
    base = base_url.strip().rstrip("/")
    if not base.startswith(("http://", "https://")):
        return {"ok": False, "kind": "invalid-url",
                "message": "Enter the full address, e.g. http://192.168.1.50:2283"}
    async with httpx.AsyncClient(base_url=base, timeout=10.0) as client:
        try:
            ping = await client.get("/api/server/ping")
        except httpx.HTTPError as exc:
            return {"ok": False, "kind": "unreachable",
                    "message": f"Can't reach {base} ({type(exc).__name__}). Check the address and "
                               "that Forge can reach it — use the IP address, not localhost."}
        if ping.status_code != 200:
            return {"ok": False, "kind": "not-immich",
                    "message": f"{base} answered, but doesn't look like Immich (HTTP {ping.status_code})."}
        if not api_key:
            return {"ok": False, "kind": "no-key", "message": "Enter an Immich API key."}
        try:
            me = await client.get("/api/users/me", headers={"x-api-key": api_key.strip()})
        except httpx.HTTPError as exc:
            return {"ok": False, "kind": "unreachable", "message": f"Immich stopped answering ({type(exc).__name__})."}
    if me.status_code == 401:
        return {"ok": False, "kind": "bad-key", "message": "Immich rejected this API key."}
    if me.status_code == 403:
        return {"ok": False, "kind": "no-permission",
                "message": "This API key doesn't have enough permissions. Create one with all permissions."}
    if me.status_code != 200:
        return {"ok": False, "kind": "error", "message": f"Immich returned HTTP {me.status_code}."}
    user = me.json()
    return {"ok": True, "user": user.get("name") or user.get("email") or "your account"}
