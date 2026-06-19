"""Thin async client for the Immich REST API (broker side).

Only the endpoints Forge needs: read asset metadata, pull the original bytes,
upload the forged image, and stack it so the forged version becomes primary.
"""
from __future__ import annotations

import mimetypes
from datetime import datetime, timezone

import httpx

from .config import Settings


class ImmichError(RuntimeError):
    pass


class ImmichClient:
    DEVICE_ID = "immich-forge"

    def __init__(self, settings: Settings):
        self._base = settings.immich_base_url.rstrip("/")
        self._headers = settings.immich_headers
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
