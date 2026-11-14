"""Connection settings that can be managed from the setup page.

The Immich address, Immich API key and Forge token used to be environment
variables only. Now the container can start without them: the setup page at
`/` saves them to `forge-settings.json` in the weights volume (already mounted
by every install, so a container update can't lose them), and the server picks
them up without a restart.

Environment variables (or `.env`) still win: a value set there is used as-is
and shown on the page as read-only, so existing deployments keep working.

First-run access is "first visitor": until a Forge token exists, anyone who can
reach the page can complete setup. Once a token exists, every change needs it.
"""
from __future__ import annotations

import json
import logging
import os
import secrets
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

from .config import Settings

log = logging.getLogger("forge.setup")

SETTINGS_FILE = "forge-settings.json"

# The managed fields: setting name -> key in the settings file.
_FIELDS = ("immich_base_url", "immich_api_key", "forge_api_token")


@dataclass(frozen=True)
class Connection:
    immich_base_url: str
    immich_api_key: str
    forge_api_token: str

    @property
    def immich_configured(self) -> bool:
        return bool(self.immich_base_url and self.immich_api_key)

    @property
    def immich_headers(self) -> dict[str, str]:
        return {"x-api-key": self.immich_api_key, "Accept": "application/json"}


class SetupConfig:
    """The effective connection settings: environment first, then the file."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._path = Path(settings.weights_dir) / SETTINGS_FILE
        self._lock = threading.Lock()
        # Only values actually provided (env / .env) count — not code defaults.
        self._from_env = {
            name for name in _FIELDS
            if name in settings.model_fields_set and getattr(settings, name)
        }
        self._mtime = self._file_mtime()
        self._stored = self._load()

    # --- reading ------------------------------------------------------------

    def _file_mtime(self) -> int | None:
        try:
            return self._path.stat().st_mtime_ns
        except FileNotFoundError:
            return None

    def _refresh_locked(self) -> None:
        # Pick up edits made by another process (app.reset_token) without a
        # restart, so a reset token works straight away.
        mtime = self._file_mtime()
        if mtime != self._mtime:
            self._mtime = mtime
            self._stored = self._load()

    def _load(self) -> dict[str, str]:
        try:
            data = json.loads(self._path.read_text())
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as exc:
            log.warning("ignoring unreadable %s (%r)", self._path, exc)
            return {}
        return {k: v for k, v in data.items() if k in _FIELDS and isinstance(v, str)}

    def _value(self, name: str) -> str:
        if name in self._from_env:
            return getattr(self._settings, name)
        return self._stored.get(name, "")

    @property
    def connection(self) -> Connection:
        with self._lock:
            self._refresh_locked()
            return Connection(*(self._value(name) for name in _FIELDS))

    def from_env(self, name: str) -> bool:
        return name in self._from_env

    @property
    def token_set(self) -> bool:
        return bool(self.connection.forge_api_token)

    @property
    def setup_required(self) -> bool:
        c = self.connection
        return not (c.forge_api_token and c.immich_configured)

    # --- writing ------------------------------------------------------------

    def save_immich(self, base_url: str, api_key: str | None) -> None:
        """Save the Immich address, and the key if one is given (None keeps the
        saved key). Raises PermissionError for a value set by the environment."""
        updates = {"immich_base_url": base_url.strip().rstrip("/")}
        if api_key:
            updates["immich_api_key"] = api_key.strip()
        self._save(updates)

    def ensure_token(self) -> str | None:
        """Create a Forge token if none exists. Returns it if newly created."""
        if self.token_set:
            return None
        token = new_token()
        self._save({"forge_api_token": token})
        return token

    def regenerate_token(self) -> str:
        token = new_token()
        self._save({"forge_api_token": token})
        return token

    def set_token(self, token: str) -> str:
        """Use a token the user chose (e.g. one their extension already has).
        Raises ValueError if it's too weak to protect the Immich library."""
        token = token.strip()
        problem = token_problem(token)
        if problem:
            raise ValueError(problem)
        self._save({"forge_api_token": token})
        return token

    def _save(self, updates: dict[str, str]) -> None:
        blocked = [name for name in updates if name in self._from_env]
        if blocked:
            raise PermissionError(f"{', '.join(blocked)} set by the container environment")
        with self._lock:
            self._refresh_locked()
            data = {**self._stored, **updates}
            _write_private(self._path, data)
            self._stored = data
            self._mtime = self._file_mtime()


def new_token() -> str:
    return secrets.token_hex(32)


MIN_TOKEN_LENGTH = 16


def token_problem(token: str) -> str | None:
    """Why a chosen token isn't acceptable, or None."""
    if len(token) < MIN_TOKEN_LENGTH:
        return f"Use at least {MIN_TOKEN_LENGTH} characters (or generate one)."
    if len(token) > 256:
        return "Use at most 256 characters."
    if any(c.isspace() or not c.isprintable() for c in token):
        return "The token can't contain spaces or hidden characters."
    return None


def _write_private(path: Path, data: dict) -> None:
    """Write JSON atomically with owner-only permissions (it holds secrets)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".forge-settings-", suffix=".tmp")
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
