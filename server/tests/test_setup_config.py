"""Tests for the connection settings managed by the setup page (setup_config.py)."""
from __future__ import annotations

import json
import stat

import pytest

from app.config import Settings
from app.setup_config import SETTINGS_FILE, SetupConfig


def _settings(tmp_path, **env) -> Settings:
    # Only what's passed counts as "set by the environment".
    return Settings(_env_file=None, FORGE_WEIGHTS_DIR=str(tmp_path), FORGE_DEVICE="cpu", **env)


def test_fresh_install_needs_setup(tmp_path):
    setup = SetupConfig(_settings(tmp_path))
    c = setup.connection
    # The code default for IMMICH_BASE_URL doesn't count as configured.
    assert c.immich_base_url == "" and c.immich_api_key == "" and c.forge_api_token == ""
    assert setup.setup_required and not setup.token_set


def test_save_immich_then_token_completes_setup(tmp_path):
    setup = SetupConfig(_settings(tmp_path))
    setup.save_immich("http://immich:2283/", "key-123")
    token = setup.ensure_token()
    assert token and len(token) == 64
    assert setup.ensure_token() is None  # only created once
    c = setup.connection
    assert c.immich_base_url == "http://immich:2283"  # trailing slash dropped
    assert c.immich_api_key == "key-123" and c.forge_api_token == token
    assert not setup.setup_required


def test_settings_survive_a_restart_with_private_permissions(tmp_path):
    setup = SetupConfig(_settings(tmp_path))
    setup.save_immich("http://immich:2283", "key-123")
    setup.ensure_token()
    path = tmp_path / SETTINGS_FILE
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    again = SetupConfig(_settings(tmp_path))
    assert again.connection == setup.connection


def test_blank_key_keeps_the_saved_one(tmp_path):
    setup = SetupConfig(_settings(tmp_path))
    setup.save_immich("http://a:2283", "key-123")
    setup.save_immich("http://b:2283", None)
    assert setup.connection.immich_base_url == "http://b:2283"
    assert setup.connection.immich_api_key == "key-123"


def test_regenerate_token_replaces_it(tmp_path):
    setup = SetupConfig(_settings(tmp_path))
    first = setup.ensure_token()
    second = setup.regenerate_token()
    assert second != first and setup.connection.forge_api_token == second


def test_user_chosen_token(tmp_path):
    setup = SetupConfig(_settings(tmp_path))
    assert setup.set_token("  my-own-token-1234567  ") == "my-own-token-1234567"
    assert setup.connection.forge_api_token == "my-own-token-1234567"
    for weak in ("short", "has a space in it ok", "tab\tseparated-token-xx"):
        with pytest.raises(ValueError):
            setup.set_token(weak)
    assert setup.connection.forge_api_token == "my-own-token-1234567"  # unchanged


def test_environment_wins_and_is_read_only(tmp_path):
    (tmp_path / SETTINGS_FILE).write_text(json.dumps({"immich_base_url": "http://file:2283"}))
    setup = SetupConfig(_settings(tmp_path, IMMICH_BASE_URL="http://env:2283", FORGE_API_TOKEN="t"))
    assert setup.connection.immich_base_url == "http://env:2283"
    assert setup.from_env("immich_base_url") and setup.from_env("forge_api_token")
    assert not setup.from_env("immich_api_key")
    with pytest.raises(PermissionError):
        setup.save_immich("http://other:2283", None)
    with pytest.raises(PermissionError):
        setup.regenerate_token()


def test_empty_environment_value_does_not_count(tmp_path):
    setup = SetupConfig(_settings(tmp_path, FORGE_API_TOKEN=""))
    assert not setup.from_env("forge_api_token")
    assert setup.ensure_token()  # can be created on the page


def test_picks_up_changes_from_another_process(tmp_path):
    # app.reset_token runs in a separate process; the server must see it.
    server = SetupConfig(_settings(tmp_path))
    old = server.ensure_token()
    new = SetupConfig(_settings(tmp_path)).regenerate_token()
    assert server.connection.forge_api_token == new != old


def test_unreadable_file_is_ignored(tmp_path):
    (tmp_path / SETTINGS_FILE).write_text("{not json")
    assert SetupConfig(_settings(tmp_path)).setup_required
