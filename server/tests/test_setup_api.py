"""Tests for the setup page's API: first-visitor setup, token-protected
management afterwards, and forging waiting for model downloads."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.config import Settings
from app.main import app
from app.setup_config import SetupConfig


class StubDownloader:
    def __init__(self, state="ready", done=1, total=1):
        self._state, self._done, self._total = state, done, total
        self.retried = False

    @property
    def ready_to_forge(self):
        return self._state in ("ready", "skipped")

    def status(self):
        return {"state": self._state, "done_bytes": self._done, "total_bytes": self._total, "files": []}

    def retry(self):
        self.retried = True


class FakeImmich:
    async def ping(self):
        return True

    async def aclose(self):
        pass


class FakeJobs:
    immich = None

    def get(self, job_id):
        return None


@pytest.fixture
def client(tmp_path, monkeypatch):
    # _env_file=None: ignore any local .env (it would count as environment values).
    settings = Settings(_env_file=None, FORGE_WEIGHTS_DIR=str(tmp_path), FORGE_DEVICE="cpu")
    app.state.settings = settings
    app.state.setup = SetupConfig(settings)
    app.state.downloader = StubDownloader()
    app.state.immich = FakeImmich()
    app.state.jobs = FakeJobs()

    async def fake_check(base_url, api_key):
        if api_key == "good-key":
            return {"ok": True, "user": "Jamie"}
        return {"ok": False, "kind": "bad-key", "message": "Immich rejected this API key."}

    monkeypatch.setattr(main, "check_connection", fake_check)
    # Swapping clients schedules a delayed close; keep tests quiet and simple.
    monkeypatch.setattr(main, "ImmichClient", lambda conn: FakeImmich())
    return TestClient(app)


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def test_page_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Forge for Immich" in r.text


def test_fresh_server_reports_setup_required(client):
    status = client.get("/setup/status").json()
    assert status["setup_required"] and not status["token_set"]
    health = client.get("/health").json()
    assert health["setup_required"] and health["immich"] is False
    assert health["models"]["state"] == "ready"
    # The API stays closed until setup is complete.
    assert client.get("/models").status_code == 503


def test_first_visitor_sets_up_and_gets_a_token(client):
    r = client.post("/setup/immich", json={"immich_base_url": "http://immich:2283", "immich_api_key": "good-key"})
    assert r.status_code == 200
    token = r.json()["token"]
    assert token and len(token) == 64
    assert not client.get("/setup/status").json()["setup_required"]
    # The extension can now use the API with that token.
    assert client.get("/models", headers=_auth(token)).status_code == 200


def test_save_refuses_a_failing_connection_unless_forced(client):
    body = {"immich_base_url": "http://immich:2283", "immich_api_key": "bad-key"}
    r = client.post("/setup/immich", json=body)
    assert r.status_code == 400 and "rejected" in r.json()["detail"]
    assert client.get("/setup/status").json()["setup_required"]
    r = client.post("/setup/immich", json={**body, "skip_test": True})
    assert r.status_code == 200 and r.json()["token"]


def test_after_setup_changes_need_the_token(client):
    token = client.post(
        "/setup/immich", json={"immich_base_url": "http://immich:2283", "immich_api_key": "good-key"}
    ).json()["token"]
    for method, path, body in [
        ("get", "/setup/config", None),
        ("post", "/setup/test", {"immich_base_url": "http://x:2283"}),
        ("post", "/setup/immich", {"immich_base_url": "http://x:2283"}),
        ("post", "/setup/token", None),
        ("post", "/setup/models/retry", None),
    ]:
        r = getattr(client, method)(path, json=body) if body else getattr(client, method)(path)
        assert r.status_code == 401, path
    config = client.get("/setup/config", headers=_auth(token)).json()
    assert config["immich_base_url"] == "http://immich:2283"
    assert config["immich_api_key_set"] and "immich_api_key" not in config  # never sent back
    # Changing the address keeps the saved key, and doesn't make a new token.
    r = client.post("/setup/immich", headers=_auth(token), json={"immich_base_url": "http://new:2283"})
    assert r.status_code == 200 and r.json()["token"] is None


def test_regenerating_the_token_retires_the_old_one(client):
    old = client.post(
        "/setup/immich", json={"immich_base_url": "http://immich:2283", "immich_api_key": "good-key"}
    ).json()["token"]
    new = client.post("/setup/token", headers=_auth(old)).json()["token"]
    assert new != old
    assert client.get("/models", headers=_auth(old)).status_code == 401
    assert client.get("/models", headers=_auth(new)).status_code == 200


def test_paste_your_own_token(client):
    old = client.post(
        "/setup/immich", json={"immich_base_url": "http://immich:2283", "immich_api_key": "good-key"}
    ).json()["token"]
    mine = "my-existing-extension-token-123"
    r = client.post("/setup/token", headers=_auth(old), json={"token": mine})
    assert r.status_code == 200 and r.json()["token"] == mine
    assert client.get("/models", headers=_auth(old)).status_code == 401
    assert client.get("/models", headers=_auth(mine)).status_code == 200
    r = client.post("/setup/token", headers=_auth(mine), json={"token": "weak"})
    assert r.status_code == 400 and "16 characters" in r.json()["detail"]


def test_token_can_be_set_before_immich_on_first_run(client):
    r = client.post("/setup/token", json={"token": "chosen-before-immich-123"})
    assert r.status_code == 200
    # Now the token protects the page, and saving Immich keeps that token.
    assert client.post("/setup/immich", json={"immich_base_url": "http://i:2283", "immich_api_key": "good-key"}).status_code == 401
    r = client.post(
        "/setup/immich", headers=_auth("chosen-before-immich-123"),
        json={"immich_base_url": "http://i:2283", "immich_api_key": "good-key"},
    )
    assert r.status_code == 200 and r.json()["token"] is None
    assert client.get("/models", headers=_auth("chosen-before-immich-123")).status_code == 200


def test_sign_in_with_immich_address_and_key(client):
    token = client.post(
        "/setup/immich", json={"immich_base_url": "http://immich:2283", "immich_api_key": "good-key"}
    ).json()["token"]
    r = client.post("/setup/signin", json={"immich_base_url": "http://immich:2283/", "immich_api_key": "good-key"})
    assert r.status_code == 200 and r.json()["token"] == token
    # The returned token unlocks the page.
    assert client.get("/setup/config", headers=_auth(r.json()["token"])).status_code == 200
    r = client.post("/setup/signin", json={"immich_base_url": "http://immich:2283", "immich_api_key": "wrong"})
    assert r.status_code == 401 and "doesn't match" in r.json()["detail"]


def test_immich_sign_in_needs_a_saved_connection(client):
    client.post("/setup/token", json={"token": "chosen-before-immich-123"})
    r = client.post("/setup/signin", json={"immich_base_url": "http://i:2283", "immich_api_key": "k"})
    assert r.status_code == 409 and "Forge token" in r.json()["detail"]


def test_environment_values_cannot_be_changed_on_the_page(client, tmp_path):
    settings = Settings(
        _env_file=None, FORGE_WEIGHTS_DIR=str(tmp_path), FORGE_API_TOKEN="env-token",
        IMMICH_BASE_URL="http://env:2283", IMMICH_API_KEY="env-key",
    )
    app.state.setup = SetupConfig(settings)
    config = client.get("/setup/config", headers=_auth("env-token")).json()
    assert config["from_env"] == {"immich_base_url": True, "immich_api_key": True, "forge_api_token": True}
    r = client.post("/setup/token", headers=_auth("env-token"))
    assert r.status_code == 409


def test_forging_waits_for_model_downloads(client):
    token = client.post(
        "/setup/immich", json={"immich_base_url": "http://immich:2283", "immich_api_key": "good-key"}
    ).json()["token"]
    app.state.downloader = StubDownloader("downloading", done=1, total=4)
    r = client.post("/forge", headers=_auth(token), json={"asset_id": "a1"})
    assert r.status_code == 503 and "downloading (25%)" in r.json()["detail"]
    app.state.downloader = StubDownloader("failed")
    r = client.post("/forge", headers=_auth(token), json={"asset_id": "a1"})
    assert r.status_code == 503 and "retry" in r.json()["detail"]


def test_retry_is_open_before_setup(client):
    downloader = app.state.downloader
    assert client.post("/setup/models/retry").status_code == 200
    assert downloader.retried
