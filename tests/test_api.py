import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("MODEL_DIR", str(Path(__file__).resolve().parents[1] / "model"))
os.environ["MODEL_API_KEY"] = "test-key"

from app.main import app  # noqa: E402

H = {"X-API-Key": "test-key"}
LOG = ("travis_fold:start:script\r\n$ make test\r\nrunning tests\r\nFAIL: test_login expected 1 got 2\r\n"
       "Traceback (most recent call last):\r\n  File \"a.py\", line 3\r\nAssertionError\r\n"
       "travis_fold:end:script\r\nThe command \"make test\" exited with 1.\r\n")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_ok(client):
    r = client.post("/predict", json={"log_text": LOG, "top_k": 3}, headers=H)
    assert r.status_code == 200
    b = r.json()
    assert b["schema_version"] == "1.0" and len(b["top_lines"]) <= 3
    scores = [t["score"] for t in b["top_lines"]]
    assert scores == sorted(scores, reverse=True)


def test_auth_required(client):
    assert client.post("/predict", json={"log_text": LOG}).status_code == 401
    assert client.post("/predict", json={"log_text": LOG}, headers={"X-API-Key": "bad"}).status_code == 401


def test_validation_does_not_echo_log(client):
    secret = "SECRET-TOKEN-123"
    r = client.post("/predict", json={"log_text": secret, "top_k": 0}, headers=H)
    assert r.status_code == 422 and secret not in r.text


def test_empty_and_nonfold_warning(client):
    assert client.post("/predict", json={"log_text": ""}, headers=H).status_code == 422
    r = client.post("/predict", json={"log_text": "hello\nworld"}, headers=H)
    assert r.status_code == 200 and r.json()["warnings"]


def test_too_large(client, monkeypatch):
    monkeypatch.setenv("MAX_BODY_BYTES", "100")
    r = client.post("/predict", json={"log_text": "x\n" * 200}, headers=H)
    assert r.status_code == 413


def test_busy(client):
    sem = app.state.slots
    held = [sem.acquire(blocking=False) for _ in range(5)]
    try:
        r = client.post("/predict", json={"log_text": LOG}, headers=H)
        assert r.status_code == 503
    finally:
        for h in held:
            if h:
                sem.release()
