"""Integration tests for serving the web client through FastAPI."""

import pytest
from fastapi.testclient import TestClient

from callshield.server.app import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.mark.integration
def test_client_index_served(client: TestClient) -> None:
    """GET /client/ serves the HTML web client with live monitor and audio inspector."""
    response = client.get("/client/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Call Shield" in response.text
    assert "audio-file-input" in response.text
    assert "tab-btn-upload" in response.text
    assert "dropzone" in response.text


@pytest.mark.integration
def test_root_serves_or_redirects_to_client(client: TestClient) -> None:
    """GET / redirects or serves the client index."""
    response = client.get("/", follow_redirects=True)
    assert response.status_code == 200
    assert "Call Shield" in response.text


@pytest.mark.integration
def test_client_static_assets_served(client: TestClient) -> None:
    """Static assets (JS and CSS) are served with correct MIME content types."""
    js_files = ["logic.js", "worklet.js", "app.js"]
    for js_file in js_files:
        resp = client.get(f"/client/{js_file}")
        assert resp.status_code == 200
        content_type = resp.headers["content-type"]
        assert "javascript" in content_type or "text/plain" in content_type

    css_resp = client.get("/client/style.css")
    assert css_resp.status_code == 200
    assert "text/css" in css_resp.headers["content-type"]
