"""
Integration test for the API Gateway health endpoints.
Verifies the service starts and responds to health checks.
"""

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient


@pytest_asyncio.fixture
async def client():
    """Create a test client for the API Gateway."""
    # Import here to allow monkeypatching before app init
    from services.api_gateway.app.main import app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_health_endpoint(client):
    response = await client.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_metrics_endpoint(client):
    response = await client.get("/metrics")
    assert response.status_code == 200
    # Prometheus text format starts with # HELP or has metric lines
    assert "http" in response.text.lower() or "process" in response.text.lower()
