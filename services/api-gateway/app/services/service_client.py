"""
Inter-Service HTTP Client
===========================
Async ``httpx``-based client for proxying requests to downstream
microservices (ingestion, ML engine, risk-scoring, alert-service).

Design decisions
----------------
* **Connection pooling** — a single ``httpx.AsyncClient`` per downstream
  service, re-used across requests.  FastAPI's lifespan manager handles
  creation and teardown.
* **Timeout budget** — each call gets a 10 s timeout.  Dashboard
  aggregation runs calls in parallel (``asyncio.gather``), so the
  worst-case wall-clock is ~10 s, not 40 s.
* **Request-ID forwarding** — the ``X-Request-ID`` header from the
  gateway is forwarded to every downstream service so the entire call
  chain can be correlated in logs.
* **Structured error handling** — downstream 4xx/5xx are translated to
  gateway-level responses, not blindly proxied.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from app.config import get_settings

settings = get_settings()

# ── Lazy singleton clients (initialised in lifespan) ──────────

_clients: dict[str, httpx.AsyncClient] = {}

_SERVICE_URLS: dict[str, str] = {
    "ingestion": settings.ingestion_service_url,
    "ml_engine": settings.ml_engine_url,
    "risk_scoring": settings.risk_scoring_service_url,
    "alert": settings.alert_service_url,
}

_TIMEOUT = httpx.Timeout(timeout=10.0, connect=3.0)


async def init_clients() -> None:
    """Create persistent HTTP clients — call during lifespan startup."""
    for name, base_url in _SERVICE_URLS.items():
        _clients[name] = httpx.AsyncClient(
            base_url=base_url,
            timeout=_TIMEOUT,
            headers={"User-Agent": "itds-api-gateway/1.0"},
        )


async def close_clients() -> None:
    """Gracefully close all HTTP clients — call during lifespan shutdown."""
    for client in _clients.values():
        await client.aclose()
    _clients.clear()


def _get_client(service: str) -> httpx.AsyncClient:
    """Return the pre-initialised client for a downstream service."""
    client = _clients.get(service)
    if not client:
        raise RuntimeError(
            f"HTTP client for '{service}' not initialised. "
            f"Ensure init_clients() was called during startup."
        )
    return client


# ── Generic proxy helpers ─────────────────────────────────────

async def proxy_get(
    service: str,
    path: str,
    params: dict[str, Any] | None = None,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Forward a GET request to a downstream service."""
    headers = {}
    if request_id:
        headers["X-Request-ID"] = request_id

    client = _get_client(service)
    resp = await client.get(path, params=params, headers=headers)
    resp.raise_for_status()
    return resp.json()


async def proxy_post(
    service: str,
    path: str,
    json_body: dict[str, Any] | None = None,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Forward a POST request to a downstream service."""
    headers = {}
    if request_id:
        headers["X-Request-ID"] = request_id

    client = _get_client(service)
    resp = await client.post(path, json=json_body, headers=headers)
    resp.raise_for_status()
    return resp.json()


async def proxy_patch(
    service: str,
    path: str,
    json_body: dict[str, Any] | None = None,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Forward a PATCH request to a downstream service."""
    headers = {}
    if request_id:
        headers["X-Request-ID"] = request_id

    client = _get_client(service)
    resp = await client.patch(path, json=json_body, headers=headers)
    resp.raise_for_status()
    return resp.json()


# ── Dashboard aggregation ────────────────────────────────────

async def aggregate_dashboard(request_id: str | None = None) -> dict[str, Any]:
    """
    Single call that fans out to risk-scoring, alert, and ingestion
    services in parallel and merges the results into one response.

    This is the *request aggregation* pattern recommended by the
    API-Gateway spec — the frontend makes one HTTP call instead of
    three, reducing round-trip latency and connection overhead.

    Fan-out targets:
      1. Risk-scoring  → ``GET /api/v1/thresholds/current``
      2. Alert service → ``GET /api/v1/alerts/stats``
      3. Alert service → ``GET /api/v1/alerts/?status=open&limit=10``
      4. Ingestion     → ``GET /healthz`` (to report ingestion health)

    Any individual failure is caught and returned as ``null`` in the
    merged payload — partial degradation is preferable to total failure.
    """

    async def _safe_get(service: str, path: str, params: dict | None = None) -> Any:
        """Wrap proxy_get with exception handling for graceful degradation."""
        try:
            return await proxy_get(service, path, params=params, request_id=request_id)
        except Exception:
            return None

    # Fan-out: all calls execute concurrently.
    (
        thresholds,
        alert_stats,
        recent_alerts,
        ingestion_health,
    ) = await asyncio.gather(
        _safe_get("risk_scoring", "/api/v1/thresholds/current"),
        _safe_get("alert", "/api/v1/alerts/stats"),
        _safe_get("alert", "/api/v1/alerts/", params={"status": "open", "limit": "10"}),
        _safe_get("ingestion", "/healthz"),
    )

    return {
        "thresholds": thresholds,
        "alert_stats": alert_stats,
        "recent_alerts": recent_alerts,
        "ingestion_status": ingestion_health,
        "_aggregated": True,
        "_source_services": ["risk_scoring", "alert", "ingestion"],
    }
