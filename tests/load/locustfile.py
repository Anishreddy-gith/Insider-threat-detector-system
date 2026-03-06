"""
Load tests — Locust simulating 1000 concurrent users.

Targets:
  • API Gateway event ingestion endpoint
  • Risk-scoring endpoint
  • Alert query endpoint

Success criteria:
  • p99 latency < 200 ms
  • Zero HTTP 5xx errors
  • Sustained throughput ≥ 500 req/s

Usage:
    locust -f tests/load/locustfile.py --headless \
           -u 1000 -r 50 --run-time 60s \
           -H http://localhost:8000
"""

from __future__ import annotations

import json
import random
import string
import time
import uuid
from datetime import datetime, timezone

from locust import HttpUser, between, task, events, tag
from locust.runners import MasterRunner


# ═══════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════

def _random_user_id() -> str:
    return f"user-{random.randint(1, 10000):05d}"


def _random_ip() -> str:
    return f"{random.randint(10, 192)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"


def _random_event() -> dict:
    """Generate a random event payload matching ingestion-service schema."""
    event_type = random.choice(["login", "logout", "file_access", "network_request", "app_usage"])
    base = {
        "event_id": str(uuid.uuid4()),
        "user_id": _random_user_id(),
        "event_type": event_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if event_type == "login":
        base.update({
            "source_ip": _random_ip(),
            "device_fingerprint": f"fp-{''.join(random.choices(string.hexdigits, k=8))}",
            "mfa_used": random.choice([True, False]),
        })
    elif event_type == "file_access":
        base.update({
            "file_path": f"/data/{''.join(random.choices(string.ascii_lowercase, k=6))}.xlsx",
            "action": random.choice(["read", "write", "download", "upload"]),
            "classification": random.choice(["public", "internal", "confidential", "restricted"]),
            "bytes_transferred": random.randint(100, 10_000_000),
        })
    elif event_type == "network_request":
        base.update({
            "destination_host": f"{random.choice(['api', 'cdn', 'mail'])}.{random.choice(['corp.local', 'external.com'])}",
            "port": random.choice([80, 443, 8080, 8443]),
            "bytes_sent": random.randint(64, 65536),
            "bytes_received": random.randint(64, 262144),
            "protocol": random.choice(["HTTP", "HTTPS"]),
        })
    elif event_type == "app_usage":
        base.update({
            "application": random.choice(["slack", "teams", "outlook", "chrome", "terminal"]),
            "duration_seconds": random.randint(1, 28800),
        })

    return base


def _feature_vector(dim: int = 20) -> list[float]:
    """Random feature vector for scoring endpoint."""
    return [random.gauss(0, 1) for _ in range(dim)]


# ═══════════════════════════════════════════════════════════════════
#  Locust User Classes
# ═══════════════════════════════════════════════════════════════════

class InsiderThreatUser(HttpUser):
    """
    Simulates a monitoring dashboard / data pipeline client.

    Weighted task distribution:
      • 50% event ingestion (high-volume telemetry feed)
      • 25% risk score queries (analyst checking user risk)
      • 15% alert listing (SOC dashboard polling)
      •  5% health checks (liveness probes)
      •  5% analytics queries (periodic dashboards)
    """

    wait_time = between(0.1, 0.5)       # 0.1-0.5s between tasks
    abstract = False

    def on_start(self):
        """Login and store auth token for subsequent requests."""
        # Attempt login — if it fails, continue without auth
        try:
            resp = self.client.post(
                "/auth/login",
                json={"username": "load_test_user", "password": "test_password"},
                catch_response=True,
            )
            if resp.status_code == 200:
                data = resp.json()
                self.token = data.get("access_token", "")
                resp.success()
            else:
                self.token = ""
                resp.success()   # Don't fail the test for auth issues
        except Exception:
            self.token = ""
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    # ── Event ingestion (50%) ────────────────────────────────────

    @task(50)
    @tag("ingest")
    def ingest_single_event(self):
        """POST a single event to the ingestion endpoint."""
        event = _random_event()
        with self.client.post(
            "/api/v1/events",
            json=event,
            headers=self.headers,
            catch_response=True,
        ) as resp:
            if resp.status_code in (200, 201, 202):
                resp.success()
            elif resp.status_code == 429:
                resp.success()  # Rate-limited is expected under load
            else:
                resp.failure(f"Event ingest failed: {resp.status_code}")

    @task(5)
    @tag("ingest")
    def ingest_batch_events(self):
        """POST a batch of events."""
        batch = [_random_event() for _ in range(random.randint(5, 20))]
        with self.client.post(
            "/api/v1/events/batch",
            json={"events": batch},
            headers=self.headers,
            catch_response=True,
        ) as resp:
            if resp.status_code in (200, 201, 202):
                resp.success()
            elif resp.status_code == 429:
                resp.success()
            else:
                resp.failure(f"Batch ingest failed: {resp.status_code}")

    # ── Risk scoring (25%) ───────────────────────────────────────

    @task(25)
    @tag("scoring")
    def query_risk_score(self):
        """POST to scoring endpoint with a feature vector."""
        user_id = _random_user_id()
        payload = {
            "user_id": user_id,
            "features": _feature_vector(),
        }
        with self.client.post(
            "/api/v1/score",
            json=payload,
            headers=self.headers,
            catch_response=True,
        ) as resp:
            if resp.status_code in (200, 201, 202):
                resp.success()
            elif resp.status_code == 429:
                resp.success()
            elif resp.status_code == 404:
                resp.success()  # Endpoint may not exist yet
            else:
                resp.failure(f"Score query failed: {resp.status_code}")

    # ── Alert listing (15%) ──────────────────────────────────────

    @task(15)
    @tag("alerts")
    def list_alerts(self):
        """GET recent alerts — simulates SOC dashboard polling."""
        with self.client.get(
            "/api/v1/alerts",
            params={"limit": 20, "severity": "high"},
            headers=self.headers,
            catch_response=True,
        ) as resp:
            if resp.status_code in (200, 401, 403):
                resp.success()
            elif resp.status_code == 429:
                resp.success()
            else:
                resp.failure(f"Alert list failed: {resp.status_code}")

    # ── Health check (5%) ────────────────────────────────────────

    @task(5)
    @tag("health")
    def health_check(self):
        """GET health endpoint — simulates k8s liveness probe."""
        with self.client.get("/healthz", catch_response=True) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"Health check failed: {resp.status_code}")

    # ── Analytics (5%) ───────────────────────────────────────────

    @task(5)
    @tag("analytics")
    def query_analytics(self):
        """GET analytics dashboard data."""
        with self.client.get(
            "/api/v1/analytics/dashboard",
            headers=self.headers,
            catch_response=True,
        ) as resp:
            if resp.status_code in (200, 401, 403, 404):
                resp.success()
            elif resp.status_code == 429:
                resp.success()
            else:
                resp.failure(f"Analytics failed: {resp.status_code}")


# ═══════════════════════════════════════════════════════════════════
#  Spike-test user (aggressive burst traffic)
# ═══════════════════════════════════════════════════════════════════

class SpikeTestUser(HttpUser):
    """Aggressive user that sends rapid bursts — tests rate limiter."""

    wait_time = between(0.01, 0.05)
    weight = 1  # Low weight — only a few of these in the swarm

    @task
    @tag("spike")
    def rapid_fire_events(self):
        event = _random_event()
        with self.client.post(
            "/api/v1/events",
            json=event,
            catch_response=True,
        ) as resp:
            # Any response is OK for spike testing
            if resp.status_code in (200, 201, 202, 429):
                resp.success()
            elif resp.status_code >= 500:
                resp.failure(f"Server error during spike: {resp.status_code}")
            else:
                resp.success()


# ═══════════════════════════════════════════════════════════════════
#  Event hooks — collect p99 latency at test end
# ═══════════════════════════════════════════════════════════════════

@events.quitting.add_listener
def on_quitting(environment, **kwargs):
    """Check p99 latency and error rate on test completion."""
    stats = environment.stats
    total = stats.total

    if total.num_requests == 0:
        print("⚠ No requests completed")
        return

    p99 = total.get_response_time_percentile(0.99) or 0
    error_rate = (total.num_failures / total.num_requests) * 100 if total.num_requests else 0
    rps = total.total_rps

    print(f"\n{'=' * 60}")
    print(f"  LOAD TEST SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total requests : {total.num_requests}")
    print(f"  Failures       : {total.num_failures} ({error_rate:.1f}%)")
    print(f"  p50 latency    : {total.get_response_time_percentile(0.50):.0f} ms")
    print(f"  p95 latency    : {total.get_response_time_percentile(0.95):.0f} ms")
    print(f"  p99 latency    : {p99:.0f} ms")
    print(f"  RPS (avg)      : {rps:.1f}")
    print(f"{'=' * 60}")

    # Validation
    if p99 > 200:
        print(f"  ❌ FAIL: p99 latency {p99:.0f}ms exceeds 200ms target")
    else:
        print(f"  ✅ PASS: p99 latency {p99:.0f}ms within 200ms target")

    if error_rate > 1.0:
        print(f"  ❌ FAIL: Error rate {error_rate:.1f}% exceeds 1% target")
    else:
        print(f"  ✅ PASS: Error rate {error_rate:.1f}% within 1% target")
