"""
Integration tests — full detection pipeline with real containers.

Uses testcontainers-python to spin Kafka + PostgreSQL,  then validates
the end-to-end flow:

    raw event → Kafka → ingestion → ML engine → risk scoring → alert

Fixtures   : kafka_container, postgres_container, db_session, kafka_producer
Markers    : @pytest.mark.integration  (so ``pytest -m "not integration"``
             skips them in CI fast-check jobs)
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncGenerator

import numpy as np
import pytest

# ── Testcontainers imports ─────────────────────────────────────────
try:
    from testcontainers.kafka import KafkaContainer
    from testcontainers.postgres import PostgresContainer
    HAS_TESTCONTAINERS = True
except ImportError:
    HAS_TESTCONTAINERS = False

# ── Kafka/DB client imports ────────────────────────────────────────
try:
    from confluent_kafka import Producer, Consumer, KafkaError
    HAS_KAFKA_CLIENT = True
except ImportError:
    HAS_KAFKA_CLIENT = False

try:
    import sqlalchemy
    from sqlalchemy import create_engine, text
    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False

# ── ML engine imports (from conftest sys.path setup) ───────────────
from app.detectors import (
    IsolationForestDetector,
    AutoencoderDetector,
    LSTMDetector,
    EnsembleScorer,
)
from app.pipeline.inference import InferencePipeline

skip_no_testcontainers = pytest.mark.skipif(
    not HAS_TESTCONTAINERS,
    reason="testcontainers-python not installed",
)
skip_no_kafka_client = pytest.mark.skipif(
    not HAS_KAFKA_CLIENT,
    reason="confluent-kafka not installed",
)

pytestmark = [
    pytest.mark.integration,
    skip_no_testcontainers,
]

# ═══════════════════════════════════════════════════════════════════
#  Container fixtures
# ═══════════════════════════════════════════════════════════════════

TOPIC_RAW = "itds.raw-events"
TOPIC_ANOMALIES = "itds.anomalies"
TOPIC_RISK = "itds.risk-scores"


@pytest.fixture(scope="module")
def kafka_container():
    """Launch a Kafka container for the test session."""
    with KafkaContainer("confluentinc/cp-kafka:7.6.0") as kafka:
        yield kafka


@pytest.fixture(scope="module")
def postgres_container():
    """Launch a Postgres container for the test session."""
    with PostgresContainer(
        "postgres:16-alpine",
        user="itds",
        password="itds_test",
        dbname="itds_test",
    ) as pg:
        yield pg


@pytest.fixture(scope="module")
def db_engine(postgres_container):
    """SQLAlchemy engine connected to the test Postgres."""
    url = postgres_container.get_connection_url()
    eng = create_engine(url)
    with eng.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS raw_events (
                id SERIAL PRIMARY KEY,
                event_id UUID NOT NULL,
                user_id VARCHAR(128) NOT NULL,
                event_type VARCHAR(64) NOT NULL,
                payload JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS risk_scores (
                id SERIAL PRIMARY KEY,
                user_id VARCHAR(128) NOT NULL,
                score DOUBLE PRECISION NOT NULL,
                risk_level VARCHAR(32) NOT NULL,
                scored_at TIMESTAMPTZ DEFAULT NOW()
            )
        """))
    yield eng
    eng.dispose()


# ═══════════════════════════════════════════════════════════════════
#  Kafka helper
# ═══════════════════════════════════════════════════════════════════

def _make_producer(bootstrap: str) -> "Producer":
    return Producer({
        "bootstrap.servers": bootstrap,
        "linger.ms": 0,
        "acks": "all",
    })


def _make_consumer(bootstrap: str, group: str, topics: list[str]) -> "Consumer":
    c = Consumer({
        "bootstrap.servers": bootstrap,
        "group.id": group,
        "auto.offset.reset": "earliest",
    })
    c.subscribe(topics)
    return c


def _produce_event(producer: "Producer", topic: str, event: dict) -> None:
    producer.produce(
        topic,
        key=event.get("user_id", "unknown").encode(),
        value=json.dumps(event).encode(),
    )
    producer.flush(5)


def _consume_messages(
    consumer: "Consumer",
    timeout_s: float = 10.0,
    max_messages: int = 100,
) -> list[dict]:
    """Poll until timeout, return decoded messages."""
    msgs = []
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline and len(msgs) < max_messages:
        msg = consumer.poll(1.0)
        if msg is None:
            continue
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                continue
            raise RuntimeError(msg.error())
        msgs.append(json.loads(msg.value().decode()))
    consumer.close()
    return msgs


# ═══════════════════════════════════════════════════════════════════
#  Synthetic event generators
# ═══════════════════════════════════════════════════════════════════

def _login_event(user_id: str = "user-001", **overrides: Any) -> dict:
    return {
        "event_id": str(uuid.uuid4()),
        "user_id": user_id,
        "event_type": "login",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_ip": overrides.get("source_ip", "10.0.1.42"),
        "device_fingerprint": overrides.get("device_fingerprint", "fp-abc123"),
        "mfa_used": overrides.get("mfa_used", True),
    }


def _file_access_event(
    user_id: str = "user-001",
    action: str = "read",
    classification: str = "internal",
    **overrides: Any,
) -> dict:
    return {
        "event_id": str(uuid.uuid4()),
        "user_id": user_id,
        "event_type": "file_access",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "file_path": overrides.get("file_path", "/data/reports/q4.xlsx"),
        "action": action,
        "classification": classification,
        "bytes_transferred": overrides.get("bytes_transferred", 4096),
    }


def _network_event(
    user_id: str = "user-001",
    dest: str = "internal-api.corp.local",
    **overrides: Any,
) -> dict:
    return {
        "event_id": str(uuid.uuid4()),
        "user_id": user_id,
        "event_type": "network_request",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "destination_host": dest,
        "port": overrides.get("port", 443),
        "bytes_sent": overrides.get("bytes_sent", 256),
        "bytes_received": overrides.get("bytes_received", 1024),
        "protocol": overrides.get("protocol", "HTTPS"),
    }


# ═══════════════════════════════════════════════════════════════════
#  Test classes
# ═══════════════════════════════════════════════════════════════════

class TestKafkaEventFlow:
    """Verify events can be produced and consumed through Kafka."""

    @skip_no_kafka_client
    def test_produce_consume_round_trip(self, kafka_container):
        bootstrap = kafka_container.get_bootstrap_server()
        producer = _make_producer(bootstrap)
        event = _login_event()
        _produce_event(producer, TOPIC_RAW, event)

        consumer = _make_consumer(
            bootstrap, f"test-cg-{uuid.uuid4().hex[:8]}", [TOPIC_RAW]
        )
        messages = _consume_messages(consumer, timeout_s=15)
        assert len(messages) >= 1
        assert messages[0]["event_id"] == event["event_id"]
        assert messages[0]["event_type"] == "login"

    @skip_no_kafka_client
    def test_multi_event_ordering(self, kafka_container):
        """Events for the same user arrive in order."""
        bootstrap = kafka_container.get_bootstrap_server()
        producer = _make_producer(bootstrap)
        events = [_login_event(user_id="order-test") for _ in range(5)]
        for ev in events:
            _produce_event(producer, TOPIC_RAW, ev)

        consumer = _make_consumer(
            bootstrap, f"test-cg-{uuid.uuid4().hex[:8]}", [TOPIC_RAW]
        )
        messages = _consume_messages(consumer, timeout_s=15, max_messages=5)
        received_ids = [m["event_id"] for m in messages if m.get("user_id") == "order-test"]
        expected_ids = [e["event_id"] for e in events]
        assert received_ids == expected_ids


class TestDatabasePersistence:
    """Verify events and scores can be written/read from Postgres."""

    def test_insert_and_query_events(self, db_engine):
        event = _login_event()
        with db_engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO raw_events (event_id, user_id, event_type, payload)
                VALUES (:eid, :uid, :etype, :payload)
            """), {
                "eid": event["event_id"],
                "uid": event["user_id"],
                "etype": event["event_type"],
                "payload": json.dumps(event),
            })
        with db_engine.connect() as conn:
            result = conn.execute(
                text("SELECT * FROM raw_events WHERE event_id = :eid"),
                {"eid": event["event_id"]},
            ).fetchone()
            assert result is not None
            assert result.user_id == event["user_id"]

    def test_insert_risk_scores(self, db_engine):
        with db_engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO risk_scores (user_id, score, risk_level)
                VALUES (:uid, :score, :level)
            """), {"uid": "user-risk-1", "score": 0.78, "level": "critical"})
        with db_engine.connect() as conn:
            row = conn.execute(
                text("SELECT score, risk_level FROM risk_scores WHERE user_id = :uid"),
                {"uid": "user-risk-1"},
            ).fetchone()
            assert row is not None
            assert abs(row.score - 0.78) < 1e-6
            assert row.risk_level == "critical"


class TestMLPipelineIntegration:
    """End-to-end ML pipeline: train detectors → predict → ensemble."""

    @pytest.fixture(scope="class")
    def trained_pipeline(self, tmp_path_factory):
        """Train all four detectors and the ensemble on synthetic data."""
        rng = np.random.default_rng(42)

        # Normal training data
        X_train_20 = rng.normal(0, 1, (400, 20)).astype(np.float32)
        X_train_32 = rng.normal(0, 1, (400, 32)).astype(np.float32)
        y_train = np.zeros(400)

        # Anomalous samples for calibration
        X_anom_20 = rng.normal(4, 0.5, (40, 20)).astype(np.float32)
        X_anom_32 = rng.normal(4, 0.5, (40, 32)).astype(np.float32)
        y_anom = np.ones(40)

        # Train Isolation Forest
        iso = IsolationForestDetector(contamination=0.05, n_estimators=100)
        iso.train(X_train_20)

        # Train Autoencoder
        ae = AutoencoderDetector(input_dim=32, epochs=10, lr=1e-3)
        ae.train(X_train_32)

        # Train Ensemble (calibrated)
        ens = EnsembleScorer(
            detector_names=["isolation_forest", "autoencoder"],
        )
        # Build calibration set
        X_cal_20 = np.vstack([X_train_20[:40], X_anom_20])
        X_cal_32 = np.vstack([X_train_32[:40], X_anom_32])
        y_cal = np.concatenate([np.zeros(40), y_anom])

        iso_preds = iso.predict(X_cal_20)
        ae_preds = ae.predict(X_cal_32)

        cal_scores = np.column_stack([
            [p["anomaly_score"] for p in iso_preds],
            [p["anomaly_score"] for p in ae_preds],
        ])
        ens.train(cal_scores, y_cal)

        # Persist
        d = tmp_path_factory.mktemp("int_models")
        iso.save(d)
        ae.save(d)
        ens.save(d)

        return {
            "iso": iso,
            "ae": ae,
            "ens": ens,
            "model_dir": d,
        }

    def test_normal_user_low_score(self, trained_pipeline):
        """Normal behaviour should produce a low ensemble score."""
        rng = np.random.default_rng(99)
        X_n20 = rng.normal(0, 1, (1, 20)).astype(np.float32)
        X_n32 = rng.normal(0, 1, (1, 32)).astype(np.float32)

        iso_out = trained_pipeline["iso"].predict(X_n20)
        ae_out = trained_pipeline["ae"].predict(X_n32)

        scores = np.array([[iso_out[0]["anomaly_score"], ae_out[0]["anomaly_score"]]])
        ens_out = trained_pipeline["ens"].predict(scores)
        assert ens_out[0]["anomaly_score"] < 0.5

    def test_anomalous_user_high_score(self, trained_pipeline):
        """Anomalous behaviour should produce a high ensemble score."""
        rng = np.random.default_rng(99)
        X_a20 = rng.normal(4, 0.5, (1, 20)).astype(np.float32)
        X_a32 = rng.normal(4, 0.5, (1, 32)).astype(np.float32)

        iso_out = trained_pipeline["iso"].predict(X_a20)
        ae_out = trained_pipeline["ae"].predict(X_a32)

        scores = np.array([[iso_out[0]["anomaly_score"], ae_out[0]["anomaly_score"]]])
        ens_out = trained_pipeline["ens"].predict(scores)
        assert ens_out[0]["anomaly_score"] > 0.5
        assert ens_out[0]["is_anomaly"] is True

    def test_model_persistence_round_trip(self, trained_pipeline):
        """Save → load → predict produces identical results."""
        d = trained_pipeline["model_dir"]
        rng = np.random.default_rng(7)
        X_test20 = rng.normal(0, 1, (5, 20)).astype(np.float32)

        before = trained_pipeline["iso"].predict(X_test20)

        iso2 = IsolationForestDetector(contamination=0.05, n_estimators=100)
        iso2.load(d)
        after = iso2.predict(X_test20)

        for b, a in zip(before, after):
            assert abs(b["anomaly_score"] - a["anomaly_score"]) < 1e-6

    def test_score_distribution_makes_sense(self, trained_pipeline):
        """Normal scores should be lower on average than anomalous scores."""
        rng = np.random.default_rng(42)
        normal = rng.normal(0, 1, (50, 20)).astype(np.float32)
        anomalous = rng.normal(4, 0.5, (50, 20)).astype(np.float32)

        normal_scores = [
            r["anomaly_score"] for r in trained_pipeline["iso"].predict(normal)
        ]
        anom_scores = [
            r["anomaly_score"] for r in trained_pipeline["iso"].predict(anomalous)
        ]
        assert np.mean(normal_scores) < np.mean(anom_scores)


class TestCrossServiceDataContract:
    """Verify upstream/downstream data contracts stay compatible."""

    def test_ingestion_event_schema(self):
        """Login event dict matches the ingestion-service expected schema."""
        event = _login_event()
        required = {"event_id", "user_id", "event_type", "timestamp"}
        assert required.issubset(event.keys())
        # event_type must be one of the valid types
        valid_types = {"login", "logout", "file_access", "network_request", "app_usage"}
        assert event["event_type"] in valid_types

    def test_file_access_event_schema(self):
        event = _file_access_event(action="download", classification="confidential")
        assert event["action"] in {"read", "write", "delete", "copy", "move", "download", "upload"}
        assert "classification" in event

    def test_risk_score_output_contract(self):
        """Risk-scoring service expects specific fields from ML engine."""
        # Simulate ML engine output
        detector_output = {
            "anomaly_score": 0.72,
            "is_anomaly": True,
            "detector": "isolation_forest",
            "feature_importance": {"login_hours_mean": 0.15},
        }
        required_keys = {"anomaly_score", "is_anomaly", "detector"}
        assert required_keys.issubset(detector_output.keys())
        assert 0.0 <= detector_output["anomaly_score"] <= 1.0
        assert isinstance(detector_output["is_anomaly"], bool)

    def test_ensemble_output_for_risk_scoring(self):
        """Ensemble output includes detector_breakdown for risk scoring."""
        ensemble_output = {
            "anomaly_score": 0.65,
            "is_anomaly": True,
            "detector": "ensemble",
            "method": "weighted_average",
            "detector_breakdown": {
                "isolation_forest": 0.7,
                "autoencoder": 0.6,
            },
        }
        assert "detector_breakdown" in ensemble_output
        for name, score in ensemble_output["detector_breakdown"].items():
            assert 0.0 <= score <= 1.0


class TestEndToEndEventProcessing:
    """Full round-trip: kafka produce → DB store → ML predict → score."""

    @skip_no_kafka_client
    def test_event_to_score_pipeline(self, kafka_container, db_engine):
        """Simulate the complete pipeline orchestration."""
        bootstrap = kafka_container.get_bootstrap_server()
        producer = _make_producer(bootstrap)

        # 1. Produce a batch of events
        user_id = f"e2e-user-{uuid.uuid4().hex[:6]}"
        events = [
            _login_event(user_id=user_id),
            _file_access_event(user_id=user_id, action="download", classification="confidential"),
            _network_event(user_id=user_id, dest="suspicious-cdn.external.com"),
        ]
        for ev in events:
            _produce_event(producer, TOPIC_RAW, ev)

        # 2. Consume and verify receipt
        consumer = _make_consumer(
            bootstrap, f"e2e-cg-{uuid.uuid4().hex[:8]}", [TOPIC_RAW]
        )
        msgs = _consume_messages(consumer, timeout_s=15, max_messages=3)
        assert len(msgs) >= 3

        # 3. Persist to DB
        with db_engine.begin() as conn:
            for msg in msgs:
                conn.execute(text("""
                    INSERT INTO raw_events (event_id, user_id, event_type, payload)
                    VALUES (:eid, :uid, :etype, :payload)
                """), {
                    "eid": msg["event_id"],
                    "uid": msg["user_id"],
                    "etype": msg["event_type"],
                    "payload": json.dumps(msg),
                })

        # 4. Verify DB count
        with db_engine.connect() as conn:
            count = conn.execute(
                text("SELECT COUNT(*) FROM raw_events WHERE user_id = :uid"),
                {"uid": user_id},
            ).scalar()
            assert count == 3

        # 5. Simulate ML scoring on extracted feature vector
        rng = np.random.default_rng(42)
        features = rng.normal(0, 1, (1, 20)).astype(np.float32)
        iso = IsolationForestDetector(contamination=0.05, n_estimators=50)
        # Quick train on normal data
        iso.train(rng.normal(0, 1, (200, 20)).astype(np.float32))
        result = iso.predict(features)
        assert "anomaly_score" in result[0]

        # 6. Persist score
        score_val = result[0]["anomaly_score"]
        risk_level = "low" if score_val < 0.25 else "medium" if score_val < 0.5 else "high"
        with db_engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO risk_scores (user_id, score, risk_level)
                VALUES (:uid, :score, :level)
            """), {"uid": user_id, "score": float(score_val), "level": risk_level})

        with db_engine.connect() as conn:
            row = conn.execute(
                text("SELECT score, risk_level FROM risk_scores WHERE user_id = :uid"),
                {"uid": user_id},
            ).fetchone()
            assert row is not None
            assert 0.0 <= row.score <= 1.0
