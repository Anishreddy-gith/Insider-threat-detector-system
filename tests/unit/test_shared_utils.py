"""
Unit tests for shared utilities — crypto, logging, constants, events.
"""

from __future__ import annotations

import os
import pytest


# ═════════════════════════════════════════════════════════════════════════
# Constants
# ═════════════════════════════════════════════════════════════════════════

class TestConstants:
    """Verify shared constant definitions are consistent and complete."""

    def test_kafka_topics_prefixed(self):
        from shared.constants import (
            TOPIC_USER_ACTIVITY,
            TOPIC_FILE_ACCESS,
            TOPIC_NETWORK_EVENT,
            TOPIC_AUTH_EVENT,
            TOPIC_ANOMALY_DETECTED,
            TOPIC_RISK_SCORE,
            TOPIC_ALERTS,
        )
        for topic in (
            TOPIC_USER_ACTIVITY,
            TOPIC_FILE_ACCESS,
            TOPIC_NETWORK_EVENT,
            TOPIC_AUTH_EVENT,
            TOPIC_ANOMALY_DETECTED,
            TOPIC_RISK_SCORE,
            TOPIC_ALERTS,
        ):
            assert topic.startswith("itds."), f"{topic} should start with 'itds.'"

    def test_risk_thresholds_cover_full_range(self):
        from shared.constants import RISK_THRESHOLDS, RISK_LOW, RISK_CRITICAL
        assert RISK_THRESHOLDS[RISK_LOW][0] == 0.0
        assert RISK_THRESHOLDS[RISK_CRITICAL][1] == 100.0
        # Check contiguity
        prev_upper = 0.0
        for level in (RISK_LOW, "medium", "high", RISK_CRITICAL):
            lo, hi = RISK_THRESHOLDS[level]
            assert lo == prev_upper, f"Gap at {level}: expected {prev_upper}, got {lo}"
            prev_upper = hi

    def test_risk_levels_are_lowercase(self):
        from shared.constants import RISK_LOW, RISK_MEDIUM, RISK_HIGH, RISK_CRITICAL
        for level in (RISK_LOW, RISK_MEDIUM, RISK_HIGH, RISK_CRITICAL):
            assert level == level.lower()

    def test_rbac_roles_defined(self):
        from shared.constants import ROLE_ADMIN, ROLE_ANALYST, ROLE_VIEWER, ROLE_AUDITOR
        roles = {ROLE_ADMIN, ROLE_ANALYST, ROLE_VIEWER, ROLE_AUDITOR}
        assert len(roles) == 4

    def test_sensitivity_labels(self):
        from shared.constants import (
            SENSITIVITY_PUBLIC,
            SENSITIVITY_INTERNAL,
            SENSITIVITY_CONFIDENTIAL,
            SENSITIVITY_RESTRICTED,
        )
        labels = [SENSITIVITY_PUBLIC, SENSITIVITY_INTERNAL, SENSITIVITY_CONFIDENTIAL, SENSITIVITY_RESTRICTED]
        assert len(set(labels)) == 4

    def test_redis_key_prefixes_unique(self):
        from shared.constants import (
            REDIS_PREFIX_SESSION,
            REDIS_PREFIX_RISK,
            REDIS_PREFIX_RATE_LIMIT,
            REDIS_PREFIX_FEATURE_CACHE,
        )
        prefixes = [REDIS_PREFIX_SESSION, REDIS_PREFIX_RISK, REDIS_PREFIX_RATE_LIMIT, REDIS_PREFIX_FEATURE_CACHE]
        assert len(set(prefixes)) == 4

    def test_pagination_defaults(self):
        from shared.constants import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE
        assert DEFAULT_PAGE_SIZE > 0
        assert MAX_PAGE_SIZE > DEFAULT_PAGE_SIZE


# ═════════════════════════════════════════════════════════════════════════
# Crypto
# ═════════════════════════════════════════════════════════════════════════

class TestCrypto:
    """AES-256-GCM encryption and HMAC pseudonymisation tests."""

    def test_generate_aes_key_length(self):
        from shared.utils.crypto import generate_aes_key, AES_KEY_BYTES
        key = generate_aes_key()
        assert len(key) == AES_KEY_BYTES

    def test_encrypt_decrypt_roundtrip(self):
        from shared.utils.crypto import encrypt_pii, decrypt_pii, generate_aes_key
        key = generate_aes_key()
        plaintext = "alice.johnson@corp.example.com"
        ciphertext = encrypt_pii(plaintext, key)
        assert ciphertext != plaintext
        assert decrypt_pii(ciphertext, key) == plaintext

    def test_encrypt_produces_different_ciphertexts(self):
        """Each encryption should use a unique nonce → different ciphertext."""
        from shared.utils.crypto import encrypt_pii, generate_aes_key
        key = generate_aes_key()
        c1 = encrypt_pii("test", key)
        c2 = encrypt_pii("test", key)
        assert c1 != c2  # random nonce

    def test_decrypt_with_wrong_key_fails(self):
        from shared.utils.crypto import encrypt_pii, decrypt_pii, generate_aes_key
        key1, key2 = generate_aes_key(), generate_aes_key()
        ct = encrypt_pii("secret-data", key1)
        with pytest.raises(Exception):
            decrypt_pii(ct, key2)

    def test_pseudonymise_deterministic(self):
        from shared.utils.crypto import pseudonymise
        key = os.urandom(32)
        p1 = pseudonymise("user-42", key)
        p2 = pseudonymise("user-42", key)
        assert p1 == p2

    def test_pseudonymise_different_inputs_differ(self):
        from shared.utils.crypto import pseudonymise
        key = os.urandom(32)
        assert pseudonymise("user-A", key) != pseudonymise("user-B", key)

    def test_pseudonymise_hex_output(self):
        from shared.utils.crypto import pseudonymise
        key = os.urandom(32)
        result = pseudonymise("hello", key)
        assert len(result) == 64  # SHA256 → 64 hex chars
        int(result, 16)  # should not raise

    def test_derive_pseudonym_key(self):
        from shared.utils.crypto import derive_pseudonym_key, AES_KEY_BYTES
        master = os.urandom(32)
        k1 = derive_pseudonym_key(master, "email")
        k2 = derive_pseudonym_key(master, "employee_id")
        assert len(k1) == AES_KEY_BYTES
        assert k1 != k2  # different contexts → different derived keys

    def test_generate_secure_token(self):
        from shared.utils.crypto import generate_secure_token
        t1 = generate_secure_token()
        t2 = generate_secure_token()
        assert t1 != t2
        assert len(t1) > 20


# ═════════════════════════════════════════════════════════════════════════
# Logging
# ═════════════════════════════════════════════════════════════════════════

class TestLogging:
    """Structured logging setup tests."""

    def test_get_logger_returns_bound_logger(self):
        from shared.utils.logging import get_logger
        log = get_logger("test_module")
        assert log is not None

    def test_correlation_id_context_var(self):
        from shared.utils.logging import correlation_id_ctx
        assert correlation_id_ctx.get() is None  # default
        token = correlation_id_ctx.set("test-correlation-123")
        assert correlation_id_ctx.get() == "test-correlation-123"
        correlation_id_ctx.reset(token)

    def test_sensitive_redaction(self):
        from shared.utils.logging import _redact_sensitive
        event = {"password": "s3cret", "user_id": "u-42", "api_key": "abc123"}
        result = _redact_sensitive(None, None, event)
        assert result["password"] == "***REDACTED***"
        assert result["api_key"] == "***REDACTED***"
        assert result["user_id"] == "u-42"  # not masked

    def test_correlation_id_injection(self):
        from shared.utils.logging import _inject_correlation_id, correlation_id_ctx
        token = correlation_id_ctx.set("cid-999")
        event: dict = {"event": "login"}
        result = _inject_correlation_id(None, None, event)
        assert result["correlation_id"] == "cid-999"
        correlation_id_ctx.reset(token)


# ═════════════════════════════════════════════════════════════════════════
# Shared Event Schemas
# ═════════════════════════════════════════════════════════════════════════

class TestEventSchemas:
    """Validate shared Kafka event envelope and payload models."""

    def test_event_type_enum_members(self):
        from shared.models.events import EventType
        assert EventType.USER_ACTIVITY == "user.activity"
        assert EventType.ANOMALY_DETECTED == "anomaly.detected"
        assert EventType.ALERT_CREATED == "alert.created"

    def test_event_envelope_defaults(self):
        from shared.models.events import EventEnvelope, EventType
        env = EventEnvelope(
            event_type=EventType.USER_ACTIVITY,
            source_service="test",
            payload={"foo": "bar"},
        )
        assert env.event_id  # auto-generated UUID
        assert env.correlation_id
        assert env.timestamp is not None
        assert env.payload == {"foo": "bar"}

    def test_user_activity_payload(self):
        from shared.models.events import UserActivityPayload
        p = UserActivityPayload(
            user_id="u-1",
            session_id="s-1",
            action="login",
        )
        assert p.user_id == "u-1"
        assert p.metadata == {}

    def test_file_access_payload(self):
        from shared.models.events import FileAccessPayload
        p = FileAccessPayload(
            user_id="u-2",
            file_path="/docs/secret.pdf",
            action="read",
            sensitivity_label="confidential",
        )
        assert p.sensitivity_label == "confidential"
        assert p.destination is None

    def test_network_event_payload(self):
        from shared.models.events import NetworkEventPayload
        p = NetworkEventPayload(
            src_ip="10.0.0.1",
            dst_ip="203.0.113.5",
            dst_port=443,
            protocol="TCP",
            bytes_sent=1024,
        )
        assert p.is_encrypted is True  # default

    def test_anomaly_payload_score_bounds(self):
        from shared.models.events import AnomalyPayload
        with pytest.raises(Exception):
            AnomalyPayload(
                user_id="u-3",
                anomaly_score=1.5,  # exceeds max
                model_name="test",
                model_version="v1",
            )

    def test_risk_score_payload_bounds(self):
        from shared.models.events import RiskScorePayload
        p = RiskScorePayload(
            user_id="u-4",
            risk_score=75.0,
            risk_level="high",
        )
        assert p.score_delta is None

    def test_alert_payload_defaults(self):
        from shared.models.events import AlertPayload
        a = AlertPayload(
            user_id="u-5",
            severity="critical",
            title="Bulk download",
            description="User downloaded 500 files",
            risk_score=92.0,
        )
        assert a.is_resolved is False
        assert a.indicators == []
        assert a.recommended_actions == []
