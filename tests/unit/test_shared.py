"""
Unit tests for shared utilities – crypto, logging, constants.
"""

import uuid
from shared.utils.crypto import encrypt_pii, decrypt_pii, pseudonymise
from shared.constants import RiskLevel, KafkaTopics


class TestCrypto:
    """AES-256-GCM encryption round-trip tests."""

    def test_encrypt_decrypt_roundtrip(self):
        key = "a" * 32  # 32-byte key
        plaintext = "john.doe@company.com"
        ciphertext = encrypt_pii(plaintext, key)
        assert ciphertext != plaintext
        assert decrypt_pii(ciphertext, key) == plaintext

    def test_different_keys_fail(self):
        key1 = "a" * 32
        key2 = "b" * 32
        ciphertext = encrypt_pii("secret", key1)
        try:
            result = decrypt_pii(ciphertext, key2)
            # Should either raise or return garbage
            assert result != "secret"
        except Exception:
            pass  # Expected – decryption with wrong key should fail

    def test_pseudonymise_deterministic(self):
        """Same input → same pseudonym (required for joins across services)."""
        key = "salt-key-for-testing"
        p1 = pseudonymise("user123", key)
        p2 = pseudonymise("user123", key)
        assert p1 == p2
        assert p1 != "user123"

    def test_pseudonymise_different_inputs(self):
        key = "salt-key"
        p1 = pseudonymise("user1", key)
        p2 = pseudonymise("user2", key)
        assert p1 != p2


class TestConstants:
    def test_risk_levels(self):
        assert RiskLevel.LOW == "low"
        assert RiskLevel.CRITICAL == "critical"

    def test_kafka_topics_prefixed(self):
        assert KafkaTopics.RAW_EVENTS.startswith("itds.")
        assert KafkaTopics.ANOMALIES.startswith("itds.")
