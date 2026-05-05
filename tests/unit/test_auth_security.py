"""
Unit tests for the Auth service – security module.
"""

from app.security import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    decode_token,
    is_account_locked,
    get_lockout_until,
)
from datetime import datetime, timezone, timedelta


class TestPasswordHashing:
    def test_hash_and_verify(self):
        pw = "SuperSecurePassword123!"
        hashed = hash_password(pw)
        assert hashed != pw
        assert verify_password(pw, hashed)

    def test_wrong_password_fails(self):
        hashed = hash_password("correct-password")
        assert not verify_password("wrong-password", hashed)

    def test_hash_is_unique(self):
        """bcrypt salts should make each hash unique."""
        h1 = hash_password("same-password")
        h2 = hash_password("same-password")
        assert h1 != h2  # different salts


class TestJWT:
    def test_access_token_roundtrip(self):
        token, expires_in = create_access_token(
            user_id="user-123", role="analyst", username="jdoe"
        )
        assert expires_in > 0
        claims = decode_token(token)
        assert claims is not None
        assert claims["sub"] == "user-123"
        assert claims["role"] == "analyst"
        assert claims["type"] == "access"

    def test_refresh_token_family(self):
        token, family = create_refresh_token("user-456")
        claims = decode_token(token)
        assert claims is not None
        assert claims["type"] == "refresh"
        assert claims["family"] == family

    def test_invalid_token_returns_none(self):
        assert decode_token("not-a-real-token") is None

    def test_refresh_token_preserves_family(self):
        _, family = create_refresh_token("user-789", family="my-family")
        assert family == "my-family"


class TestAccountLockout:
    def test_not_locked_when_none(self):
        assert not is_account_locked(None)

    def test_locked_when_future(self):
        future = datetime.now(timezone.utc) + timedelta(minutes=30)
        assert is_account_locked(future)

    def test_not_locked_when_past(self):
        past = datetime.now(timezone.utc) - timedelta(minutes=1)
        assert not is_account_locked(past)

    def test_lockout_duration(self):
        lockout = get_lockout_until()
        assert lockout > datetime.now(timezone.utc)
