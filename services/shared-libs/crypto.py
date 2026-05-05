"""
Cryptographic utilities.

- AES-256-GCM authenticated encryption for data at rest
- HMAC-SHA256 pseudonymisation for PII fields
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


# ── AES-256-GCM ─────────────────────────────────────────────────────
def generate_key() -> bytes:
    """Return a 256-bit key suitable for AES-256-GCM."""
    return AESGCM.generate_key(bit_length=256)


def encrypt(plaintext: bytes, key: bytes, aad: bytes | None = None) -> bytes:
    """
    Encrypt *plaintext* with AES-256-GCM.

    Returns ``nonce || ciphertext`` (12-byte nonce prepended).
    """
    nonce = os.urandom(12)
    ct = AESGCM(key).encrypt(nonce, plaintext, aad)
    return nonce + ct


def decrypt(blob: bytes, key: bytes, aad: bytes | None = None) -> bytes:
    """
    Decrypt a ``nonce || ciphertext`` blob produced by :func:`encrypt`.
    """
    nonce, ct = blob[:12], blob[12:]
    return AESGCM(key).decrypt(nonce, ct, aad)


# ── Pseudonymisation ────────────────────────────────────────────────
def pseudonymise(value: str, secret: str) -> str:
    """
    HMAC-SHA256 pseudonymisation.

    Given a PII string (e.g. email, employee-id) and a secret key,
    returns a deterministic but irreversible hex digest.
    """
    return hmac.new(
        secret.encode(),
        value.encode(),
        hashlib.sha256,
    ).hexdigest()
