"""
Cryptographic Utilities
=======================
Thin wrappers around standard-library / well-audited crypto primitives.

Design decisions:
  • bcrypt for password hashing (adaptive cost, resistant to GPU attacks).
  • AES-256-GCM for symmetric encryption of PII at rest — GDPR Art. 32
    requires "encryption of personal data" as a technical measure.
  • HKDF for deterministic pseudonymisation keys derived from a master secret.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
from typing import Final

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes

# Key length constants
AES_KEY_BYTES: Final[int] = 32  # 256-bit
NONCE_BYTES: Final[int] = 12    # 96-bit for AES-GCM (NIST recommendation)


def generate_aes_key() -> bytes:
    """Generate a cryptographically secure 256-bit AES key."""
    return AESGCM.generate_key(bit_length=256)


def encrypt_pii(plaintext: str, key: bytes) -> str:
    """
    Encrypt a PII string with AES-256-GCM.

    Returns a base64-encoded blob: ``nonce || ciphertext || tag``.

    WHY AES-GCM?
      • Authenticated encryption — integrity + confidentiality in one pass.
      • Hardware-accelerated (AES-NI) on modern CPUs.
      • NIST-approved for protecting sensitive data at rest.
    """
    aesgcm = AESGCM(key)
    nonce = os.urandom(NONCE_BYTES)
    ciphertext = aesgcm.encrypt(nonce, plaintext.encode("utf-8"), None)
    return base64.b64encode(nonce + ciphertext).decode("ascii")


def decrypt_pii(token: str, key: bytes) -> str:
    """Decrypt a base64 AES-256-GCM token back to plaintext."""
    raw = base64.b64decode(token)
    nonce, ciphertext = raw[:NONCE_BYTES], raw[NONCE_BYTES:]
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, ciphertext, None).decode("utf-8")


def derive_pseudonym_key(master_secret: bytes, context: str) -> bytes:
    """
    Derive a deterministic 256-bit key for pseudonymising a specific data
    category (e.g. "email", "employee_id").

    Uses HKDF-SHA256 so we never store the master key in the DB — only
    derived keys are kept in memory per context.

    WHY HKDF?
      GDPR Art. 4(5) defines pseudonymisation as processing so that data
      can no longer be attributed to a subject without *additional
      information*.  HKDF lets us separate the master key (stored in a
      vault) from per-field keys (derived at runtime).
    """
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=AES_KEY_BYTES,
        salt=None,  # salt is optional for HKDF; using context as info
        info=context.encode("utf-8"),
    )
    return hkdf.derive(master_secret)


def pseudonymise(value: str, key: bytes) -> str:
    """
    Produce a deterministic pseudonym via HMAC-SHA256.

    WHY deterministic?
      We need the *same* pseudonym for the *same* input so we can join
      across tables without revealing the original identifier.  HMAC
      prevents rainbow-table attacks because the key is secret.
    """
    return hmac.new(key, value.encode("utf-8"), hashlib.sha256).hexdigest()


def generate_secure_token(nbytes: int = 32) -> str:
    """Generate a URL-safe random token (e.g. for password reset links)."""
    return secrets.token_urlsafe(nbytes)
