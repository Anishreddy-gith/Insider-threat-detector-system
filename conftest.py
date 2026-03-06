"""
Root conftest.py — ensures services are importable by all tests.

Placed at the repository root so pytest processes it before any
test module is imported.

IMPORTANT: Only ml-engine is added here because both ml-engine and
api-gateway have an `app/` package.  Adding both causes a namespace
collision.  Integration tests that need api-gateway should add it
to sys.path in their own conftest.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# Order matters — ml-engine's `app` must shadow any other `app` package
for sub in ("", "services/ml-engine", "shared"):
    path = str(REPO_ROOT / sub)
    if path not in sys.path:
        sys.path.insert(0, path)
