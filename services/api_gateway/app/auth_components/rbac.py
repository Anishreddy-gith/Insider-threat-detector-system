"""
Role-Based Access Control (RBAC) Dependencies
===============================================
FastAPI ``Depends()`` callables that enforce per-endpoint access policies.

Usage in route handlers::

    @router.get("/alerts")
    async def list_alerts(
        user: TokenPayload = Depends(require_role(Role.SOC_ANALYST, Role.SOC_MANAGER, Role.ADMIN)),
    ):
        ...

Design decisions
----------------
* **Dependency injection** rather than decorators — integrates cleanly
  with FastAPI's OpenAPI generation and composability model.
* **Role hierarchy is NOT implicit** — each endpoint explicitly lists
  the roles that may access it.  This avoids subtle privilege-escalation
  bugs where a new role accidentally inherits permissions.
* ``TokenPayload`` is a typed Pydantic model so route handlers get
  auto-complete and type safety on user claims.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import Depends, HTTPException, Request, status
from pydantic import BaseModel

from services.api_gateway.app.auth_components.jwt_handler import Role


# ── Token payload schema ──────────────────────────────────────

class TokenPayload(BaseModel):
    """Decoded JWT claims available to route handlers."""
    sub: str          # user UUID
    role: str         # RBAC role
    type: str = "access"
    iat: float | None = None
    exp: float | None = None

    @property
    def user_id(self) -> str:
        return self.sub


# ── Dependency factories ─────────────────────────────────────

def get_current_user(request: Request) -> TokenPayload:
    """
    Extract the authenticated user from ``request.state``.

    The JWT middleware has already decoded the token and stored the
    payload in ``request.state.user``.  This dependency just wraps it
    into a typed model.
    """
    user_data: dict[str, Any] | None = getattr(request.state, "user", None)
    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated.",
        )
    return TokenPayload(**user_data)


class _RoleChecker:
    """
    Callable dependency that verifies the caller has one of the
    allowed roles.

    Instantiated by ``require_role()`` factory below.
    """

    def __init__(self, allowed_roles: tuple[str, ...]) -> None:
        self.allowed_roles = allowed_roles

    def __call__(self, user: TokenPayload = Depends(get_current_user)) -> TokenPayload:
        if user.role not in self.allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"Role '{user.role}' is not authorised for this endpoint. "
                    f"Required: {', '.join(self.allowed_roles)}."
                ),
            )
        return user


def require_role(*roles: str | Role) -> _RoleChecker:
    """
    Factory that returns a FastAPI dependency enforcing RBAC.

    Example::

        @router.post("/thresholds")
        async def update_thresholds(
            user: TokenPayload = Depends(
                require_role(Role.SOC_MANAGER, Role.ADMIN)
            ),
        ):
            ...
    """
    return _RoleChecker(tuple(str(r) for r in roles))

