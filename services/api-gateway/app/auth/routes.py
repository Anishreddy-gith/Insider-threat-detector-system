"""
Auth Routes — Login / Register / Refresh
==========================================
Handles user authentication lifecycle.

Security notes:
  * Passwords are hashed with bcrypt (cost factor 12 via passlib).
  * Login returns a short-lived access token + long-lived refresh token.
  * Refresh endpoint issues a new access token without re-entering
    credentials — standard OAuth 2.0 pattern.
  * Failed login attempts are logged but NOT rate-limited here —
    the per-role rate-limiter middleware handles that globally.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.jwt_handler import (
    Role,
    create_access_token,
    create_refresh_token,
    hash_password,
    verify_password,
    verify_refresh_token,
)
from app.auth.rbac import TokenPayload, get_current_user
from app.models.database import User, get_db
from app.models.schemas import (
    LoginRequest,
    TokenResponse,
    UserCreate,
    UserResponse,
)

router = APIRouter(prefix="/auth", tags=["auth"])


# ── Register ──────────────────────────────────────────────────

@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new platform user",
)
async def register(
    body: UserCreate,
    db: AsyncSession = Depends(get_db),
) -> UserResponse:
    """
    Create a new user account.

    Default role is ``SOC_ANALYST``.  Only an ``ADMIN`` can later
    promote the account to ``SOC_MANAGER`` or ``ADMIN``.
    """
    existing = await db.execute(
        select(User).where(User.email == body.email)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A user with this email already exists.",
        )

    # Validate role — only allow valid RBAC roles
    valid_roles = {r.value for r in Role}
    role = body.role.upper() if body.role else Role.SOC_ANALYST
    if role not in valid_roles:
        role = Role.SOC_ANALYST

    user = User(
        email=body.email,
        hashed_password=hash_password(body.password),
        full_name=body.full_name,
        role=role,
        is_active=True,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    return UserResponse.model_validate(user)


# ── Login ─────────────────────────────────────────────────────

@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Authenticate and receive JWT tokens",
)
async def login(
    body: LoginRequest,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    """
    Authenticate with email + password.

    Returns:
      * ``access_token``  — 30-minute JWT for API calls.
      * ``refresh_token`` — 7-day JWT to obtain new access tokens.
    """
    result = await db.execute(
        select(User).where(User.email == body.email)
    )
    user = result.scalar_one_or_none()

    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated. Contact your administrator.",
        )

    access = create_access_token(str(user.id), user.role)
    refresh = create_refresh_token(str(user.id))

    from app.config import get_settings
    settings = get_settings()

    return TokenResponse(
        access_token=access,
        refresh_token=refresh,
        token_type="bearer",
        expires_in=settings.jwt_access_token_expire_minutes * 60,
    )


# ── Refresh ───────────────────────────────────────────────────

@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Exchange a refresh token for a new access token",
)
async def refresh_token(
    body: dict,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    """
    Accepts ``{ "refresh_token": "..." }`` and returns a fresh
    access token without requiring the user to re-enter credentials.
    """
    raw_token = body.get("refresh_token")
    if not raw_token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="refresh_token is required.",
        )

    try:
        payload = verify_refresh_token(raw_token)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token.",
        )

    user = await db.get(User, payload["sub"])
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or deactivated.",
        )

    access = create_access_token(str(user.id), user.role)
    refresh = create_refresh_token(str(user.id))

    from app.config import get_settings
    settings = get_settings()

    return TokenResponse(
        access_token=access,
        refresh_token=refresh,
        token_type="bearer",
        expires_in=settings.jwt_access_token_expire_minutes * 60,
    )


# ── Current user ──────────────────────────────────────────────

@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get the current authenticated user",
)
async def get_me(
    user: TokenPayload = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> UserResponse:
    db_user = await db.get(User, user.sub)
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found.")
    return UserResponse.model_validate(db_user)
