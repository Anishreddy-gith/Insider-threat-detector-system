"""
Auth route handlers â€“ login, register, refresh, password change, current user.

Security controls:
- Account lockout after N failed attempts (OWASP ASVS Â§2.2.1)
- Refresh-token rotation with family-based revocation
- Audit logging on every auth event for SOC visibility
- Generic error messages to prevent user-enumeration
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from services.auth_service.app.config import get_settings
from services.auth_service.app.main import async_session
from services.auth_service.app.models import User
from services.auth_service.app.schemas import (
    LoginRequest,
    MessageResponse,
    PasswordChangeRequest,
    RefreshRequest,
    RegisterRequest,
    TokenResponse,
    UserResponse,
)
from services.auth_service.app.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    get_lockout_until,
    hash_password,
    is_account_locked,
    verify_password,
)

settings = get_settings()
logger = logging.getLogger(settings.SERVICE_NAME)
router = APIRouter()


# â”€â”€ Dependency: async DB session â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

async def get_db() -> AsyncSession:
    """Yield an async session; roll back on unhandled errors."""
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# â”€â”€ Dependency: current authenticated user â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

async def get_current_user(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    Extract and validate the Bearer JWT from the Authorization header.

    WHY not use OAuth2PasswordBearer?  Our API Gateway already validates
    tokens; this dependency is a defence-in-depth layer for when the auth
    service is called directly (e.g., /me or password-change).
    """
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid authorization header",
        )
    token = auth_header.removeprefix("Bearer ")
    claims = decode_token(token)
    if claims is None or claims.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )
    user_id = claims.get("sub")
    result = await db.execute(select(User).where(User.id == uuid.UUID(user_id)))
    user = result.scalar_one_or_none()
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or disabled",
        )
    return user


# â”€â”€ POST /register â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
)
async def register(body: RegisterRequest, db: AsyncSession = Depends(get_db)):
    """
    Register a new user.

    New accounts default to the ``viewer`` role; an admin must promote them.
    This prevents privilege escalation via self-registration.
    """
    # Check for existing user (email OR username)
    existing = await db.execute(
        select(User).where(
            or_(User.email == body.email, User.username == body.username)
        )
    )
    if existing.scalar_one_or_none():
        # Generic message â€“ don't reveal which field collided
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with these credentials already exists",
        )

    user = User(
        email=body.email,
        username=body.username,
        hashed_password=hash_password(body.password),
        full_name=body.full_name,
        role="viewer",  # least-privilege default
    )
    db.add(user)
    await db.flush()  # populate id & server defaults before response
    await db.refresh(user)

    logger.info(
        "user_registered",
        extra={"user_id": str(user.id), "username": user.username},
    )
    return UserResponse(
        id=user.id,
        email=user.email,
        username=user.username,
        full_name=user.full_name,
        role=user.role,
        is_active=user.is_active,
        created_at=user.created_at,
    )


# â”€â”€ POST /login â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest, db: AsyncSession = Depends(get_db)):
    """
    Authenticate with email/username + password.

    Security flow:
    1. Look up user by email or username.
    2. If account is locked â†’ 423 Locked.
    3. Verify password â†’ on failure, increment counter; on threshold, lock.
    4. On success â†’ reset counter, issue access + refresh tokens.
    """
    result = await db.execute(
        select(User).where(
            or_(User.email == body.username, User.username == body.username)
        )
    )
    user = result.scalar_one_or_none()

    # Generic failure message to prevent user-enumeration
    credentials_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials",
    )

    if user is None:
        raise credentials_error

    # â”€â”€ Account lockout check â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if is_account_locked(user.locked_until):
        logger.warning(
            "login_attempt_on_locked_account",
            extra={"user_id": str(user.id)},
        )
        raise HTTPException(
            status_code=status.HTTP_423_LOCKED,
            detail="Account temporarily locked due to repeated failed attempts",
        )

    # â”€â”€ Password verification â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if not verify_password(body.password, user.hashed_password):
        user.failed_login_attempts += 1
        if user.failed_login_attempts >= settings.MAX_FAILED_ATTEMPTS:
            user.locked_until = get_lockout_until()
            logger.warning(
                "account_locked",
                extra={
                    "user_id": str(user.id),
                    "locked_until": user.locked_until.isoformat(),
                },
            )
        await db.flush()
        raise credentials_error

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled",
        )

    # â”€â”€ Success: reset lockout & issue tokens â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    user.failed_login_attempts = 0
    user.locked_until = None

    access_token, expires_in = create_access_token(
        user_id=str(user.id),
        role=user.role,
        username=user.username,
    )
    refresh_token, family = create_refresh_token(str(user.id))
    user.refresh_token_family = family
    await db.flush()

    logger.info(
        "user_logged_in",
        extra={"user_id": str(user.id), "username": user.username},
    )
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=expires_in,
    )


# â”€â”€ POST /refresh â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@router.post("/refresh", response_model=TokenResponse)
async def refresh(body: RefreshRequest, db: AsyncSession = Depends(get_db)):
    """
    Rotate a refresh token.

    WHY rotate?  A stolen refresh token can only be used once.  If the
    legitimate user refreshes first, the attacker's token is invalidated.
    If the attacker refreshes first, the legitimate user's next attempt
    will fail â†’ revealing the compromise.
    """
    claims = decode_token(body.refresh_token)
    if claims is None or claims.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )

    user_id = claims["sub"]
    family = claims.get("family")

    result = await db.execute(select(User).where(User.id == uuid.UUID(user_id)))
    user = result.scalar_one_or_none()

    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or disabled",
        )

    # â”€â”€ Family mismatch â†’ token replay detected â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if user.refresh_token_family != family:
        # Revoke the entire family by clearing it â€“ forces re-login
        user.refresh_token_family = None
        await db.flush()
        logger.warning(
            "refresh_token_replay_detected",
            extra={"user_id": user_id, "family": family},
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token reuse detected â€“ please log in again",
        )

    # â”€â”€ Issue new pair in the same family â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    access_token, expires_in = create_access_token(
        user_id=str(user.id),
        role=user.role,
        username=user.username,
    )
    new_refresh, new_family = create_refresh_token(str(user.id), family=family)
    user.refresh_token_family = new_family
    await db.flush()

    return TokenResponse(
        access_token=access_token,
        refresh_token=new_refresh,
        expires_in=expires_in,
    )


# â”€â”€ POST /change-password â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@router.post("/change-password", response_model=MessageResponse)
async def change_password(
    body: PasswordChangeRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Change password for the authenticated user.

    After a password change, all refresh tokens are invalidated by
    clearing the family â€” the user must log in again on other devices.
    """
    if not verify_password(body.current_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect",
        )

    current_user.hashed_password = hash_password(body.new_password)
    # Invalidate all existing refresh tokens
    current_user.refresh_token_family = None
    await db.flush()

    logger.info(
        "password_changed",
        extra={"user_id": str(current_user.id)},
    )
    return MessageResponse(message="Password updated successfully")


# â”€â”€ GET /me â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@router.get("/me", response_model=UserResponse)
async def me(current_user: User = Depends(get_current_user)):
    """Return the profile of the currently authenticated user."""
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        username=current_user.username,
        full_name=current_user.full_name,
        role=current_user.role,
        is_active=current_user.is_active,
        created_at=current_user.created_at,
    )

