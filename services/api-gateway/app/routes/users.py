"""
User Management Routes
========================
CRUD for platform users (analysts, managers, admins).

RBAC:
  • ADMIN — full user management (list, promote, deactivate)
  • Others — can only read their own profile via ``/auth/me``
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.jwt_handler import Role
from app.auth.rbac import TokenPayload, require_role
from app.models.database import User, get_db
from app.models.schemas import UserResponse

router = APIRouter(prefix="/users", tags=["users"])


@router.get(
    "",
    response_model=list[UserResponse],
    summary="List all platform users (ADMIN only)",
)
async def list_users(
    _user: TokenPayload = Depends(require_role(Role.ADMIN)),
    db: AsyncSession = Depends(get_db),
) -> list[UserResponse]:
    result = await db.execute(
        select(User).where(User.is_active == True).order_by(User.created_at.desc())  # noqa: E712
    )
    return [UserResponse.model_validate(u) for u in result.scalars().all()]


@router.patch(
    "/{user_id}/role",
    response_model=UserResponse,
    summary="Change a user's RBAC role (ADMIN only)",
)
async def update_user_role(
    user_id: str,
    body: dict,
    _user: TokenPayload = Depends(require_role(Role.ADMIN)),
    db: AsyncSession = Depends(get_db),
) -> UserResponse:
    new_role = body.get("role", "").upper()
    valid_roles = {r.value for r in Role}
    if new_role not in valid_roles:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid role. Must be one of: {', '.join(valid_roles)}",
        )

    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    user.role = new_role
    await db.commit()
    await db.refresh(user)
    return UserResponse.model_validate(user)


@router.patch(
    "/{user_id}/deactivate",
    response_model=UserResponse,
    summary="Deactivate a user account (ADMIN only)",
)
async def deactivate_user(
    user_id: str,
    _user: TokenPayload = Depends(require_role(Role.ADMIN)),
    db: AsyncSession = Depends(get_db),
) -> UserResponse:
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    user.is_active = False
    await db.commit()
    await db.refresh(user)
    return UserResponse.model_validate(user)
