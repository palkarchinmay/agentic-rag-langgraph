"""
api/routers/auth.py — POST /api/v1/auth/login, /login/sso, GET /me

Replaces the old custom-JWT implementation. All token issuance/verification
now goes through app.providers.auth.factory.get_auth_provider() — see
app/providers/auth/base.py for the interface every cloud implements.

TWO LOGIN PATHS:
  /login      — email + password. Calls provider.sign_in().
  /login/sso  — federated identity (currently: Google via Chainlit's
                @cl.oauth_callback). Protected by a shared service secret
                (INTERNAL_SERVICE_SECRET) since this endpoint trusts the
                caller's claim that identity was already verified
                elsewhere — see _require_internal_service_secret() below
                before touching this endpoint's auth logic.
"""

import logging
import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Header, status
from pydantic import BaseModel

from api.middleware.auth import require_auth

logger = logging.getLogger("financial_rag.api.auth")
router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


# ── Request / Response models ─────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str  # kept as "username" for backwards-compat with the
                    # Chainlit client — in practice this is the user's email
    password: str

    class Config:
        json_schema_extra = {
            "example": {"username": "admin@example.com", "password": "admin123"}
        }


class SSOLoginRequest(BaseModel):
    email:     str
    full_name: str = ""


class LoginResponse(BaseModel):
    access_token:  str
    refresh_token: Optional[str] = None
    token_type:    str = "bearer"
    expires_in:    int              # seconds
    username:      str
    role:          str
    full_name:     str


class MeResponse(BaseModel):
    username:  str
    role:      str
    full_name: str


# ── Internal helpers ─────────────────────────────────────────────────────────

def _require_internal_service_secret(x_internal_service_key: Optional[str]) -> None:
    """Guards /login/sso — only our own Chainlit service should be able to
    call this, since it bypasses password verification entirely on the
    strength of "trust me, I already checked this email via Google
    OAuth". Without this check, anyone could POST an arbitrary email and
    get a valid session for it."""
    expected = os.getenv("INTERNAL_SERVICE_SECRET", "")
    if not expected:
        # Fail closed, not open — a missing secret in prod should never
        # silently disable this check.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SSO login is not configured (INTERNAL_SERVICE_SECRET unset).",
        )
    if not x_internal_service_key or x_internal_service_key != expected:
        logger.warning("SSO login rejected: missing or invalid service key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid service credentials.",
        )


def _response_from_tokens(tokens, provider) -> LoginResponse:
    """verify_token() the token we just minted so the response carries the
    same normalized claims (role, full_name) that middleware.auth will
    read back out of it later — one source of truth for that mapping."""
    claims = provider.verify_token(tokens.id_token)
    return LoginResponse(
        access_token=  tokens.id_token,
        refresh_token= tokens.refresh_token,
        token_type=    "bearer",
        expires_in=    tokens.expires_in or 3600,
        username=      claims.email,
        role=          claims.role,
        full_name=     claims.full_name,
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post(
    "/login",
    response_model=LoginResponse,
    summary="Login with email + password",
    description="Authenticate via the configured auth provider (Firebase by default). "
                "Pass the returned token as `Authorization: Bearer <token>` on subsequent requests.",
)
async def login(req: LoginRequest) -> LoginResponse:
    from app.providers.auth.factory import get_auth_provider

    provider = get_auth_provider()
    try:
        tokens = provider.sign_in(req.username, req.password)
    except ValueError as e:
        logger.warning("Login failed: email='%s' reason='%s'", req.username, e)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    resp = _response_from_tokens(tokens, provider)
    logger.info("Login success: email='%s' role='%s'", resp.username, resp.role)
    return resp


@router.post(
    "/login/sso",
    response_model=LoginResponse,
    summary="Login via federated identity (internal use only)",
    description="Called by the Chainlit service after it completes a Google OAuth "
                "handshake via @cl.oauth_callback. Requires the X-Internal-Service-Key "
                "header — not intended to be called directly by end users.",
)
async def login_sso(
    req: SSOLoginRequest,
    x_internal_service_key: Optional[str] = Header(None),
) -> LoginResponse:
    _require_internal_service_secret(x_internal_service_key)

    from app.providers.auth.factory import get_auth_provider

    provider = get_auth_provider()
    user = provider.get_or_create_user_by_email(
        email=req.email, full_name=req.full_name, default_role="user",
    )
    tokens = provider.mint_session_tokens(user.uid)

    resp = _response_from_tokens(tokens, provider)
    logger.info("SSO login success: email='%s' role='%s'", resp.username, resp.role)
    return resp


@router.get(
    "/me",
    response_model=MeResponse,
    summary="Get current user",
    description="Returns info about the currently authenticated user. "
                "Pass `Authorization: Bearer <token>`.",
)
async def me(user: dict = Depends(require_auth)) -> MeResponse:
    return MeResponse(
        username=  user.get("email", user.get("sub", "")),
        role=      user.get("role", "user"),
        full_name= user.get("full_name", ""),
    )