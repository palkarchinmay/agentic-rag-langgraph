"""
api/middleware/auth.py — JWT authentication dependency.

Use as FastAPI dependency on protected routes:
    from api.middleware.auth import require_auth, require_role
    
    @router.get("/protected")
    async def protected(user = Depends(require_auth)):
        return {"user": user["sub"]}
        
    @router.get("/admin-only")  
    async def admin(user = Depends(require_role("admin"))):
        return {"user": user["sub"]}
        
Public routes (health, login) skip this dependency entirely.
"""

import logging
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger  = logging.getLogger("financial_rag.api.auth_middleware")
_bearer = HTTPBearer(auto_error=False)


def _get_payload(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> dict:
    """Extract and validate the bearer token via the configured AuthProvider
    (Firebase / Cognito / Azure AD — see app/providers/auth/factory.py).
    Returns the same dict shape regardless of provider, so callers below
    (require_auth, require_role, and every route depending on them) never
    need to know which cloud issued the token."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Pass 'Authorization: Bearer <token>'",
            headers={"WWW-Authenticate": "Bearer"},
        )

    from app.providers.auth.factory import get_auth_provider

    try:
        claims = get_auth_provider().verify_token(credentials.credentials)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token is invalid or expired. Please login again. ({e})",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return claims.as_dict()


# ── Public dependencies ───────────────────────────────────────────────────────

def require_auth(payload: dict = Depends(_get_payload)) -> dict:
    """Require any valid JWT. Returns the decoded payload."""
    return payload


def require_role(*roles: str):
    """
    Require a specific role. Usage:
        Depends(require_role("admin"))
        Depends(require_role("admin", "engineer"))
    """
    def _check(payload: dict = Depends(_get_payload)) -> dict:
        if payload.get("role") not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required role: {list(roles)}. "
                       f"Your role: {payload.get('role')}",
            )
        return payload
    return _check