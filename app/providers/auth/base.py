"""
app/providers/auth/base.py — Cloud-agnostic auth provider interface.

Every concrete provider (Firebase, Cognito, Azure AD) implements this same
contract. Nothing outside app/providers/auth/ should ever import a
cloud-specific SDK (firebase_admin, boto3, msal, ...) directly — always go
through get_auth_provider() from factory.py.

DESIGN NOTE — these providers are NOT perfectly symmetric under the hood:
  - Firebase Admin SDK has no server-side password verification; sign_in()
    calls the Identity Toolkit REST API instead (needs a Web API key, a
    different credential than the Admin SDK service account).
  - Cognito's equivalent is InitiateAuth(AuthFlow=USER_PASSWORD_AUTH).
  - Azure AD B2C discourages server-side password grants (ROPC) entirely;
    a real deployment there likely wants a browser redirect flow instead,
    which doesn't fit this interface's sign_in() shape at all. Flag this
    to whoever implements azure_ad_provider.py before assuming it "just
    works" the same way Firebase/Cognito do.

The role model assumed everywhere in this app is a single string:
"admin" | "engineer" | "user". Each provider is responsible for mapping
its own claims mechanism (Firebase custom claims, Cognito custom
attributes, Azure AD app roles / extension attributes) onto that one
field in UserClaims.role / UserRecord.role.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


VALID_ROLES = ("admin", "engineer", "user")


@dataclass
class UserClaims:
    """Normalized result of verifying a caller's token. This is the shape
    api/middleware/auth.py hands to every @router.get(..., Depends(require_auth))
    route as `user["sub"]`, `user["role"]`, etc. — kept dict-compatible via
    as_dict() since the rest of the API already expects payload.get(...)."""
    uid:            str
    email:          str
    role:           str = "user"
    email_verified: bool = False
    full_name:      str = ""
    raw_claims:     dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "sub":            self.uid,
            "email":          self.email,
            "role":           self.role,
            "email_verified": self.email_verified,
            "full_name":      self.full_name,
        }


@dataclass
class AuthTokens:
    """Result of a successful sign_in() call."""
    id_token:      str
    refresh_token: Optional[str] = None
    expires_in:    Optional[int] = None   # seconds


@dataclass
class UserRecord:
    """Result of get_user() / list_users() / create_user() — admin-facing
    user data, distinct from UserClaims (which comes from a verified token)."""
    uid:       str
    email:     str
    role:      str = "user"
    full_name: str = ""
    disabled:  bool = False


class AuthProvider(ABC):
    """Cloud-agnostic auth interface. See module docstring for the
    asymmetry caveats before assuming every method behaves identically
    across providers."""

    # ── Token verification (used on every protected API request) ──────────

    @abstractmethod
    def verify_token(self, id_token: str) -> UserClaims:
        """Verify a bearer token and return normalized claims.
        Raise ValueError (caught by callers as 401) on invalid/expired token."""
        raise NotImplementedError

    # ── Sign-in (email/password login exchange) ────────────────────────────

    @abstractmethod
    def sign_in(self, email: str, password: str) -> AuthTokens:
        """Verify email/password and return fresh tokens.
        Raise ValueError (caught by callers as 401) on bad credentials."""
        raise NotImplementedError

    @abstractmethod
    def get_or_create_user_by_email(self, email: str, full_name: str = "",
                                     default_role: str = "user") -> UserRecord:
        """SSO login path (Google, or any other federated IdP): the caller
        (e.g. api/routers/auth.py's /login/sso endpoint) has already
        verified the caller's identity via a service-to-service trust
        boundary — this method does NOT re-verify anything, it just
        looks up or creates the corresponding user record. New users get
        default_role and no password set (provider-only account)."""
        raise NotImplementedError

    @abstractmethod
    def mint_session_tokens(self, uid: str) -> AuthTokens:
        """Issue fresh tokens for a uid we already trust (used right after
        get_or_create_user_by_email, since SSO has no password to run
        through sign_in()). Firebase: mint a custom token, then exchange
        it for a real ID token via the Identity Toolkit REST API, so the
        result is verify_token()-compatible just like the password path."""
        raise NotImplementedError

    # ── Admin user management (used by /api/v1/admin/users/*) ──────────────

    @abstractmethod
    def create_user(self, email: str, password: str, role: str,
                     full_name: str = "") -> UserRecord:
        raise NotImplementedError

    @abstractmethod
    def get_user(self, uid: str) -> UserRecord:
        raise NotImplementedError

    @abstractmethod
    def list_users(self, page_token: Optional[str] = None,
                    limit: int = 100) -> tuple[list[UserRecord], Optional[str]]:
        """Returns (users, next_page_token). next_page_token is None on the
        last page."""
        raise NotImplementedError

    @abstractmethod
    def update_user(self, uid: str, email: Optional[str] = None,
                     full_name: Optional[str] = None,
                     password: Optional[str] = None) -> UserRecord:
        raise NotImplementedError

    @abstractmethod
    def set_user_role(self, uid: str, role: str) -> None:
        """role must be one of VALID_ROLES."""
        raise NotImplementedError

    @abstractmethod
    def disable_user(self, uid: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def enable_user(self, uid: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def delete_user(self, uid: str) -> None:
        raise NotImplementedError