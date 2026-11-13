"""
app/providers/auth/firebase_provider.py — Firebase Auth implementation.

CREDENTIALS — two separate secrets, not one:
  1. FIREBASE_SERVICE_ACCOUNT_JSON — the full service account JSON key
     (Admin SDK: verify tokens, create/update/delete/list users, set
     custom claims). Register this in app/config/secrets.py's
     SECRET_NAMES so it's pulled from Secret Manager in production the
     same way OPENAI_API_KEY etc. already are.
  2. FIREBASE_WEB_API_KEY — the project's public Web API key (found in
     Firebase Console > Project Settings > General). Used ONLY for the
     sign_in() REST call — the Admin SDK cannot verify a password itself,
     that's a deliberate Firebase design choice (password verification is
     a client-SDK / REST concern, not an Admin-SDK one).

Both must be set for this provider to be usable. Neither is optional.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

import httpx

from app.providers.auth.base import (
    AuthProvider, UserClaims, AuthTokens, UserRecord, VALID_ROLES,
)

logger = logging.getLogger("financial_rag.auth.firebase")

_IDENTITY_TOOLKIT_URL = (
    "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword"
)

_firebase_app = None  # module-level singleton — firebase_admin errors if
                       # initialize_app() is called more than once


def _get_app():
    """Lazily initialize the Firebase Admin SDK app exactly once per process."""
    global _firebase_app
    if _firebase_app is not None:
        return _firebase_app

    import firebase_admin
    from firebase_admin import credentials

    raw = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON", "")
    if not raw:
        raise RuntimeError(
            "FIREBASE_SERVICE_ACCOUNT_JSON is not set. Register it in "
            "app/config/secrets.py SECRET_NAMES (production) or set it "
            "directly in .env (local dev) — full service account JSON as "
            "a single-line string."
        )

    try:
        service_account_info = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"FIREBASE_SERVICE_ACCOUNT_JSON is not valid JSON: {e}"
        ) from e

    cred = credentials.Certificate(service_account_info)
    _firebase_app = firebase_admin.initialize_app(cred)
    logger.info("Firebase Admin SDK initialized (project=%s)",
                service_account_info.get("project_id", "?"))
    return _firebase_app


def _web_api_key() -> str:
    key = os.getenv("FIREBASE_WEB_API_KEY", "")
    if not key:
        raise RuntimeError(
            "FIREBASE_WEB_API_KEY is not set. This is the project's public "
            "Web API key (Firebase Console > Project Settings > General) — "
            "a different credential from the Admin SDK service account, "
            "required for password sign-in."
        )
    return key


def _record_from_firebase_user(user) -> UserRecord:
    """Map a firebase_admin.auth.UserRecord to our normalized UserRecord."""
    claims = user.custom_claims or {}
    return UserRecord(
        uid=user.uid,
        email=user.email or "",
        role=claims.get("role", "user"),
        full_name=user.display_name or "",
        disabled=user.disabled,
    )


class FirebaseAuthProvider(AuthProvider):

    def verify_token(self, id_token: str) -> UserClaims:
        from firebase_admin import auth as fb_auth

        _get_app()
        try:
            decoded = fb_auth.verify_id_token(id_token)
        except Exception as e:
            raise ValueError(f"Invalid or expired token: {e}") from e

        return UserClaims(
            uid=decoded.get("uid", ""),
            email=decoded.get("email", ""),
            role=decoded.get("role", "user"),  # custom claim, set via set_user_role()
            email_verified=decoded.get("email_verified", False),
            full_name=decoded.get("name", ""),
            raw_claims=decoded,
        )

    def sign_in(self, email: str, password: str) -> AuthTokens:
        _get_app()  # not strictly needed for the REST call, but fail fast
                    # here if service-account creds are missing too, since
                    # a broken deployment should fail loudly on either half
        try:
            r = httpx.post(
                _IDENTITY_TOOLKIT_URL,
                params={"key": _web_api_key()},
                json={"email": email, "password": password, "returnSecureToken": True},
                timeout=15,
            )
        except httpx.RequestError as e:
            raise ValueError(f"Could not reach Firebase Identity Toolkit: {e}") from e

        if r.status_code != 200:
            # Firebase returns {"error": {"message": "INVALID_PASSWORD", ...}}
            try:
                msg = r.json().get("error", {}).get("message", r.text)
            except Exception:
                msg = r.text
            raise ValueError(f"Sign-in failed: {msg}")

        data = r.json()
        return AuthTokens(
            id_token=data["idToken"],
            refresh_token=data.get("refreshToken"),
            expires_in=int(data["expiresIn"]) if data.get("expiresIn") else None,
        )

    def get_or_create_user_by_email(self, email: str, full_name: str = "",
                                     default_role: str = "user") -> UserRecord:
        from firebase_admin import auth as fb_auth

        if default_role not in VALID_ROLES:
            raise ValueError(f"default_role must be one of {VALID_ROLES}, got {default_role!r}")

        _get_app()
        try:
            user = fb_auth.get_user_by_email(email)
            logger.info("SSO login: existing Firebase user email=%s uid=%s", email, user.uid)
            return _record_from_firebase_user(user)
        except fb_auth.UserNotFoundError:
            pass  # fall through to create

        # New user — no password (provider-only account, sign_in() will
        # never work for this uid, only the SSO path / mint_session_tokens)
        user = fb_auth.create_user(email=email, display_name=full_name or None,
                                    email_verified=True)  # trusted via Google
        fb_auth.set_custom_user_claims(user.uid, {"role": default_role})
        logger.info("SSO login: created Firebase user email=%s uid=%s role=%s",
                    email, user.uid, default_role)
        return UserRecord(uid=user.uid, email=email, role=default_role,
                           full_name=full_name, disabled=False)

    def mint_session_tokens(self, uid: str) -> AuthTokens:
        from firebase_admin import auth as fb_auth

        _get_app()
        custom_token = fb_auth.create_custom_token(uid)
        # create_custom_token returns bytes
        if isinstance(custom_token, bytes):
            custom_token = custom_token.decode("utf-8")

        # Exchange the custom token for a real ID token — same Identity
        # Toolkit REST API as sign_in(), different verb. This is Firebase's
        # own documented pattern for backend-trusted-identity sign-in.
        exchange_url = (
            "https://identitytoolkit.googleapis.com/v1/accounts:signInWithCustomToken"
        )
        try:
            r = httpx.post(
                exchange_url,
                params={"key": _web_api_key()},
                json={"token": custom_token, "returnSecureToken": True},
                timeout=15,
            )
        except httpx.RequestError as e:
            raise ValueError(f"Could not reach Firebase Identity Toolkit: {e}") from e

        if r.status_code != 200:
            try:
                msg = r.json().get("error", {}).get("message", r.text)
            except Exception:
                msg = r.text
            raise ValueError(f"Custom token exchange failed: {msg}")

        data = r.json()
        return AuthTokens(
            id_token=data["idToken"],
            refresh_token=data.get("refreshToken"),
            expires_in=int(data["expiresIn"]) if data.get("expiresIn") else None,
        )

    def create_user(self, email: str, password: str, role: str,
                     full_name: str = "") -> UserRecord:
        from firebase_admin import auth as fb_auth

        if role not in VALID_ROLES:
            raise ValueError(f"role must be one of {VALID_ROLES}, got {role!r}")

        _get_app()
        user = fb_auth.create_user(
            email=email, password=password,
            display_name=full_name or None,
        )
        fb_auth.set_custom_user_claims(user.uid, {"role": role})
        logger.info("Created Firebase user uid=%s email=%s role=%s",
                    user.uid, email, role)
        return UserRecord(uid=user.uid, email=email, role=role,
                           full_name=full_name, disabled=False)

    def get_user(self, uid: str) -> UserRecord:
        from firebase_admin import auth as fb_auth

        _get_app()
        try:
            user = fb_auth.get_user(uid)
        except fb_auth.UserNotFoundError as e:
            raise ValueError(f"No user with uid={uid}") from e
        return _record_from_firebase_user(user)

    def list_users(self, page_token: Optional[str] = None,
                    limit: int = 100) -> tuple[list[UserRecord], Optional[str]]:
        from firebase_admin import auth as fb_auth

        _get_app()
        page = fb_auth.list_users(page_token=page_token, max_results=limit)
        users = [_record_from_firebase_user(u) for u in page.users]
        return users, page.next_page_token or None

    def update_user(self, uid: str, email: Optional[str] = None,
                     full_name: Optional[str] = None,
                     password: Optional[str] = None) -> UserRecord:
        from firebase_admin import auth as fb_auth

        _get_app()
        kwargs = {}
        if email is not None:
            kwargs["email"] = email
        if full_name is not None:
            kwargs["display_name"] = full_name
        if password is not None:
            kwargs["password"] = password

        if not kwargs:
            return self.get_user(uid)

        user = fb_auth.update_user(uid, **kwargs)
        logger.info("Updated Firebase user uid=%s fields=%s", uid, list(kwargs))
        return _record_from_firebase_user(user)

    def set_user_role(self, uid: str, role: str) -> None:
        from firebase_admin import auth as fb_auth

        if role not in VALID_ROLES:
            raise ValueError(f"role must be one of {VALID_ROLES}, got {role!r}")

        _get_app()
        # Merge with existing custom claims rather than overwrite — a user
        # could have other custom claims set outside this app.
        try:
            user = fb_auth.get_user(uid)
        except fb_auth.UserNotFoundError as e:
            raise ValueError(f"No user with uid={uid}") from e
        claims = dict(user.custom_claims or {})
        claims["role"] = role
        fb_auth.set_custom_user_claims(uid, claims)
        logger.info("Set role=%s for Firebase user uid=%s", role, uid)

    def disable_user(self, uid: str) -> None:
        from firebase_admin import auth as fb_auth
        _get_app()
        try:
            fb_auth.update_user(uid, disabled=True)
        except fb_auth.UserNotFoundError as e:
            raise ValueError(f"No user with uid={uid}") from e
        logger.info("Disabled Firebase user uid=%s", uid)

    def enable_user(self, uid: str) -> None:
        from firebase_admin import auth as fb_auth
        _get_app()
        try:
            fb_auth.update_user(uid, disabled=False)
        except fb_auth.UserNotFoundError as e:
            raise ValueError(f"No user with uid={uid}") from e
        logger.info("Enabled Firebase user uid=%s", uid)

    def delete_user(self, uid: str) -> None:
        from firebase_admin import auth as fb_auth
        _get_app()
        try:
            fb_auth.delete_user(uid)
        except fb_auth.UserNotFoundError as e:
            raise ValueError(f"No user with uid={uid}") from e
        logger.info("Deleted Firebase user uid=%s", uid)