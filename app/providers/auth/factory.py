"""
app/providers/auth/factory.py — Auth provider factory.

Mirrors app/providers/llm/factory.py and app/providers/embeddings/factory.py:
one function, driven by an env var, returns the concrete implementation.
This is the ONLY place that should import a concrete provider class —
everywhere else in the app (api/middleware/auth.py, api/routers/auth.py,
future admin user-management routes) calls get_auth_provider() and only
ever touches the AuthProvider interface.

Switching clouds:
    AUTH_PROVIDER=firebase   (default)
    AUTH_PROVIDER=cognito    # requires cognito_provider.py to be implemented
    AUTH_PROVIDER=azure_ad   # requires azure_ad_provider.py to be implemented
"""

import os
import logging
from functools import lru_cache

from app.providers.auth.base import AuthProvider

logger = logging.getLogger("financial_rag.auth_factory")


@lru_cache(maxsize=1)
def get_auth_provider() -> AuthProvider:
    """Returns a process-wide singleton — provider init (e.g. Firebase
    Admin SDK app creation) should happen once, not per-request."""
    provider = os.getenv("AUTH_PROVIDER", "firebase").lower().strip()
    logger.info("Auth provider: %s", provider)

    if provider == "firebase":
        from app.providers.auth.firebase_provider import FirebaseAuthProvider
        return FirebaseAuthProvider()

    elif provider == "cognito":
        from app.providers.auth.cognito_provider import CognitoAuthProvider
        return CognitoAuthProvider()

    elif provider == "azure_ad":
        from app.providers.auth.azure_ad_provider import AzureADAuthProvider
        return AzureADAuthProvider()

    else:
        raise ValueError(
            f"Unknown AUTH_PROVIDER={provider!r}. "
            f"Expected one of: firebase, cognito, azure_ad"
        )