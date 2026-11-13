"""
app/config/secrets.py — Unified secrets loader (Step 1: GCP first).

HOW IT WORKS
============
Production (GCP Cloud Run / GKE):
  - SECRETS_BACKEND=gcp  + GCP_PROJECT_ID are set as non-secret env vars.
  - Every secret value is fetched from GCP Secret Manager at cold-start.
  - Zero plaintext credentials in the image, Docker layers, or env vars.

Local dev (unchanged from today):
  - SECRETS_BACKEND defaults to "local" (or is absent).
  - All values come from .env exactly as before — no code changes needed.

Other clouds (future):
  - SECRETS_BACKEND=aws  → AWS Secrets Manager  (Step 7)
  - SECRETS_BACKEND=azure → Azure Key Vault       (Step 13)
  - The adapter pattern means app/config/settings.py never changes.

ADDING A NEW SECRET (checklist)
================================
1. Add it to the SECRET_NAMES dict below (logical name → GCP secret name).
2. Add os.getenv(LOGICAL_NAME) in app/config/settings.py as today.
3. In GCP: gcloud secrets create <gcp-name> --data-file=- <<< "value"
4. Grant Cloud Run SA roles/secretmanager.secretAccessor on the secret.
That's it — the loader injects it before settings.py reads it.

NEVER import from this module at module level outside of settings.py.
Call load_secrets() exactly once, as early as possible (main.py / api/main.py).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger("financial_rag.secrets")


# ── Secret name registry ──────────────────────────────────────────────────────
# Maps the env var name your app already uses → the GCP Secret Manager name.
# Convention: GCP secret names use hyphens (GCP style), env vars use underscores.
#
# Add every secret here. Non-secret config (ports, flags, model names)
# stays in plain env vars / .env — only credentials belong here.

SECRET_NAMES: dict[str, str] = {
    # Firebase Auth (AUTH_PROVIDER=firebase)
    "FIREBASE_SERVICE_ACCOUNT_JSON": "rag-firebase-service-account",  # Admin SDK
    "FIREBASE_WEB_API_KEY":          "rag-firebase-web-api-key",      # sign_in() REST calls

    # Service-to-service trust (API <-> Chainlit, /login/sso endpoint)
    "INTERNAL_SERVICE_SECRET":       "rag-internal-service-secret",

    # Google OAuth (Chainlit @cl.oauth_callback — note: these env var
    # names are fixed by Chainlit itself, not our choice, do not rename)
    "OAUTH_GOOGLE_CLIENT_ID":        "rag-oauth-google-client-id",
    "OAUTH_GOOGLE_CLIENT_SECRET":    "rag-oauth-google-client-secret",

    # AWS Cognito (AUTH_PROVIDER=cognito) — add here once cognito_provider.py is implemented
    # "COGNITO_APP_CLIENT_SECRET":   "rag-cognito-app-client-secret",

    # Azure AD / Entra ID (AUTH_PROVIDER=azure_ad) — add here once azure_ad_provider.py is implemented
    # "AZURE_AD_CLIENT_SECRET":      "rag-azure-ad-client-secret",

    # LLM providers
    "OPENAI_API_KEY":                "rag-openai-api-key",
    "ANTHROPIC_API_KEY":             "rag-anthropic-api-key",
    "GOOGLE_API_KEY":                "rag-google-api-key",         # AI Studio
    "BEDROCK_API_KEY":               "rag-bedrock-api-key",

    # Azure OpenAI
    "AZURE_OPENAI_API_KEY":          "rag-azure-openai-api-key",
    "AZURE_EMBEDDING_API_KEY":       "rag-azure-embedding-api-key",

    # AWS credentials (used for Bedrock IAM fallback)
    "AWS_ACCESS_KEY_ID":             "rag-aws-access-key-id",
    "AWS_SECRET_ACCESS_KEY":         "rag-aws-secret-access-key",

    # Vector DB (Qdrant Cloud — Step 2)
    "QDRANT_API_KEY":                "rag-qdrant-api-key",

    # Cache (Upstash Redis — Step 3)
    "REDIS_PASSWORD":                "rag-redis-password",
    "REDIS_URL":                     "rag-redis-url",              # full URL with token

    # DB (Supabase PostgreSQL — Step 4)
    "AUDIT_DB_URL":                  "rag-audit-db-url",           # full conn string

    # GCP OCR / Document AI
    "GCP_PROCESSOR_ID":              "rag-gcp-processor-id",
}


# ── Public API ────────────────────────────────────────────────────────────────

def load_secrets() -> None:
    """
    Fetch all secrets for the active backend and inject them into os.environ.

    Call this ONCE at process startup, before any settings.py import.
    Subsequent imports of settings.py will see the injected values via os.getenv().

    Silently skips secrets that already exist in os.environ so that:
      - Unit tests can override individual values without fighting the loader.
      - Docker --env flags take precedence over Secret Manager when needed.
    """
    backend = _backend()
    if backend == "local":
        logger.debug("SECRETS_BACKEND=local — using .env values as-is")
        return

    if backend == "gcp":
        _load_gcp_secrets()
        return

    if backend == "aws":
        # Placeholder — implemented in Step 7
        logger.warning("SECRETS_BACKEND=aws not yet implemented; falling back to env")
        return

    if backend == "azure":
        # Placeholder — implemented in Step 13
        logger.warning("SECRETS_BACKEND=azure not yet implemented; falling back to env")
        return

    logger.warning("Unknown SECRETS_BACKEND=%r — using env vars as-is", backend)


def get_secret(logical_name: str) -> Optional[str]:
    """
    Get a single secret by its env-var logical name (e.g. "OPENAI_API_KEY").

    Checks os.environ first (covers local dev + already-loaded secrets).
    Falls back to GCP Secret Manager if SECRETS_BACKEND=gcp and value missing.

    Useful for lazy-loading a secret that was not in SECRET_NAMES at startup.
    """
    value = os.environ.get(logical_name)
    if value:
        return value

    if _backend() == "gcp":
        gcp_name = SECRET_NAMES.get(logical_name, logical_name.lower().replace("_", "-"))
        return _fetch_gcp_secret(gcp_name)

    return None


# ── Private helpers ───────────────────────────────────────────────────────────

def _backend() -> str:
    return os.getenv("SECRETS_BACKEND", "local").lower().strip()


def _load_gcp_secrets() -> None:
    """
    Fetch every secret in SECRET_NAMES from GCP Secret Manager and
    inject into os.environ (skip if already set).

    Authentication (in priority order — no code change needed):
      1. Workload Identity (Cloud Run, GKE) — automatic, zero config.
      2. GOOGLE_APPLICATION_CREDENTIALS env var → path to service account JSON.
      3. gcloud auth application-default login → local dev with GCP access.
    """
    project_id = os.getenv("GCP_PROJECT_ID")
    if not project_id:
        raise EnvironmentError(
            "SECRETS_BACKEND=gcp requires GCP_PROJECT_ID to be set as a "
            "plain env var (it is not a secret — just the project identifier)."
        )

    try:
        from google.cloud import secretmanager  # type: ignore
    except ImportError:
        raise ImportError(
            "google-cloud-secret-manager is not installed.\n"
            "Add to requirements.txt:  google-cloud-secret-manager>=2.20.0\n"
            "Or install now:           pip install google-cloud-secret-manager"
        )

    client = secretmanager.SecretManagerServiceClient()
    loaded, skipped, failed = [], [], []

    for env_var, secret_name in SECRET_NAMES.items():
        # Skip if already in environment (unit tests, Docker --env flags)
        if os.environ.get(env_var):
            skipped.append(env_var)
            continue

        value = _fetch_gcp_secret_with_client(client, project_id, secret_name)
        if value is not None:
            os.environ[env_var] = value
            loaded.append(env_var)
        else:
            # Missing secret is a warning, not a crash — some secrets are
            # optional (e.g. BEDROCK_API_KEY when using OpenAI only).
            failed.append(f"{env_var} (gcp:{secret_name})")

    logger.info(
        "GCP secrets loaded=%d skipped=%d missing=%d  project=%s",
        len(loaded), len(skipped), len(failed), project_id,
    )
    if failed:
        logger.warning(
            "The following secrets were not found in GCP Secret Manager "
            "(they may be optional): %s",
            ", ".join(failed),
        )


def _fetch_gcp_secret_with_client(client, project_id: str, secret_name: str) -> Optional[str]:
    """Fetch the latest version of a GCP secret. Returns None if not found."""
    resource = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
    try:
        response = client.access_secret_version(request={"name": resource})
        return response.payload.data.decode("utf-8").strip()
    except Exception as exc:
        # google.api_core.exceptions.NotFound → secret doesn't exist
        # google.api_core.exceptions.PermissionDenied → IAM not configured
        exc_type = type(exc).__name__
        if "NotFound" in exc_type:
            logger.debug("GCP secret not found: %s", secret_name)
        elif "PermissionDenied" in exc_type:
            logger.error(
                "Permission denied reading GCP secret '%s'. "
                "Grant roles/secretmanager.secretAccessor to your service account.",
                secret_name,
            )
        else:
            logger.warning("Failed to fetch GCP secret '%s': %s", secret_name, exc)
        return None


def _fetch_gcp_secret(secret_name: str) -> Optional[str]:
    """Standalone fetch for get_secret() lazy calls."""
    project_id = os.getenv("GCP_PROJECT_ID")
    if not project_id:
        return None
    try:
        from google.cloud import secretmanager  # type: ignore
        client = secretmanager.SecretManagerServiceClient()
        return _fetch_gcp_secret_with_client(client, project_id, secret_name)
    except Exception:
        return None