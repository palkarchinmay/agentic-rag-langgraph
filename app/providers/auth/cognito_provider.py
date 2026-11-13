"""
app/providers/auth/cognito_provider.py — AWS Cognito implementation (STUB).

Not yet implemented. Fill in when deploying to AWS. Notes for whoever
does this:

  verify_token()   — Cognito ID tokens are JWTs signed by Cognito's own
                      JWKS endpoint (https://cognito-idp.{region}.amazonaws.com/
                      {user_pool_id}/.well-known/jwks.json). Verify with
                      python-jose or PyJWT + that JWKS, checking `aud` /
                      `token_use=id` / `exp`. boto3 has no direct "verify
                      this token" call — you do the JWT verification
                      yourself, similar to what the OLD custom-JWT
                      implementation in this repo already did (see git
                      history of api/routers/auth.py pre-Firebase).

  sign_in()        — boto3 cognito-idp client, InitiateAuth with
                      AuthFlow="USER_PASSWORD_AUTH". Needs the User Pool's
                      App Client ID (and secret hash if the app client has
                      a secret configured — SECRET_HASH computation is
                      easy to get wrong, budget time for it).

  get_or_create_user_by_email() / mint_session_tokens() — Cognito has
                      native Google federation via a Hosted UI + Identity
                      Provider configuration (User Pool > Sign-in
                      experience > Federated identity providers), which
                      handles the whole SSO handshake without needing
                      Firebase's custom-token round-trip at all. If you
                      wire that up, these two methods may not even be
                      needed the same way — AdminInitiateAuth with
                      AuthFlow="ADMIN_USER_PASSWORD_AUTH" after a
                      federated sign-in can often issue tokens directly.
                      Worth re-evaluating this whole SSO design for
                      Cognito rather than assuming it maps 1:1.

  create_user()     — AdminCreateUser + AdminSetUserPassword (permanent).
  get_user()        — AdminGetUser.
  list_users()      — ListUsers, paginated via PaginationToken.
  update_user()     — AdminUpdateUserAttributes.
  set_user_role()   — Cognito doesn't have "custom claims" like Firebase.
                       Use a custom attribute (e.g. "custom:role"), which
                       must be defined on the User Pool schema BEFORE any
                       user is created — it cannot be added retroactively
                       to existing users' schema after pool creation.
  disable_user()    — AdminDisableUser.
  enable_user()      — AdminEnableUser.
  delete_user()      — AdminDeleteUser.

Credentials: standard AWS SDK credential chain (IAM role on the compute,
or AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY) — no separate "web API key"
concept like Firebase; boto3 handles auth to AWS itself, separately from
end-user authentication via Cognito.
"""

from app.providers.auth.base import AuthProvider


class CognitoAuthProvider(AuthProvider):
    def __init__(self):
        raise NotImplementedError(
            "CognitoAuthProvider is not implemented yet. See this file's "
            "module docstring for what each method needs to do."
        )