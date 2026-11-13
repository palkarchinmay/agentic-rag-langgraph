"""
app/providers/auth/azure_ad_provider.py — Azure AD / Entra ID External ID
implementation (STUB).

Not yet implemented. Fill in when deploying to Azure. Read this whole
docstring before starting — the sign_in() method is the one place this
provider genuinely cannot match Firebase/Cognito's shape, not just a
"different SDK call" difference.

  verify_token()   — Standard JWT verification against Azure AD's JWKS
                      endpoint (https://login.microsoftonline.com/{tenant}/
                      discovery/v2.0/keys), checking `aud`/`iss`/`exp` via
                      python-jose or PyJWT. Same general shape as Cognito.

  sign_in()        — THIS IS THE PROBLEM METHOD. Azure AD B2C / External ID
                      deliberately does not support a clean server-side
                      "here's an email and password, give me a token" flow
                      the way Firebase's Identity Toolkit REST API or
                      Cognito's USER_PASSWORD_AUTH do. The Resource Owner
                      Password Credentials (ROPC) grant exists in principle
                      via MSAL but Microsoft explicitly recommends against
                      it for B2C and it doesn't support MFA or modern auth
                      policies. The realistic options are:
                        (a) Use ROPC anyway (msal.PublicClientApplication
                            .acquire_token_by_username_password) accepting
                            the limitations, or
                        (b) Redesign the Chainlit login flow for Azure
                            specifically to redirect through a browser-based
                            OAuth2 Authorization Code flow instead of
                            Chainlit's @cl.password_auth_callback, which
                            expects a synchronous username/password check
                            and doesn't naturally fit a redirect flow.
                      Don't assume (a) will "just work" the way the other
                      two providers do — test it early, and raise this
                      tradeoff with the person deploying to Azure before
                      committing to an approach.

  create_user()      — Microsoft Graph API: POST /users. Requires an app
                        registration with User.ReadWrite.All (or narrower
                        Microsoft Graph permissions), consented by an Azure
                        AD admin.
  get_user()          — Graph API: GET /users/{id}.
  list_users()        — Graph API: GET /users, paginated via @odata.nextLink.
  update_user()       — Graph API: PATCH /users/{id}.
  set_user_role()     — No direct Firebase-style custom-claims equivalent.
                         Options: (a) Azure AD App Roles (assigned via
                         Graph API app role assignments, shows up in the
                         token's `roles` claim), or (b) a custom
                         "extension attribute" on the user object. App
                         Roles is the more standard approach and is what
                         verify_token() should read from.
  disable_user()      — Graph API: PATCH /users/{id} with accountEnabled=false.
  enable_user()       — Same, accountEnabled=true.
  delete_user()       — Graph API: DELETE /users/{id}.

Credentials: an Azure AD app registration (client ID + client secret or
certificate) with admin-consented Microsoft Graph application permissions,
authenticated via MSAL's ConfidentialClientApplication to get a Graph API
access token before any of the above calls.
"""

from app.providers.auth.base import AuthProvider


class AzureADAuthProvider(AuthProvider):
    def __init__(self):
        raise NotImplementedError(
            "AzureADAuthProvider is not implemented yet — and sign_in() in "
            "particular needs a design decision, not just an SDK swap. See "
            "this file's module docstring."
        )