"""
scripts/promote_to_admin.py — Promote an EXISTING Firebase user to admin.

Use this when you already have a user account but no admin exists to
promote them through the normal Admin Panel (chicken-and-egg problem --
every admin-only endpoint needs an existing admin to call it).

Usage:
    python scripts/promote_to_admin.py

Reads FIREBASE_SERVICE_ACCOUNT_JSON from .env, same as the app does.
Self-contained: initializes the Firebase Admin SDK directly rather than
relying on the app's own lazy-init (which only happens as a side effect
inside FirebaseAuthProvider's methods, not on raw firebase_admin.auth
calls like the one this script needs for the email->uid lookup).
"""
import json
import os
import sys
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import firebase_admin
from firebase_admin import credentials, auth as fb_auth


def _ensure_firebase_initialized():
    if firebase_admin._apps:
        return  # already initialized
    raw = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
    if not raw:
        print("\n❌ FIREBASE_SERVICE_ACCOUNT_JSON not found in .env — "
              "can't initialize the Firebase Admin SDK.")
        sys.exit(1)
    cred = credentials.Certificate(json.loads(raw))
    firebase_admin.initialize_app(cred)


def main():
    _ensure_firebase_initialized()

    email = input("Email of the user to promote to admin: ").strip()

    try:
        fb_user = fb_auth.get_user_by_email(email)
    except fb_auth.UserNotFoundError:
        print(f"\n❌ No user found with email={email}. "
              f"Create one first with scripts/create_first_admin.py.")
        return

    claims = dict(fb_user.custom_claims or {})
    claims["role"] = "admin"
    fb_auth.set_custom_user_claims(fb_user.uid, claims)

    print(f"\n✅ {email} (uid={fb_user.uid}) is now an admin.")
    print("They'll need to log out and back in for the new role to take "
          "effect in their session (Firebase caches roles in the token).")


if __name__ == "__main__":
    main()