"""
scripts/create_first_admin.py — One-off script to bootstrap the first admin
user in Firebase, since there's no Admin Panel UI yet to do this through.

Usage:
    python scripts/create_first_admin.py

Reads FIREBASE_SERVICE_ACCOUNT_JSON from .env (same as the app does).
Delete this script's usefulness once the Admin Panel user management UI
exists — at that point, use the UI instead of a one-off script.
"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.providers.auth.factory import get_auth_provider

def main():
    email = input("Admin email: ").strip()
    password = input("Admin password (min 6 chars): ").strip()
    full_name = input("Full name (optional): ").strip()

    provider = get_auth_provider()
    user = provider.create_user(email=email, password=password, role="admin", full_name=full_name)
    print(f"\n✅ Created admin user: uid={user.uid} email={user.email} role={user.role}")

if __name__ == "__main__":
    main()
