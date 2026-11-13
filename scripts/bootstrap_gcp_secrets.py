#!/usr/bin/env python3
"""
scripts/bootstrap_gcp_secrets.py — One-time secret bootstrap for GCP.

PURPOSE
=======
Reads your existing .env file and creates (or updates) every secret in
GCP Secret Manager.  Run this once when standing up a new GCP project.
After that, rotate secrets via the GCP console or gcloud — not via this script.

USAGE
=====
# Dry run — see what would be created/updated without touching GCP:
python scripts/bootstrap_gcp_secrets.py --dry-run

# Live run — writes to GCP Secret Manager:
python scripts/bootstrap_gcp_secrets.py

# Specific project (overrides GCP_PROJECT_ID in .env):
python scripts/bootstrap_gcp_secrets.py --project my-gcp-project-id

# Only create missing secrets (never overwrite existing):
python scripts/bootstrap_gcp_secrets.py --no-overwrite

PREREQUISITES
=============
1. gcloud auth login  (or GOOGLE_APPLICATION_CREDENTIALS set)
2. pip install google-cloud-secret-manager python-dotenv
3. GCP_PROJECT_ID in your .env or passed via --project
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _read_dotenv(env_path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not env_path.exists():
        print(f"Warning: .env not found at {env_path}", file=sys.stderr)
        return values
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        val = val.strip().strip('"').strip("'")
        if val:
            values[key.strip()] = val
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap GCP Secret Manager from .env")
    parser.add_argument("--project", help="GCP project ID")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-overwrite", action="store_true")
    args = parser.parse_args()

    env_values = _read_dotenv(Path(args.env_file))
    project_id = args.project or env_values.get("GCP_PROJECT_ID") or os.getenv("GCP_PROJECT_ID")
    if not project_id:
        print("ERROR: GCP_PROJECT_ID not found. Set in .env or pass --project.", file=sys.stderr)
        sys.exit(1)

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from app.config.secrets import SECRET_NAMES

    if not args.dry_run:
        try:
            from google.cloud import secretmanager
        except ImportError:
            print("ERROR: pip install google-cloud-secret-manager", file=sys.stderr)
            sys.exit(1)
        client = secretmanager.SecretManagerServiceClient()
        parent = f"projects/{project_id}"

    created = updated = skipped = missing = 0

    for env_var, gcp_name in SECRET_NAMES.items():
        value = env_values.get(env_var)
        if not value:
            print(f"  SKIP  {env_var:40s}  (not in .env)")
            missing += 1
            continue

        display = value[:4] + "***"
        if args.dry_run:
            print(f"  DRY   {env_var:40s}  -> {gcp_name}  ({display})")
            continue

        secret_path = f"{parent}/secrets/{gcp_name}"
        secret_exists = False
        try:
            client.get_secret(request={"name": secret_path})
            secret_exists = True
        except Exception as e:
            if "NotFound" not in type(e).__name__:
                print(f"  ERR   {env_var}: {e}")
                continue

        if secret_exists and args.no_overwrite:
            print(f"  SKIP  {env_var:40s}  (exists, --no-overwrite)")
            skipped += 1
            continue

        if not secret_exists:
            try:
                client.create_secret(request={
                    "parent": parent,
                    "secret_id": gcp_name,
                    "secret": {
                        "replication": {"automatic": {}},
                        "labels": {"app": "rag-platform"},
                    },
                })
            except Exception as e:
                print(f"  ERR   {env_var}: create failed: {e}")
                continue

        try:
            client.add_secret_version(request={
                "parent": secret_path,
                "payload": {"data": value.encode("utf-8")},
            })
            verb = "UPDATE" if secret_exists else "CREATE"
            print(f"  {verb:6s}  {env_var:40s}  -> {gcp_name}  ({display})")
            if secret_exists:
                updated += 1
            else:
                created += 1
        except Exception as e:
            print(f"  ERR   {env_var}: add version failed: {e}")

    print(f"\nDone. created={created} updated={updated} skipped={skipped} missing_in_env={missing}")
    if not args.dry_run:
        print(f"\nNext: set these NON-SECRET env vars on Cloud Run:")
        print(f"  SECRETS_BACKEND=gcp")
        print(f"  GCP_PROJECT_ID={project_id}")


if __name__ == "__main__":
    main()