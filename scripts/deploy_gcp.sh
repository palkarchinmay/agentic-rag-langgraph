#!/usr/bin/env bash
# scripts/deploy_gcp.sh — Deploy RAG platform to Cloud Run (asia-south1).
#
# USAGE
# =====
#   chmod +x scripts/deploy_gcp.sh
#   ./scripts/deploy_gcp.sh
#
# PREREQUISITES
# =============
#   gcloud auth login
#   gcloud config set project YOUR_PROJECT_ID
#   Docker installed and running
#   All secrets already in GCP Secret Manager (run bootstrap_gcp_secrets.py first)
#
# WHAT THIS DOES
# ==============
#   1. Builds the Docker image
#   2. Pushes to Artifact Registry
#   3. Deploys to Cloud Run with:
#      - SECRETS_BACKEND=gcp and GCP_PROJECT_ID as plain env vars (non-secret)
#      - NO secret values in env vars, Docker layers, or this script
#      - Workload Identity → service account can read Secret Manager
#
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Config — edit these ───────────────────────────────────────────────────────
PROJECT_ID="${GCP_PROJECT_ID:-$(gcloud config get-value project)}"
REGION="asia-south1"
SERVICE_NAME="rag-api"
IMAGE_NAME="rag-platform"
REGISTRY="${REGION}-docker.pkg.dev/${PROJECT_ID}/rag-platform/${IMAGE_NAME}"
SA_EMAIL="rag-platform-sa@${PROJECT_ID}.iam.gserviceaccount.com"
# ─────────────────────────────────────────────────────────────────────────────

echo "═══════════════════════════════════════════════════"
echo "  Deploying RAG Platform to Cloud Run"
echo "  Project : ${PROJECT_ID}"
echo "  Region  : ${REGION}"
echo "  Service : ${SERVICE_NAME}"
echo "═══════════════════════════════════════════════════"

# ── Step 1: Ensure Artifact Registry repo exists ──────────────────────────────
echo ""
echo "▶ Ensuring Artifact Registry repository..."
gcloud artifacts repositories describe rag-platform \
    --location="${REGION}" --project="${PROJECT_ID}" &>/dev/null \
  || gcloud artifacts repositories create rag-platform \
       --repository-format=docker \
       --location="${REGION}" \
       --description="RAG Platform images" \
       --project="${PROJECT_ID}"

# ── Step 2: Build & push image ────────────────────────────────────────────────
echo ""
echo "▶ Building Docker image..."
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

docker build \
  --platform linux/amd64 \
  --file Dockerfile.api \
  --tag "${REGISTRY}:latest" \
  --tag "${REGISTRY}:$(git rev-parse --short HEAD 2>/dev/null || echo 'local')" \
  .

echo ""
echo "▶ Pushing image..."
docker push "${REGISTRY}:latest"

# ── Step 3: Ensure service account exists ─────────────────────────────────────
echo ""
echo "▶ Ensuring service account..."
gcloud iam service-accounts describe "${SA_EMAIL}" \
    --project="${PROJECT_ID}" &>/dev/null \
  || gcloud iam service-accounts create rag-platform-sa \
       --display-name="RAG Platform Service Account" \
       --project="${PROJECT_ID}"

# Grant Secret Manager access (idempotent)
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/secretmanager.secretAccessor" \
  --condition=None \
  --quiet

# ── Step 4: Deploy to Cloud Run ───────────────────────────────────────────────
echo ""
echo "▶ Deploying to Cloud Run..."
gcloud run deploy "${SERVICE_NAME}" \
  --image="${REGISTRY}:latest" \
  --region="${REGION}" \
  --platform=managed \
  --service-account="${SA_EMAIL}" \
  --allow-unauthenticated \
  --min-instances=0 \
  --max-instances=10 \
  --memory=2Gi \
  --cpu=2 \
  --timeout=300 \
  --concurrency=80 \
  --set-env-vars="SECRETS_BACKEND=gcp,GCP_PROJECT_ID=${PROJECT_ID},GCP_LOCATION=${REGION},LOG_FORMAT=json,LOG_BACKEND=cloud" \
  --project="${PROJECT_ID}"

# ── Summary ───────────────────────────────────────────────────────────────────
SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
  --region="${REGION}" --project="${PROJECT_ID}" \
  --format="value(status.url)")

echo ""
echo "═══════════════════════════════════════════════════"
echo "  Deploy complete!"
echo "  URL    : ${SERVICE_URL}"
echo "  Health : ${SERVICE_URL}/health"
echo "  Docs   : ${SERVICE_URL}/docs"
echo ""
echo "  Secrets: fetched from GCP Secret Manager at cold-start"
echo "  No credentials in image or env vars ✓"
echo "═══════════════════════════════════════════════════"