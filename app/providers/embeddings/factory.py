"""
embeddings/factory.py — Multi-cloud embedding provider factory.

Supported providers:
  openai      : text-embedding-3-small / text-embedding-3-large
  azure       : Azure OpenAI embeddings
  vertex      : GCP textembedding-gecko / text-embedding-004
  bedrock     : Amazon Titan Embeddings
  huggingface : Local sentence-transformers model (no API cost)
"""

import os
import logging

logger = logging.getLogger("financial_rag.embedding_factory")


def get_embeddings(provider: str, model: str):
    """
    Return a LangChain-compatible embedding model.

    Args:
        provider : 'openai' | 'azure' | 'vertex' | 'bedrock' | 'huggingface'
        model    : model name for the provider

    Returns:
        LangChain Embeddings instance
    """
    provider = provider.lower().strip()
    logger.info("Creating embeddings: provider=%s model=%s", provider, model)

    # ── OpenAI ────────────────────────────────────────────────────────────────
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=model,
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    # ── Azure OpenAI ──────────────────────────────────────────────────────────
    elif provider == "azure":
        from langchain_openai import AzureOpenAIEmbeddings
        return AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", model),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        )

    # ── GCP Vertex AI ─────────────────────────────────────────────────────────
    elif provider == "vertex":
        try:
            from langchain_google_vertexai import VertexAIEmbeddings
            return VertexAIEmbeddings(
                model_name=model,
                project=os.getenv("GCP_PROJECT_ID"),
                location=os.getenv("GCP_LOCATION", "us-central1"),
            )
        except ImportError:
            raise ImportError("pip install langchain-google-vertexai")

    # ── AWS Bedrock Titan ─────────────────────────────────────────────────────
    elif provider == "bedrock":
        try:
            bedrock_api_key = os.getenv("BEDROCK_API_KEY", "").strip()
            region          = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

            # Validate API key format — fall back to IAM if invalid
            _VALID_PREFIXES = ("bak-", "brk-", "bk-", "aws-", "bedr")
            key_valid = bedrock_api_key and any(
                bedrock_api_key.startswith(p) for p in _VALID_PREFIXES
            )
            if bedrock_api_key and not key_valid:
                import logging as _log
                _log.getLogger("financial_rag.embedding_factory").warning(
                    "BEDROCK_API_KEY format invalid — falling back to IAM credentials"
                )
                bedrock_api_key = ""   # force IAM fallback

            if bedrock_api_key:
                # Bedrock API key auth — custom embedding wrapper
                import requests, json, numpy as np
                from langchain_core.embeddings import Embeddings

                class BedrockAPIKeyEmbeddings(Embeddings):
                    def embed_documents(self, texts):
                        return [self._embed(t) for t in texts]

                    def embed_query(self, text):
                        return self._embed(text)

                    def _embed(self, text):
                        url = (
                            f"https://bedrock-runtime.{region}.amazonaws.com"
                            f"/model/{model}/invoke"
                        )
                        payload = {"inputText": text}
                        headers = {
                            "Content-Type":  "application/json",
                            "Authorization": f"Bearer {bedrock_api_key}",
                        }
                        resp = requests.post(url, headers=headers,
                                            data=json.dumps(payload), timeout=30)
                        if resp.status_code != 200:
                            raise RuntimeError(
                                f"Bedrock embedding error {resp.status_code}: {resp.text[:200]}"
                            )
                        return resp.json().get("embedding", [])

                return BedrockAPIKeyEmbeddings()

            else:
                # IAM auth — standard boto3
                from langchain_aws import BedrockEmbeddings
                import boto3
                client = boto3.client(
                    "bedrock-runtime",
                    region_name=region,
                )
                return BedrockEmbeddings(client=client, model_id=model)

        except ImportError:
            raise ImportError("pip install langchain-aws boto3")

    # ── Azure Embeddings ──────────────────────────────────────────────────────
    elif provider == "azure":
        try:
            from langchain_openai import AzureOpenAIEmbeddings
            # Azure embeddings use services.ai.azure.com endpoint (Foundry)
            # which is different from the LLM endpoint (openai.azure.com)
            endpoint   = os.getenv(
                "AZURE_EMBEDDING_ENDPOINT",
                os.getenv("AZURE_OPENAI_ENDPOINT", "")
            ).rstrip("/")
            api_key    = os.getenv(
                "AZURE_EMBEDDING_API_KEY",
                os.getenv("AZURE_OPENAI_API_KEY", "")
            )
            api_ver    = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
            deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", model)

            if not endpoint or not api_key:
                raise ValueError(
                    "Set AZURE_EMBEDDING_ENDPOINT + AZURE_EMBEDDING_API_KEY in .env"
                )

            # Foundry services endpoint uses OpenAI-compatible path
            if "services.ai.azure.com" in endpoint:
                from langchain_openai import OpenAIEmbeddings
                return OpenAIEmbeddings(
                    model=deployment,
                    api_key=api_key,
                    base_url=f"{endpoint}/openai/deployments/{deployment}",
                    default_query={"api-version": api_ver},
                )
            else:
                return AzureOpenAIEmbeddings(
                    azure_deployment=deployment,
                    azure_endpoint=endpoint,
                    api_key=api_key,
                    api_version=api_ver,
                )
        except ImportError:
            raise ImportError("pip install langchain-openai")

    # ── Google Generative AI Embeddings ───────────────────────────────────────
    elif provider == "google":
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not set in .env")
            # Google embedding models need "models/" prefix
            model_name = model if model.startswith("models/") else f"models/{model}"
            return GoogleGenerativeAIEmbeddings(
                model=model_name,
                google_api_key=api_key,
            )
        except ImportError:
            raise ImportError("pip install langchain-google-genai")

    # ── HuggingFace local ─────────────────────────────────────────────────────
    elif provider == "huggingface":
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=model)

    else:
        raise ValueError(
            f"Unsupported embedding provider: '{provider}'. "
            f"Choose from: openai, azure, vertex, bedrock, huggingface"
        )


def available_providers() -> list:
    return ["openai", "azure", "vertex", "bedrock", "huggingface"]


def default_model(provider: str) -> str:
    defaults = {
        "openai":      "text-embedding-3-small",
        "azure":       "text-embedding-3-small",
        "vertex":      "text-embedding-004",
        "bedrock":     "amazon.titan-embed-text-v2:0",
        "huggingface": "sentence-transformers/all-MiniLM-L6-v2",
    }
    return defaults.get(provider, "text-embedding-3-small")