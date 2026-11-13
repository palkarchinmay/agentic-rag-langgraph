"""
llm/factory.py — Multi-cloud LLM provider factory.

Bedrock supports two auth methods:
  1. IAM keys (AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY) — traditional
  2. Bedrock API key (BEDROCK_API_KEY) — newer, simpler, like OpenAI keys
     Uses requests directly against the Bedrock endpoint with x-api-key header
"""

import os
import logging

logger = logging.getLogger("financial_rag.llm_factory")


def get_llm(provider: str, model: str, temperature: float = 0.0):
    provider = provider.lower().strip()
    logger.info("Creating LLM: provider=%s model=%s", provider, model)

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=temperature,
        )

    elif provider == "azure":
        endpoint   = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
        api_key    = os.getenv("AZURE_OPENAI_API_KEY")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", model)
        api_ver    = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")

        # Azure AI Foundry endpoints end with /openai/v1
        # They are fully OpenAI-SDK compatible — use ChatOpenAI with base_url
        # The endpoint from "Get code" is the full base URL including /openai/v1
        #
        # Example: https://<your-resource>.openai.azure.com/openai/v1
        # Chat completions hits: <base_url>/chat/completions
        #
        # Standard Azure OpenAI endpoints do NOT include /openai/v1
        # Example: https://myresource.openai.azure.com
        # These use AzureChatOpenAI with api-version query param

        from langchain_openai import ChatOpenAI

        # Use the endpoint exactly as provided in AZURE_OPENAI_ENDPOINT
        # If it already contains /openai/v1, use it directly as base_url
        # If not, it is a classic endpoint — append deployment path
        if "/openai/v1" in endpoint:
            # Foundry-style: base_url is the full endpoint as-is
            base_url = endpoint
        else:
            # Classic style: build OpenAI-compatible base URL
            base_url = f"{endpoint}/openai/v1"

        # Never pass temperature=0.0 — newer models (gpt-5-mini) reject it.
        # Only pass temperature when user explicitly sets it above default.
        from langchain_openai import ChatOpenAI as _CO
        if temperature > 0.0:
            return _CO(model=deployment, api_key=api_key,
                       base_url=base_url, temperature=temperature)
        return _CO(model=deployment, api_key=api_key, base_url=base_url)

    elif provider == "bedrock":
        return _get_bedrock_llm(model, temperature)

    elif provider == "vertex":
        # Vertex AI — uses GCP service account credentials
        # Requires: GOOGLE_APPLICATION_CREDENTIALS, GCP_PROJECT_ID, GCP_LOCATION
        # Enable API: console.cloud.google.com → Vertex AI API → Enable
        try:
            from langchain_google_vertexai import ChatVertexAI
            project  = os.getenv("GCP_PROJECT_ID")
            location = os.getenv("GCP_LOCATION", "us-central1")
            if not project:
                raise ValueError(
                    "GCP_PROJECT_ID not set in .env\n"
                    "Also ensure GOOGLE_APPLICATION_CREDENTIALS points to your service account JSON\n"
                    "And enable Vertex AI API: console.cloud.google.com/apis → Vertex AI API"
                )
            # temperature=0 not supported by some Gemini models on Vertex
            kwargs = dict(model_name=model, project=project, location=location)
            if temperature > 0.0:
                kwargs["temperature"] = temperature
            return ChatVertexAI(**kwargs)
        except ImportError:
            raise ImportError("pip install langchain-google-vertexai")

    elif provider == "google":
        # Google AI Studio — uses simple API key (free tier available)
        # Get key: aistudio.google.com/app/apikey
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError(
                    "GOOGLE_API_KEY not set in .env\n"
                    "Get free key: aistudio.google.com/app/apikey"
                )
            # temperature=0 not supported by some Gemini models
            kwargs = dict(model=model, google_api_key=api_key)
            if temperature > 0.0:
                kwargs["temperature"] = temperature
            return ChatGoogleGenerativeAI(**kwargs)
        except ImportError:
            raise ImportError("pip install langchain-google-genai")

    elif provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY not set in .env\n"
                    "Get key: console.anthropic.com → API Keys"
                )
            return ChatAnthropic(
                model=model,
                api_key=api_key,
                temperature=temperature,
            )
        except ImportError:
            raise ImportError("pip install langchain-anthropic")

    else:
        raise ValueError(f"Unsupported LLM provider: '{provider}'")


def _get_bedrock_llm(model: str, temperature: float):
    """
    Support both Bedrock auth methods:
      - BEDROCK_API_KEY env var  → use API key auth (newer, simpler)
      - AWS_ACCESS_KEY_ID env var → use IAM auth (traditional boto3)
    """
    bedrock_api_key = os.getenv("BEDROCK_API_KEY", "").strip()
    aws_access_key  = os.getenv("AWS_ACCESS_KEY_ID", "").strip()

    # AWS Bedrock API keys (introduced July 2025) must start with a known prefix
    # Old keys or incorrectly formatted keys fall back to IAM credentials
    _VALID_PREFIXES = ("bak-", "brk-", "bk-", "aws-", "bedr")
    key_valid = bedrock_api_key and any(
        bedrock_api_key.startswith(p) for p in _VALID_PREFIXES
    )

    if key_valid:
        return _bedrock_with_api_key(model, temperature, bedrock_api_key)
    elif bedrock_api_key and not key_valid:
        import logging as _log
        _log.getLogger("financial_rag.llm_factory").warning(
            "BEDROCK_API_KEY format appears invalid (must start with a "
            "recognised prefix). Falling back to IAM credentials. "
            "Generate a new key: AWS Console → Amazon Bedrock → API Keys"
        )
        if aws_access_key:
            return _bedrock_with_iam(model, temperature)
        raise ValueError(
            "BEDROCK_API_KEY format invalid and no IAM credentials found.\n"
            "Either generate a new key at AWS Console → Bedrock → API Keys\n"
            "or set AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY"
        )
    elif aws_access_key:
        return _bedrock_with_iam(model, temperature)
    else:
        raise ValueError(
            "No Bedrock credentials found. Set either:\n"
            "  BEDROCK_API_KEY=your-bedrock-api-key  (newer API key)\n"
            "  AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY  (IAM keys)"
        )


def _bedrock_with_api_key(model: str, temperature: float, api_key: str):
    """
    Bedrock API key auth — uses a custom LangChain wrapper that
    calls the Bedrock endpoint directly with x-api-key header.

    Endpoint: https://bedrock.us-east-1.amazonaws.com
    Auth header: x-api-key: <your-bedrock-api-key>
    """
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    import json, requests

    region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

    class BedrockAPIKeyLLM(BaseChatModel):
        """Minimal Bedrock LLM using API key auth."""

        model_id:    str   = model
        api_key_val: str   = api_key
        temp:        float = temperature
        region_name: str   = region

        @property
        def _llm_type(self) -> str:
            return "bedrock-api-key"

        def _is_claude(self) -> bool:
            return "anthropic" in self.model_id.lower()

        def _is_nova(self) -> bool:
            return "nova" in self.model_id.lower()

        def _build_payload(self, messages):
            """Build correct payload based on model family."""
            system_content = ""
            user_messages  = []

            for msg in messages:
                if isinstance(msg, SystemMessage):
                    system_content = msg.content
                elif isinstance(msg, HumanMessage):
                    user_messages.append({
                        "role": "user",
                        "content": [{"text": msg.content}] if self._is_nova()
                                   else msg.content
                    })
                elif isinstance(msg, AIMessage):
                    user_messages.append({
                        "role": "assistant",
                        "content": [{"text": msg.content}] if self._is_nova()
                                   else msg.content
                    })

            if not user_messages:
                last = str(messages[-1].content)
                user_messages = [{
                    "role": "user",
                    "content": [{"text": last}] if self._is_nova() else last
                }]

            if self._is_claude():
                # Anthropic Claude format
                payload = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 2048,
                    "temperature": self.temp,
                    "messages": user_messages,
                }
                if system_content:
                    payload["system"] = system_content

            elif self._is_nova():
                # Amazon Nova / Converse API format
                payload = {
                    "messages": user_messages,
                    "inferenceConfig": {
                        "maxTokens":   2048,
                        "temperature": self.temp,
                    },
                }
                if system_content:
                    payload["system"] = [{"text": system_content}]

            else:
                # Generic Titan / other Amazon models
                all_text = " ".join(
                    m.content if isinstance(m.content, str)
                    else str(m.content)
                    for m in messages
                )
                payload = {
                    "inputText": all_text,
                    "textGenerationConfig": {
                        "maxTokenCount": 2048,
                        "temperature":   self.temp,
                    },
                }

            return payload

        def _extract_response(self, body: dict) -> str:
            """Extract text from response based on model family."""
            if self._is_claude():
                return body.get("content", [{}])[0].get("text", "")
            elif self._is_nova():
                # Converse API response format
                return (body.get("output", {})
                            .get("message", {})
                            .get("content", [{}])[0]
                            .get("text", ""))
            else:
                # Titan format
                return (body.get("results", [{}])[0]
                            .get("outputText", ""))

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            payload = self._build_payload(messages)

            url = (
                f"https://bedrock-runtime.{self.region_name}.amazonaws.com"
                f"/model/{self.model_id}/invoke"
            )
            headers = {
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {self.api_key_val}",
            }

            resp = requests.post(url, headers=headers,
                                 data=json.dumps(payload), timeout=60)

            if resp.status_code != 200:
                raise RuntimeError(
                    f"Bedrock API error {resp.status_code}: {resp.text[:300]}"
                )

            body    = resp.json()
            content = self._extract_response(body)
            message = AIMessage(content=content)
            return ChatResult(generations=[ChatGeneration(message=message)])

        async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
            return self._generate(messages, stop=stop, **kwargs)

    return BedrockAPIKeyLLM()


def _bedrock_with_iam(model: str, temperature: float):
    """Bedrock IAM auth — standard boto3/langchain-aws."""
    try:
        from langchain_aws import ChatBedrock
        import boto3

        client = boto3.client(
            "bedrock-runtime",
            region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        )
        return ChatBedrock(
            client=client,
            model_id=model,
            model_kwargs={"temperature": temperature},
        )
    except ImportError:
        raise ImportError("pip install langchain-aws boto3")


def available_providers() -> list:
    return ["openai", "azure", "bedrock", "vertex", "google"]


def default_model(provider: str) -> str:
    return {
        "openai":  "gpt-4o-mini",
        "azure":   "gpt-4o",
        "bedrock": "us.anthropic.claude-sonnet-4-5-20250514-v1:0",
        "vertex":  "gemini-2.0-flash-001",
        "google":  "gemini-2.0-flash",
    }.get(provider, "gpt-4o-mini")