"""
vision_models.py -- the choices for the UI's "Vision/Caption Model" dropdown,
and the default when a user hasn't picked one.

The default is VISION_MODEL from the environment (provider:model), so the
dropdown, the upload path and the API all agree with the ingestion code
(app/ingestion/parser.py falls back to the same variable). A VISION_MODEL
that isn't in the list below -- a Bedrock inference profile, a local Ollama
model, an Azure deployment name -- is added to the list automatically, so
any provider the LLM factory supports can be selected.
"""

import os

DEFAULT_VISION_MODEL = "anthropic:claude-opus-5-5"

VISION_MODEL_CHOICES = [
    "anthropic:claude-opus-5-5",
    "anthropic:claude-sonnet-5-5",
    "anthropic:claude-haiku-5-5",
    "openai:gpt-4o",
    "openai:gpt-4o-mini",
    "google:gemini-2.5-flash",
    "google:gemini-2.0-flash",
    "azure:gpt-4o",
    "azure:gpt-4o-mini",
    "ollama:llama3.2-vision",
]


def default_vision_spec() -> str:
    spec = (os.getenv("VISION_MODEL") or "").strip()
    return spec if ":" in spec else DEFAULT_VISION_MODEL


def vision_model_choices(current: str = "") -> list:
    """The dropdown values: VISION_MODEL first, then the list above, plus
    the user's current pick if it is something else."""
    out = []
    for spec in [default_vision_spec(), *VISION_MODEL_CHOICES, current]:
        if spec and ":" in spec and spec not in out:
            out.append(spec)
    return out


def caption_provider_model(config: dict) -> tuple:
    """(provider, model) for captioning: the user's saved pick, else
    VISION_MODEL. Never falls back to the chat LLM, which may not see images."""
    provider = (config or {}).get("caption_provider")
    model = (config or {}).get("caption_model")
    if provider and model:
        return provider, model
    p, m = default_vision_spec().split(":", 1)
    return p, m


def caption_spec(config: dict) -> str:
    return "%s:%s" % caption_provider_model(config)
