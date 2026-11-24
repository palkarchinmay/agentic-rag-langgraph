"""Step 9: vision-model dropdown choices and defaults (offline)."""
import importlib.util
from pathlib import Path

from app.config import vision_models as vm


def test_default_is_vision_model_env(monkeypatch):
    monkeypatch.delenv("VISION_MODEL", raising=False)
    assert vm.default_vision_spec() == "anthropic:claude-opus-5-5"
    monkeypatch.setenv("VISION_MODEL", "openai:gpt-4o")
    assert vm.default_vision_spec() == "openai:gpt-4o"
    monkeypatch.setenv("VISION_MODEL", "garbage")
    assert vm.default_vision_spec() == "anthropic:claude-opus-5-5"


def test_choices_start_with_default_and_keep_custom(monkeypatch):
    monkeypatch.setenv("VISION_MODEL", "bedrock:my-inference-profile")
    ch = vm.vision_model_choices("ollama:qwen2.5vl")
    assert ch[0] == "bedrock:my-inference-profile"
    assert "anthropic:claude-opus-5-5" in ch and "anthropic:claude-sonnet-5-5" in ch
    assert "anthropic:claude-haiku-5-5" in ch and "openai:gpt-4o-mini" in ch
    assert ch[-1] == "ollama:qwen2.5vl"
    assert len(ch) == len(set(ch))
    assert not any("claude-3" in c for c in ch)


def test_caption_provider_model(monkeypatch):
    monkeypatch.setenv("VISION_MODEL", "anthropic:claude-opus-5-5")
    assert vm.caption_provider_model({}) == ("anthropic", "claude-opus-5-5")
    # the chat LLM is NOT used as the vision default any more
    assert vm.caption_provider_model({"llm_provider": "openai", "llm_model": "gpt-4o-mini"}) \
        == ("anthropic", "claude-opus-5-5")
    assert vm.caption_spec({"caption_provider": "google", "caption_model": "gemini-2.5-flash"}) \
        == "google:gemini-2.5-flash"


OLD_WIDGET = '''import chainlit as cl

def build(config):
    return [
            cl.input_widget.Select(
                id="caption_model_full",
                label="🖼️ Vision/Caption Model (provider:model)",
                values=[
                    "openai:gpt-4o-mini", "openai:gpt-4o",
                    "anthropic:claude-3-5-sonnet", "anthropic:claude-3-haiku",
                ],
                initial_value=(
                    f"{config.get('caption_provider', config.get('llm_provider', 'openai'))}:"
                    f"{config.get('caption_model', config.get('llm_model', 'gpt-4o-mini'))}"
                ),
            ),
    ]
'''


def test_patch_script_rewrites_dropdown(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "apply_vision_model_ui.py"
    spec = importlib.util.spec_from_file_location("apply_ui", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    new, n = mod.DROPDOWN.subn(r'\1vision_model_choices(caption_spec(config)),\3caption_spec(config),',
                               OLD_WIDGET)
    assert n == 1
    assert "values=vision_model_choices(caption_spec(config))," in new
    assert "initial_value=caption_spec(config)," in new
    assert "claude-3" not in new
    new = mod.add_import(new)
    assert new.splitlines()[0].startswith("from app.config.vision_models import")
    compile(new, "w.py", "exec")
