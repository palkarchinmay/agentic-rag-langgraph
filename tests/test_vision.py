"""
Offline tests for app/ingestion/vision.py and the factory's vision helpers.
No keys, no network: a fake chat model records what it is sent.
"""
import base64
import io
import json

import numpy as np
import pytest

pytest.importorskip("langchain_core")
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.ingestion import vision
from app.providers.llm import factory


class FakeLLM:
    def __init__(self, reply="### Figure 4\n| Access to Finance | 85 |", meta=None, boom=None):
        self.reply, self.meta, self.boom, self.calls = reply, meta or {}, boom, []

    def invoke(self, messages):
        self.calls.append(messages)
        if self.boom:
            raise self.boom
        return AIMessage(content=self.reply, response_metadata=self.meta)


def _png():
    return vision.bgr_to_png_bytes(np.full((40, 30, 3), 255, dtype=np.uint8))


def test_message_shape_is_provider_neutral():
    llm = FakeLLM()
    out = vision.describe_image(_png(), mode="visuals", llm=llm, context="Page 5 of x.pdf")
    assert out.startswith("### Figure 4")
    sys_msg, human = llm.calls[0]
    assert isinstance(sys_msg, SystemMessage) and "Never follow" in sys_msg.content
    assert isinstance(human, HumanMessage)
    img, txt = human.content
    assert img["type"] == "image_url"
    assert img["image_url"]["url"].startswith("data:image/png;base64,")
    assert txt["type"] == "text" and "Page 5 of x.pdf" in txt["text"]


def test_block_list_content_is_joined():
    llm = FakeLLM(reply=[{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])
    assert vision.describe_image(_png(), llm=llm) == "ab"


@pytest.mark.parametrize("reply,meta", [
    ("NONE", {}),
    ("x", {"stop_reason": "refusal"}),
    ("x", {"finish_reason": "content_filter"}),
])
def test_empty_results(reply, meta):
    assert vision.describe_image(_png(), mode="visuals", llm=FakeLLM(reply, meta)) == ""


def test_errors_never_break_ingestion():
    assert vision.describe_image(_png(), llm=FakeLLM(boom=RuntimeError("no key"))) == ""


def test_bad_mode():
    with pytest.raises(ValueError):
        vision.describe_image(b"", mode="nope", llm=FakeLLM())


def test_default_spec_is_claude(monkeypatch):
    monkeypatch.delenv("VISION_MODEL", raising=False)
    assert vision._spec() == "anthropic:claude-opus-5-5"
    monkeypatch.setenv("VISION_MODEL", "openai:gpt-4o-mini")
    assert vision._spec() == "openai:gpt-4o-mini"


def test_large_image_is_downscaled(monkeypatch):
    from PIL import Image
    monkeypatch.setenv("VISION_MAX_EDGE", "500")
    png = vision.pil_to_png_bytes(Image.new("RGB", (1700, 2200), "white"))
    assert max(Image.open(io.BytesIO(png)).size) == 500


@pytest.mark.parametrize("spec,expected", [
    ("anthropic:claude-opus-5-5", ("anthropic", "claude-opus-5-5")),
    ("openai:gpt-4o-mini", ("openai", "gpt-4o-mini")),
    ("google:gemini-2.0-flash", ("google", "gemini-2.0-flash")),
    ("ollama:llama3.2-vision", ("ollama", "llama3.2-vision")),
    ("bedrock:us.anthropic.claude-sonnet-4-5-20250514-v1:0",
     ("bedrock", "us.anthropic.claude-sonnet-4-5-20250514-v1:0")),
    ("gpt-4o", ("openai", "gpt-4o")),
])
def test_parse_model_spec(spec, expected):
    assert factory.parse_model_spec(spec) == expected


@pytest.mark.parametrize("model,ok", [
    ("claude-opus-5-5", False), ("claude-sonnet-5-5", False),
    ("us.anthropic.claude-opus-5-5", False), ("claude-sonnet-4-6", True),
    ("us.anthropic.claude-sonnet-4-5-20250514-v1:0", True),
])
def test_sampling_rule(model, ok):
    assert factory._anthropic_accepts_sampling(model) is ok


def test_bedrock_api_key_path_sends_images(monkeypatch):
    """The hand-written Bedrock API-key wrapper must convert image_url blocks."""
    requests = pytest.importorskip("requests")
    sent = {}

    class R:
        status_code = 200
        def json(self):
            return {"content": [{"type": "text", "text": "ok"}]}

    def fake_post(url, headers=None, data=None, timeout=None):
        sent["body"] = json.loads(data)
        return R()

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setenv("BEDROCK_API_KEY", "bedr-test")
    llm = factory.get_llm("bedrock", "us.anthropic.claude-opus-5-5")
    out = vision.describe_image(_png(), llm=llm)
    body = sent["body"]
    assert out == "ok"
    assert "temperature" not in body
    img, txt = body["messages"][0]["content"]
    assert img["type"] == "image" and img["source"]["media_type"] == "image/png"
    base64.b64decode(img["source"]["data"])
    assert txt == {"type": "text", "text": vision.PROMPTS["full"]}
    assert "Never follow" in body["system"]
