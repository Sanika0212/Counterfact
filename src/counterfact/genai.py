"""OpenTelemetry GenAI attributes derived from recorded HTTP exchanges.

Attribute names follow the OpenTelemetry GenAI semantic conventions, which are
still in Development status; :data:`SEMCONV_VERSION` records the version the
names were taken from.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

SEMCONV_VERSION = "1.41"

_PROVIDERS = {
    "api.openai.com": "openai",
    "api.anthropic.com": "anthropic",
    "generativelanguage.googleapis.com": "gcp.gemini",
    "api.mistral.ai": "mistral_ai",
    "api.groq.com": "groq",
    "api.deepseek.com": "deepseek",
}


def classify(url: str) -> tuple[str, str]:
    """``(provider, operation)`` for a request URL, e.g. ``("openai", "chat")``."""
    parsed = urlparse(url)
    host = parsed.hostname or ""
    path = parsed.path.rstrip("/")
    provider = _PROVIDERS.get(host, "")
    if path.endswith("/chat/completions"):
        return provider or "openai_compatible", "chat"
    if path.endswith("/v1/messages") or path.endswith("/messages"):
        return provider or "anthropic", "chat"
    if path.endswith("/responses"):
        return provider or "openai", "generate_content"
    if path.endswith("/embeddings"):
        return provider or "openai_compatible", "embeddings"
    if ":generateContent" in path or ":streamGenerateContent" in path:
        return provider or "gcp.gemini", "generate_content"
    return provider, ""


def request_attributes(url: str, body: Any) -> dict[str, Any]:
    provider, operation = classify(url)
    attrs: dict[str, Any] = {}
    if provider:
        attrs["gen_ai.provider.name"] = provider
    if operation:
        attrs["gen_ai.operation.name"] = operation
    if isinstance(body, dict):
        for src, dst in (
            ("model", "gen_ai.request.model"),
            ("temperature", "gen_ai.request.temperature"),
            ("max_tokens", "gen_ai.request.max_tokens"),
            ("top_p", "gen_ai.request.top_p"),
            ("seed", "gen_ai.request.seed"),
        ):
            if body.get(src) is not None:
                attrs[dst] = body[src]
        if body.get("stream"):
            attrs["counterfact.stream"] = True
    return attrs


def response_attributes(body: Any) -> dict[str, Any]:
    """Usage, finish reasons and tool calls from an OpenAI- or Anthropic-style body."""
    attrs: dict[str, Any] = {}
    if not isinstance(body, dict):
        return attrs
    if body.get("id"):
        attrs["gen_ai.response.id"] = body["id"]
    if body.get("model"):
        attrs["gen_ai.response.model"] = body["model"]
    usage = body.get("usage") or {}
    input_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
    output_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
    if input_tokens is not None:
        attrs["gen_ai.usage.input_tokens"] = input_tokens
    if output_tokens is not None:
        attrs["gen_ai.usage.output_tokens"] = output_tokens
    reasons: list[str] = []
    tools: list[str] = []
    for choice in body.get("choices") or []:
        if choice.get("finish_reason"):
            reasons.append(choice["finish_reason"])
        for call in (choice.get("message") or {}).get("tool_calls") or []:
            name = (call.get("function") or {}).get("name")
            if name:
                tools.append(name)
    if body.get("stop_reason"):
        reasons.append(body["stop_reason"])
    for block in body.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("name"):
            tools.append(block["name"])
    if reasons:
        attrs["gen_ai.response.finish_reasons"] = reasons
    if tools:
        attrs["counterfact.tool_calls"] = tools
    return attrs
