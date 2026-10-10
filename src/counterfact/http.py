"""Capture and replay LLM API calls at the HTTP layer.

``instrument()`` wraps ``Client.send`` and ``AsyncClient.send`` of every installed
httpx-compatible library (``httpx`` and ``httpx2``; recent OpenAI and Anthropic SDKs
use the latter). Every request made inside a recording becomes a boundary call:
recorded live, served from the tape on replay, or replaced by an intervention.
Working at the HTTP layer avoids SDK internals, so it keeps working when SDK
classes are refactored.

Streaming responses are buffered while recording (the agent receives the same
bytes, all at once) so that they can be replayed exactly.
"""

from __future__ import annotations

import base64
import importlib
import json
import threading
from types import ModuleType
from typing import Any

from counterfact import runtime
from counterfact.genai import classify, request_attributes
from counterfact.model import SpanKind

_KEEP_HEADERS = ("content-type", "x-request-id", "request-id", "openai-model", "anthropic-version")
_LIBRARIES = ("httpx", "httpx2")
_lock = threading.Lock()
_originals: dict[str, tuple[ModuleType, Any, Any]] = {}


def _request_payload(request: Any) -> dict[str, Any]:
    content = request.content
    body: Any
    try:
        body = json.loads(content) if content else None
    except (UnicodeDecodeError, json.JSONDecodeError):
        body = {"__base64__": base64.b64encode(content).decode("ascii")}
    return {"method": request.method, "url": str(request.url.copy_with(query=None)), "body": body}


def _response_payload(response: Any) -> dict[str, Any]:
    content = response.content
    try:
        body: Any = json.loads(content) if content else None
        encoding = "json"
    except (UnicodeDecodeError, json.JSONDecodeError):
        try:
            body, encoding = content.decode("utf-8"), "text"
        except UnicodeDecodeError:
            body, encoding = base64.b64encode(content).decode("ascii"), "base64"
    headers = {k: v for k, v in response.headers.items() if k.lower() in _KEEP_HEADERS}
    return {"status": response.status_code, "headers": headers, "encoding": encoding, "body": body}


def _rebuild(lib: ModuleType, payload: dict[str, Any], request: Any) -> Any:
    encoding = payload.get("encoding", "json")
    body = payload.get("body")
    if encoding == "json":
        content = b"" if body is None else json.dumps(body).encode("utf-8")
    elif encoding == "text":
        content = str(body).encode("utf-8")
    else:
        content = base64.b64decode(str(body or ""))
    headers = dict(payload.get("headers") or {})
    if encoding == "json":
        headers.setdefault("content-type", "application/json")
    return lib.Response(
        payload.get("status", 200), headers=headers, content=content, request=request
    )


def _name(request: Any) -> str:
    provider, operation = classify(str(request.url))
    if provider and operation:
        return f"{provider}.{operation}"
    return f"http.{request.url.host or 'request'}"


def _kind(request: Any) -> SpanKind:
    _, operation = classify(str(request.url))
    return SpanKind.CHAT if operation in {"chat", "generate_content"} else SpanKind.TOOL


def _patch(lib: ModuleType) -> None:
    sync_original = lib.Client.send
    async_original = lib.AsyncClient.send

    def send(self: Any, request: Any, **kwargs: Any) -> Any:
        if runtime.current_run() is None:
            return sync_original(self, request, **kwargs)

        def live(payload: dict[str, Any]) -> dict[str, Any]:
            response = sync_original(self, request, **kwargs)
            response.read()
            return _response_payload(response)

        payload = _request_payload(request)
        recorded = runtime.call(
            _name(request),
            live,
            payload,
            kind=_kind(request),
            attributes=request_attributes(payload["url"], payload["body"]),
        )
        return _rebuild(lib, recorded, request)

    async def asend(self: Any, request: Any, **kwargs: Any) -> Any:
        if runtime.current_run() is None:
            return await async_original(self, request, **kwargs)

        async def live(payload: dict[str, Any]) -> dict[str, Any]:
            response = await async_original(self, request, **kwargs)
            await response.aread()
            return _response_payload(response)

        payload = _request_payload(request)
        recorded = await runtime.acall(
            _name(request),
            live,
            payload,
            kind=_kind(request),
            attributes=request_attributes(payload["url"], payload["body"]),
        )
        return _rebuild(lib, recorded, request)

    _originals[lib.__name__] = (lib, sync_original, async_original)
    lib.Client.send = send
    lib.AsyncClient.send = asend


def instrument() -> list[str]:
    """Route httpx-compatible traffic through Counterfact while a run is recorded.

    Returns the names of the libraries that were instrumented.
    """
    with _lock:
        for name in _LIBRARIES:
            if name in _originals:
                continue
            try:
                lib = importlib.import_module(name)
            except ImportError:
                continue
            _patch(lib)
        return sorted(_originals)


def uninstrument() -> None:
    with _lock:
        for lib, sync_original, async_original in _originals.values():
            lib.Client.send = sync_original
            lib.AsyncClient.send = async_original
        _originals.clear()


def chat_reply(text: str, *, provider: str = "openai", model: str = "replaced") -> dict[str, Any]:
    """A response payload for :class:`~counterfact.runtime.Replace` on a chat call."""
    if provider == "anthropic":
        body: dict[str, Any] = {
            "id": "msg_counterfactual",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
    else:
        body = {
            "id": "chatcmpl-counterfactual",
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": text},
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
    return {
        "status": 200,
        "headers": {"content-type": "application/json"},
        "encoding": "json",
        "body": body,
    }
