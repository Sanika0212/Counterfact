"""LLM-as-judge baseline: ask a model which step caused a failure.

The model reads the trace (inputs and outputs of every step) and names the root
cause. Reading the trace is the usual approach to automated failure attribution,
so it serves as the non-interventional baseline for Counterfact's ``blame``.

Keys are read from the environment or a local ``.env`` file. They are never logged
or stored.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from counterfact.jsonable import canonical
from counterfact.model import Run

PROMPT = """An AI agent run failed: its final result was wrong. Below are the agent's \
steps in order, with each step's inputs and output.

Identify the ROOT CAUSE: the earliest step whose own mistake led to the failure. Do not \
name later steps that only passed along or reacted to an earlier mistake.

Steps:
{steps}

Reply with only a JSON object of the form {{"step": <step number>}}."""

DEFAULT_MODELS = {"anthropic": "claude-haiku-5-5", "openai": "gpt-4.1-mini"}


def load_env(path: str | Path = ".env") -> None:
    """Load ``KEY=VALUE`` lines from ``path`` into the environment (existing values win)."""
    env = Path(path)
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def trace_text(run: Run, max_chars: int = 400) -> str:
    lines = []
    for number, span in enumerate(run.boundaries):
        args = canonical(span.input)
        out = canonical(span.output)
        lines.append(
            f"[{number}] {span.name}\n    input: {args[:max_chars]}\n    output: {out[:max_chars]}"
        )
    return "\n".join(lines)


class JudgeError(RuntimeError):
    """The model API refused the request (no credit, bad key, unknown model, ...)."""


# Statuses worth retrying: rate limits, overload and transient server errors.
_RETRY = {429, 500, 502, 503, 504, 529}
# 429s that retrying cannot fix.
_FATAL_CODES = {"insufficient_quota", "credit_balance_exhausted", "billing_hard_limit_reached"}


def _api_error(response: httpx.Response) -> tuple[str, str]:
    """``(code, message)`` from a provider error body, never echoing request headers."""
    try:
        error = response.json().get("error", {})
    except ValueError:
        return "", response.reason_phrase
    if not isinstance(error, dict):
        return "", str(error)[:300]
    code = str(error.get("code") or error.get("type") or "")
    return code, str(error.get("message") or "")[:300]


@dataclass
class Judge:
    provider: str
    model: str
    api_key: str
    client: httpx.Client | None = None
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    max_retries: int = 4
    sleep: Callable[[float], None] = time.sleep

    @classmethod
    def from_env(cls, env_file: str | Path = ".env") -> Judge | None:
        load_env(env_file)
        provider = os.environ.get("COUNTERFACT_JUDGE_PROVIDER")
        if provider is None:
            if os.environ.get("ANTHROPIC_API_KEY"):
                provider = "anthropic"
            elif os.environ.get("OPENAI_API_KEY"):
                provider = "openai"
            else:
                return None
        key = os.environ.get("ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY")
        if not key:
            return None
        model = os.environ.get("COUNTERFACT_JUDGE_MODEL", DEFAULT_MODELS[provider])
        return cls(provider=provider, model=model, api_key=key)

    def _http(self) -> httpx.Client:
        if self.client is None:
            self.client = httpx.Client(timeout=60.0)
        return self.client

    def _post(self, url: str, headers: dict[str, str], payload: dict[str, Any]) -> Any:
        """POST with retries on rate limits and transient errors; JudgeError otherwise."""
        for attempt in range(self.max_retries + 1):
            response = self._http().post(url, headers=headers, json=payload)
            if response.is_success:
                return response.json()
            code, message = _api_error(response)
            retryable = response.status_code in _RETRY and code not in _FATAL_CODES
            if not retryable or attempt == self.max_retries:
                detail = f"{code}: {message}" if code else message
                raise JudgeError(f"{self.provider} API returned {response.status_code} ({detail})")
            wait = response.headers.get("retry-after")
            self.sleep(float(wait) if wait and wait.replace(".", "", 1).isdigit() else 2.0**attempt)
        raise AssertionError("unreachable")  # pragma: no cover

    def ask(self, prompt: str) -> str:
        self.calls += 1
        if self.provider == "anthropic":
            body = self._post(
                "https://api.anthropic.com/v1/messages",
                {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
                {
                    "model": self.model,
                    "max_tokens": 64,
                    "temperature": 0,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            usage = body.get("usage", {})
            self.input_tokens += usage.get("input_tokens", 0)
            self.output_tokens += usage.get("output_tokens", 0)
            return "".join(b.get("text", "") for b in body.get("content", []))
        base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        body = self._post(
            f"{base}/chat/completions",
            {"Authorization": f"Bearer {self.api_key}"},
            {
                "model": self.model,
                "temperature": 0,
                "max_tokens": 64,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        usage = body.get("usage", {})
        self.input_tokens += usage.get("prompt_tokens", 0)
        self.output_tokens += usage.get("completion_tokens", 0)
        return str(body["choices"][0]["message"]["content"] or "")

    def blame(self, run: Run) -> str | None:
        """Boundary key the model names as root cause (``None`` if unparseable)."""
        reply = self.ask(PROMPT.format(steps=trace_text(run)))
        match = re.search(r"\{[^{}]*\}", reply)
        try:
            step = int(json.loads(match.group(0))["step"]) if match else int(reply.strip())
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            return None
        spans = run.boundaries
        if not 0 <= step < len(spans) or spans[step].boundary is None:
            return None
        return spans[step].boundary.key  # type: ignore[union-attr]
