"""Redact secrets and personal data before anything is written to disk.

Rules are applied to every string inside JSON-like payloads, and values under
sensitive keys (``authorization``, ``api_key``, ...) are dropped entirely.
Redaction affects stored copies only; the running agent always sees real values.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

SENSITIVE_KEYS = frozenset(
    {
        "authorization",
        "api_key",
        "apikey",
        "x-api-key",
        "api-key",
        "password",
        "secret",
        "token",
        "access_token",
        "refresh_token",
        "cookie",
        "set-cookie",
    }
)


def _luhn_ok(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: re.Pattern[str]
    accept: Callable[[str], bool] = lambda _: True


def _card_number(match: str) -> bool:
    digits = re.sub(r"\D", "", match)
    return 13 <= len(digits) <= 19 and _luhn_ok(digits)


DEFAULT_RULES: tuple[Rule, ...] = (
    Rule("anthropic_key", re.compile(r"sk-ant-[A-Za-z0-9_\-]{16,}")),
    Rule("openai_key", re.compile(r"sk-(?:proj-|svcacct-)?[A-Za-z0-9_\-]{20,}")),
    Rule("google_key", re.compile(r"AIza[0-9A-Za-z_\-]{35}")),
    Rule("aws_access_key", re.compile(r"(?:AKIA|ASIA)[0-9A-Z]{16}")),
    Rule(
        "github_token", re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})")
    ),
    Rule("jwt", re.compile(r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")),
    Rule("bearer", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9\-._~+/]{16,}=*")),
    Rule("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    Rule(
        "phone",
        re.compile(r"(?<![\d\w])(?:\+\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?![\d\w])"),
    ),
    Rule("card_number", re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)"), _card_number),
)


@dataclass
class Redactor:
    rules: tuple[Rule, ...] = DEFAULT_RULES
    sensitive_keys: frozenset[str] = SENSITIVE_KEYS
    counts: dict[str, int] = field(default_factory=dict)

    @classmethod
    def without(cls, names: Iterable[str]) -> Redactor:
        drop = set(names)
        return cls(rules=tuple(r for r in DEFAULT_RULES if r.name not in drop))

    def text(self, value: str) -> str:
        for rule in self.rules:

            def replace(match: re.Match[str], rule: Rule = rule) -> str:
                if not rule.accept(match.group(0)):
                    return match.group(0)
                self.counts[rule.name] = self.counts.get(rule.name, 0) + 1
                return f"[REDACTED:{rule.name}]"

            value = rule.pattern.sub(replace, value)
        return value

    def __call__(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, Mapping):
            out = {}
            for key, item in value.items():
                if str(key).lower() in self.sensitive_keys and item not in (None, ""):
                    self.counts["sensitive_key"] = self.counts.get("sensitive_key", 0) + 1
                    out[key] = "[REDACTED:sensitive_key]"
                else:
                    out[key] = self(item)
            return out
        if isinstance(value, list):
            return [self(item) for item in value]
        return value
