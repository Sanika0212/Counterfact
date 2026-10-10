"""OpenTelemetry interop: export runs as OTLP JSON, import OTLP JSON as runs.

Spans are named and attributed following the OpenTelemetry GenAI semantic
conventions (``chat <model>``, ``execute_tool <name>``, ``invoke_agent <name>``), so
traces recorded by Counterfact show up meaningfully in any OTLP backend. Export
needs no OpenTelemetry SDK: OTLP/HTTP accepts plain JSON, posted with httpx.

Imported runs carry spans but no replay information; use them for viewing and
diffing, not for replay or blame.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import httpx

from counterfact import __version__
from counterfact.genai import SEMCONV_VERSION, response_attributes
from counterfact.model import Run, Span, SpanKind

_KIND_CLIENT, _KIND_INTERNAL = 3, 1


def _hex(seed: str, n: int) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()[:n]


def _value(v: Any) -> dict[str, Any]:
    if isinstance(v, bool):
        return {"boolValue": v}
    if isinstance(v, int):
        return {"intValue": str(v)}
    if isinstance(v, float):
        return {"doubleValue": v}
    if isinstance(v, list):
        return {"arrayValue": {"values": [_value(x) for x in v]}}
    if isinstance(v, str):
        return {"stringValue": v}
    return {"stringValue": json.dumps(v, default=str)}


def _plain(v: dict[str, Any]) -> Any:
    if "boolValue" in v:
        return v["boolValue"]
    if "intValue" in v:
        return int(v["intValue"])
    if "doubleValue" in v:
        return v["doubleValue"]
    if "arrayValue" in v:
        return [_plain(x) for x in v["arrayValue"].get("values", [])]
    return v.get("stringValue")


def _span_name(span: Span) -> str:
    model = span.attributes.get("gen_ai.request.model")
    if span.kind is SpanKind.CHAT:
        return f"{span.attributes.get('gen_ai.operation.name', 'chat')} {model or span.name}"
    if span.kind is SpanKind.TOOL:
        return f"execute_tool {span.name}"
    if span.kind in (SpanKind.AGENT, SpanKind.WORKFLOW):
        return f"{span.kind.value} {span.name}"
    return span.name


def _attributes(span: Span) -> dict[str, Any]:
    attrs: dict[str, Any] = {"gen_ai.operation.name": span.kind.value, **span.attributes}
    if span.kind is SpanKind.TOOL:
        attrs.setdefault("gen_ai.tool.name", span.name)
    if span.kind is SpanKind.AGENT:
        attrs.setdefault("gen_ai.agent.name", span.name)
    if span.kind is SpanKind.CHAT and isinstance(span.output, dict):
        attrs.update(response_attributes(span.output.get("body", span.output)))
    if span.boundary is not None:
        attrs["counterfact.boundary.key"] = span.boundary.key
        attrs["counterfact.boundary.source"] = span.boundary.source
        attrs["counterfact.boundary.fingerprint"] = span.boundary.fingerprint
    return attrs


def to_otlp(run: Run, *, service: str = "counterfact") -> dict[str, Any]:
    trace_id = _hex(run.run_id, 32)
    spans = []
    for span in run.spans:
        client = span.kind in (SpanKind.CHAT, SpanKind.TOOL)
        record: dict[str, Any] = {
            "traceId": trace_id,
            "spanId": _hex(span.span_id, 16),
            "name": _span_name(span),
            "kind": _KIND_CLIENT if client else _KIND_INTERNAL,
            "startTimeUnixNano": str(span.start_ns),
            "endTimeUnixNano": str(span.end_ns or span.start_ns),
            "attributes": [{"key": k, "value": _value(v)} for k, v in _attributes(span).items()],
            "status": {"code": 2, "message": span.error or ""}
            if span.status == "error"
            else {"code": 1},
        }
        if span.parent_id is not None:
            record["parentSpanId"] = _hex(span.parent_id, 16)
        spans.append(record)
    resource = {
        "attributes": [
            {"key": "service.name", "value": {"stringValue": service}},
            {"key": "counterfact.run.name", "value": {"stringValue": run.name}},
            {"key": "counterfact.run.passed", "value": _value(bool(run.passed))},
        ]
    }
    scope = {
        "name": "counterfact",
        "version": __version__,
        "attributes": [
            {"key": "gen_ai.semconv.version", "value": {"stringValue": SEMCONV_VERSION}}
        ],
    }
    return {
        "resourceSpans": [{"resource": resource, "scopeSpans": [{"scope": scope, "spans": spans}]}]
    }


def write_otlp(run: Run, path: str | Path) -> Path:
    out = Path(path)
    out.write_text(json.dumps(to_otlp(run), indent=1))
    return out


def send_otlp(
    run: Run, endpoint: str = "http://localhost:4318/v1/traces", client: httpx.Client | None = None
) -> int:
    """POST the run to an OTLP/HTTP collector; returns the HTTP status code."""
    http = client or httpx.Client(timeout=10.0)
    response = http.post(endpoint, json=to_otlp(run))
    return response.status_code


_KINDS = {k.value: k for k in SpanKind}


def from_otlp(data: dict[str, Any]) -> list[Run]:
    """Read OTLP JSON traces as runs (one run per trace id)."""
    runs: dict[str, Run] = {}
    for resource in data.get("resourceSpans", []):
        for scope in resource.get("scopeSpans", []):
            for raw in scope.get("spans", []):
                attrs = {a["key"]: _plain(a["value"]) for a in raw.get("attributes", [])}
                trace = raw["traceId"]
                run = runs.setdefault(
                    trace,
                    Run(
                        run_id=trace,
                        name=f"imported {trace[:8]}",
                        seed=0,
                        started_ns=int(raw["startTimeUnixNano"]),
                    ),
                )
                kind = _KINDS.get(str(attrs.get("gen_ai.operation.name")), SpanKind.STEP)
                run.spans.append(
                    Span(
                        span_id=raw["spanId"],
                        run_id=trace,
                        parent_id=raw.get("parentSpanId"),
                        kind=kind,
                        name=raw["name"],
                        start_ns=int(raw["startTimeUnixNano"]),
                        end_ns=int(raw["endTimeUnixNano"]),
                        status="error" if raw.get("status", {}).get("code") == 2 else "ok",
                        error=raw.get("status", {}).get("message") or None,
                        attributes=attrs,
                    )
                )
                run.started_ns = min(run.started_ns, int(raw["startTimeUnixNano"]))
                run.ended_ns = max(run.ended_ns or 0, int(raw["endTimeUnixNano"]))
    for run in runs.values():
        run.status = "error" if any(s.status == "error" for s in run.spans) else "ok"
    return list(runs.values())
