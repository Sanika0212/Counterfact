"""Runs, spans and tapes.

Span kinds follow the OpenTelemetry GenAI semantic conventions (operation names
``invoke_agent``, ``invoke_workflow``, ``execute_tool``, ``chat``), plus a generic
``step`` for user code. A *boundary* span is a nondeterministic call (an LLM call, a
tool, a random draw) whose output Counterfact records and can replay or intervene on.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from counterfact.jsonable import to_jsonable


class SpanKind(StrEnum):
    AGENT = "invoke_agent"
    WORKFLOW = "invoke_workflow"
    TOOL = "execute_tool"
    CHAT = "chat"
    STEP = "step"


Source = Literal["live", "tape", "intervention"]


@dataclass(frozen=True)
class Boundary:
    """Replay metadata for a nondeterministic call."""

    index: int  # order among boundary calls in the run
    key: str  # "<name>#<occurrence>", stable across replays with the same control flow
    fingerprint: str  # hash of the call's inputs
    seed: int  # exogenous noise used by the call
    source: Source  # where the output came from


@dataclass
class Span:
    span_id: str
    run_id: str
    parent_id: str | None
    kind: SpanKind
    name: str
    start_ns: int
    end_ns: int | None = None
    status: Literal["ok", "error"] = "ok"
    error: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    input: Any = None
    output: Any = None
    boundary: Boundary | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind.value
        data["input"] = to_jsonable(self.input)
        data["output"] = to_jsonable(self.output)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Span:
        data = dict(data)
        data["kind"] = SpanKind(data["kind"])
        if data.get("boundary") is not None:
            data["boundary"] = Boundary(**data["boundary"])
        return cls(**data)


@dataclass(frozen=True)
class TapeEvent:
    key: str
    name: str
    index: int
    fingerprint: str
    seed: int
    output: Any
    error: str | None = None


@dataclass
class Tape:
    """The recorded boundary calls of one run, addressable by boundary key."""

    events: list[TapeEvent] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._by_key = {event.key: event for event in self.events}
        self._by_content: dict[tuple[str, str], list[TapeEvent]] = {}
        for event in self.events:
            self._by_content.setdefault((event.name, event.fingerprint), []).append(event)

    def get(self, key: str) -> TapeEvent | None:
        return self._by_key.get(key)

    def find(self, name: str, fingerprint: str, nth: int = 0) -> TapeEvent | None:
        """The ``nth`` recorded call of ``name`` with exactly these inputs."""
        matches = self._by_content.get((name, fingerprint), [])
        return matches[nth] if nth < len(matches) else None

    def keys(self) -> list[str]:
        return [event.key for event in self.events]

    def seeds(self) -> dict[str, int]:
        return {event.key: event.seed for event in self.events}

    def __len__(self) -> int:
        return len(self.events)

    def to_json(self) -> str:
        return json.dumps(
            [{**asdict(e), "output": to_jsonable(e.output)} for e in self.events], indent=1
        )

    @classmethod
    def from_json(cls, text: str) -> Tape:
        return cls([TapeEvent(**item) for item in json.loads(text)])

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json())

    @classmethod
    def load(cls, path: str | Path) -> Tape:
        return cls.from_json(Path(path).read_text())


@dataclass
class Run:
    run_id: str
    name: str
    seed: int
    started_ns: int
    ended_ns: int | None = None
    status: Literal["running", "ok", "error"] = "running"
    error: str | None = None
    outcome: Any = None
    passed: bool | None = None
    parent_run_id: str | None = None
    spans: list[Span] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def boundaries(self) -> list[Span]:
        spans = [s for s in self.spans if s.boundary is not None]
        return sorted(spans, key=lambda s: s.boundary.index if s.boundary else 0)

    def tape(self) -> Tape:
        events = []
        for span in self.boundaries:
            assert span.boundary is not None
            events.append(
                TapeEvent(
                    key=span.boundary.key,
                    name=span.name,
                    index=span.boundary.index,
                    fingerprint=span.boundary.fingerprint,
                    seed=span.boundary.seed,
                    output=copy.deepcopy(span.output),
                    error=span.error,
                )
            )
        return Tape(events)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "name": self.name,
            "seed": self.seed,
            "started_ns": self.started_ns,
            "ended_ns": self.ended_ns,
            "status": self.status,
            "error": self.error,
            "outcome": to_jsonable(self.outcome),
            "passed": self.passed,
            "parent_run_id": self.parent_run_id,
            "meta": to_jsonable(self.meta),
            "spans": [s.to_dict() for s in self.spans],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Run:
        data = dict(data)
        data["spans"] = [Span.from_dict(s) for s in data.get("spans", [])]
        return cls(**data)
