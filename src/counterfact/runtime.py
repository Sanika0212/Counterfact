"""Recording, replay and intervention for agent runs.

Every nondeterministic call an agent makes (an LLM request, a tool, a random draw)
is wrapped as a **boundary**. A boundary is identified by a stable key,
``"<name>#<occurrence>"``, and receives its own exogenous noise seed. Inside a
boundary, :func:`rng` returns a ``random.Random`` seeded with that noise.

A :class:`Plan` decides how each boundary is resolved. This is Pearl's three-step
counterfactual for agent traces:

- **Abduction:** a recorded run supplies a tape of outputs plus the noise seed of
  every boundary.
- **Action:** :class:`Replace` or :class:`Resample` interventions change chosen
  boundaries.
- **Prediction:** every other boundary reuses its recorded output when its inputs
  are unchanged (a memo hit). When its inputs changed, it re-executes with the
  *same* noise seed, so a fault that was caused downstream is reproduced, not
  accidentally healed.
"""

from __future__ import annotations

import asyncio
import copy
import functools
import hashlib
import random
import threading
import time
import uuid
from collections import Counter
from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Literal, ParamSpec, TypeVar, overload

from counterfact.jsonable import fingerprint, to_jsonable
from counterfact.model import Boundary, Run, Span, SpanKind, Tape

P = ParamSpec("P")
R = TypeVar("R")


class Divergence(RuntimeError):
    """Raised in strict replay when a boundary's inputs differ from the tape."""

    def __init__(self, key: str, expected: str | None, actual: str) -> None:
        super().__init__(f"replay diverged at {key}: recorded {expected}, got {actual}")
        self.key = key
        self.expected = expected
        self.actual = actual


class ReplayedError(RuntimeError):
    """A boundary that failed in the recording fails the same way on replay."""


@dataclass(frozen=True)
class Replace:
    """Intervention: the boundary returns ``value`` instead of running."""

    value: Any


@dataclass(frozen=True)
class Resample:
    """Intervention: the boundary runs again with fresh noise (``salt`` selects the draw)."""

    salt: int = 0


Intervention = Replace | Resample


@dataclass
class Plan:
    """How boundary calls are resolved during one execution."""

    memo: Tape | None = None
    seeds: Mapping[str, int] = field(default_factory=dict)
    base_seed: int = 0
    interventions: Mapping[str, Intervention] = field(default_factory=dict)
    strict: bool = False
    reuse_noise: bool = True  # False = ablation: memo misses draw fresh noise
    match: Literal["content", "position"] = "content"  # how misses find their recording

    @classmethod
    def replay(cls, tape: Tape, *, strict: bool = False) -> Plan:
        return cls(memo=tape, seeds=tape.seeds(), strict=strict)

    @classmethod
    def counterfactual(
        cls,
        tape: Tape,
        interventions: Mapping[str, Intervention],
        *,
        reuse_noise: bool = True,
        match: Literal["content", "position"] = "content",
    ) -> Plan:
        return cls(
            memo=tape,
            seeds=tape.seeds(),
            interventions=dict(interventions),
            reuse_noise=reuse_noise,
            match=match,
        )

    def seed_for(self, key: str) -> int:
        if key in self.seeds:
            return self.seeds[key]
        digest = hashlib.sha256(f"{self.base_seed}:{key}".encode()).digest()
        return int.from_bytes(digest[:8], "big") >> 1


def _salted(seed: int, salt: int) -> int:
    digest = hashlib.sha256(f"{seed}:resample:{salt}".encode()).digest()
    return int.from_bytes(digest[:8], "big") >> 1


@dataclass
class _RunState:
    run: Run
    plan: Plan
    on_span: Callable[[Span], None] | None
    occurrences: Counter[str] = field(default_factory=Counter)
    content_seen: Counter[tuple[str, str]] = field(default_factory=Counter)
    next_index: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def claim(self, name: str) -> tuple[int, str]:
        with self.lock:
            occurrence = self.occurrences[name]
            self.occurrences[name] += 1
            index = self.next_index
            self.next_index += 1
        return index, f"{name}#{occurrence}"

    def content_occurrence(self, name: str, fingerprint: str) -> int:
        with self.lock:
            nth = self.content_seen[(name, fingerprint)]
            self.content_seen[(name, fingerprint)] += 1
        return nth

    def add(self, span: Span) -> None:
        with self.lock:
            self.run.spans.append(span)
        if self.on_span is not None:
            self.on_span(span)


_state: ContextVar[_RunState | None] = ContextVar("counterfact_run", default=None)
_parent: ContextVar[Span | None] = ContextVar("counterfact_span", default=None)
_noise: ContextVar[random.Random | None] = ContextVar("counterfact_noise", default=None)


def current_run() -> Run | None:
    state = _state.get()
    return state.run if state else None


def rng() -> random.Random:
    """The noise source of the boundary currently executing.

    Use it for every random choice inside a boundary so that replays and
    counterfactuals can hold that noise fixed.
    """
    noise = _noise.get()
    if noise is None:
        raise RuntimeError("counterfact.rng() can only be used inside a boundary call")
    return noise


def _new_span(state: _RunState, name: str, kind: SpanKind, attributes: dict[str, Any]) -> Span:
    parent = _parent.get()
    return Span(
        span_id=uuid.uuid4().hex[:16],
        run_id=state.run.run_id,
        parent_id=parent.span_id if parent else None,
        kind=kind,
        name=name,
        start_ns=time.time_ns(),
        attributes=dict(attributes),
    )


@contextmanager
def span(name: str, kind: SpanKind = SpanKind.STEP, **attributes: Any) -> Iterator[Span | None]:
    """A structural span (an agent, a workflow, a step) that nests the calls inside it."""
    state = _state.get()
    if state is None:
        yield None
        return
    node = _new_span(state, name, kind, attributes)
    token = _parent.set(node)
    try:
        yield node
    except BaseException as exc:
        node.status, node.error = "error", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        _parent.reset(token)
        node.end_ns = time.time_ns()
        state.add(node)


class _Call:
    """Resolution of one boundary call: tape hit, intervention, or live execution."""

    def __init__(self, name: str, kind: SpanKind, args: Any, attributes: dict[str, Any]) -> None:
        state = _state.get()
        assert state is not None
        self.state = state
        self.index, self.key = state.claim(name)
        self.fingerprint = fingerprint(args)
        self.seed = state.plan.seed_for(self.key)
        self.span = _new_span(state, name, kind, attributes)
        self.span.input = to_jsonable(args)
        intervention = state.plan.interventions.get(self.key)
        self.replacement: Replace | None = None
        self.recorded = None
        if isinstance(intervention, Replace):
            self.replacement = intervention
        elif isinstance(intervention, Resample):
            self.seed = _salted(self.seed, intervention.salt)
        elif state.plan.memo is not None:
            event = state.plan.memo.get(self.key)
            nth = state.content_occurrence(name, self.fingerprint)
            if event is not None and event.fingerprint == self.fingerprint:
                self.recorded = event
            elif state.plan.match == "content" and (
                moved := state.plan.memo.find(name, self.fingerprint, nth)
            ):
                # Same call, different position (e.g. a reordered plan): reuse it.
                self.recorded = moved
                self.seed = moved.seed
            elif state.plan.strict:
                raise Divergence(self.key, event.fingerprint if event else None, self.fingerprint)
            elif event is not None and not state.plan.reuse_noise:
                self.seed = _salted(self.seed, 7919)

    def source(self) -> str:
        if self.replacement is not None:
            return "intervention"
        return "tape" if self.recorded is not None else "live"

    def cached(self) -> tuple[bool, Any]:
        """``(True, output)`` when the call must not execute."""
        if self.replacement is not None:
            return True, copy.deepcopy(self.replacement.value)
        if self.recorded is not None:
            if self.recorded.error is not None:
                raise ReplayedError(self.recorded.error)
            return True, copy.deepcopy(self.recorded.output)
        return False, None

    def finish(self, output: Any = None, error: BaseException | None = None) -> None:
        span = self.span
        span.end_ns = time.time_ns()
        span.output = to_jsonable(output)
        if error is not None:
            span.status = "error"
            span.error = (
                str(error)
                if isinstance(error, ReplayedError)
                else f"{type(error).__name__}: {error}"
            )
        span.boundary = Boundary(
            index=self.index,
            key=self.key,
            fingerprint=self.fingerprint,
            seed=self.seed,
            source=self.source(),  # type: ignore[arg-type]
        )
        self.state.add(span)


def call(
    name: str,
    fn: Callable[..., R],
    *args: Any,
    kind: SpanKind = SpanKind.TOOL,
    attributes: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> R:
    """Run ``fn(*args, **kwargs)`` as a boundary (outside a recording it just runs)."""
    if _state.get() is None:
        return fn(*args, **kwargs)
    pending = _Call(name, kind, {"args": args, "kwargs": kwargs}, dict(attributes or {}))
    try:
        hit, value = pending.cached()
        if hit:
            pending.finish(value)
            result: R = value
            return result
        noise_token = _noise.set(random.Random(pending.seed))
        parent_token = _parent.set(pending.span)
        try:
            output = fn(*args, **kwargs)
        finally:
            _parent.reset(parent_token)
            _noise.reset(noise_token)
    except BaseException as exc:
        pending.finish(error=exc)
        raise
    pending.finish(output)
    return output


async def acall(
    name: str,
    fn: Callable[..., Awaitable[R]],
    *args: Any,
    kind: SpanKind = SpanKind.TOOL,
    attributes: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> R:
    """Async version of :func:`call`."""
    if _state.get() is None:
        return await fn(*args, **kwargs)
    pending = _Call(name, kind, {"args": args, "kwargs": kwargs}, dict(attributes or {}))
    try:
        hit, value = pending.cached()
        if hit:
            pending.finish(value)
            result: R = value
            return result
        noise_token = _noise.set(random.Random(pending.seed))
        parent_token = _parent.set(pending.span)
        try:
            output = await fn(*args, **kwargs)
        finally:
            _parent.reset(parent_token)
            _noise.reset(noise_token)
    except BaseException as exc:
        pending.finish(error=exc)
        raise
    pending.finish(output)
    return output


@overload
def boundary(fn: Callable[P, R], /) -> Callable[P, R]: ...
@overload
def boundary(
    *, name: str | None = None, kind: SpanKind = SpanKind.TOOL
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def boundary(
    fn: Callable[P, R] | None = None,
    /,
    *,
    name: str | None = None,
    kind: SpanKind = SpanKind.TOOL,
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator form of :func:`call` / :func:`acall` (sync and async functions)."""

    def decorate(func: Callable[P, R]) -> Callable[P, R]:
        label = name or func.__name__
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
                return await acall(label, func, *args, kind=kind, **kwargs)

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            return call(label, func, *args, kind=kind, **kwargs)

        return wrapper

    return decorate(fn) if fn is not None else decorate


def _begin(
    name: str,
    plan: Plan,
    seed: int,
    parent_run_id: str | None,
    meta: Mapping[str, Any],
    on_span: Callable[[Span], None] | None,
) -> _RunState:
    run = Run(
        run_id=uuid.uuid4().hex[:16],
        name=name,
        seed=seed,
        started_ns=time.time_ns(),
        parent_run_id=parent_run_id,
        meta=dict(meta),
    )
    return _RunState(run=run, plan=plan, on_span=on_span)


def execute(
    agent: Callable[[], R],
    *,
    name: str = "run",
    plan: Plan | None = None,
    seed: int = 0,
    check: Callable[[R], bool] | None = None,
    parent_run_id: str | None = None,
    meta: Mapping[str, Any] | None = None,
    on_span: Callable[[Span], None] | None = None,
) -> Run:
    """Run ``agent()`` under ``plan`` and return the recorded :class:`Run`.

    Exceptions raised by the agent are captured on the run (``status="error"``)
    instead of propagating. ``check`` scores the outcome (``run.passed``).
    """
    plan = plan if plan is not None else Plan(base_seed=seed)
    state = _begin(name, plan, seed, parent_run_id, meta or {}, on_span)
    token = _state.set(state)
    run = state.run
    try:
        with span(name, SpanKind.AGENT):
            outcome = agent()
        run.outcome = outcome
        run.status = "ok"
        run.passed = bool(check(outcome)) if check is not None else None
    except Exception as exc:
        run.status = "error"
        run.error = f"{type(exc).__name__}: {exc}"
        run.passed = False if check is not None else None
    finally:
        _state.reset(token)
        run.ended_ns = time.time_ns()
    return run


async def aexecute(
    agent: Callable[[], Awaitable[R]],
    *,
    name: str = "run",
    plan: Plan | None = None,
    seed: int = 0,
    check: Callable[[R], bool] | None = None,
    parent_run_id: str | None = None,
    meta: Mapping[str, Any] | None = None,
    on_span: Callable[[Span], None] | None = None,
) -> Run:
    """Async version of :func:`execute`."""
    plan = plan if plan is not None else Plan(base_seed=seed)
    state = _begin(name, plan, seed, parent_run_id, meta or {}, on_span)
    token = _state.set(state)
    run = state.run
    try:
        with span(name, SpanKind.AGENT):
            outcome = await agent()
        run.outcome = outcome
        run.status = "ok"
        run.passed = bool(check(outcome)) if check is not None else None
    except Exception as exc:
        run.status = "error"
        run.error = f"{type(exc).__name__}: {exc}"
        run.passed = False if check is not None else None
    finally:
        _state.reset(token)
        run.ended_ns = time.time_ns()
    return run
