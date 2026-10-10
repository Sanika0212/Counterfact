"""Assertions for agent contract tests.

Use them in pytest (the ``counterfact`` plugin adds fixtures) to pin down how an
agent behaves, not only what it returns: which tools it may call, how many model
calls it may make, that no secret ever reaches a prompt, and that a recorded run
replays exactly.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from counterfact.jsonable import canonical
from counterfact.model import Run, SpanKind
from counterfact.redact import Redactor
from counterfact.runtime import Plan, execute


class ContractError(AssertionError):
    pass


def calls(run: Run, kind: SpanKind | None = None) -> list[str]:
    return [s.name for s in run.boundaries if kind is None or s.kind is kind]


def assert_tools_within(run: Run, allowed: Iterable[str]) -> None:
    permitted = set(allowed)
    used = set(calls(run, SpanKind.TOOL))
    extra = sorted(used - permitted)
    if extra:
        raise ContractError(f"agent called tools outside the contract: {extra}")


def assert_max_calls(run: Run, limit: int, kind: SpanKind | None = SpanKind.CHAT) -> None:
    count = len(calls(run, kind))
    if count > limit:
        label = kind.value if kind else "boundary"
        raise ContractError(f"{count} {label} calls exceed the budget of {limit}")


def assert_order(run: Run, *names: str) -> None:
    """The named calls appear in this relative order (other calls may interleave)."""
    sequence = calls(run)
    position = -1
    for name in names:
        try:
            position = sequence.index(name, position + 1)
        except ValueError:
            message = f"{name!r} does not occur after the previous step in {sequence}"
            raise ContractError(message) from None


def assert_no_secrets(run: Run, redactor: Redactor | None = None) -> None:
    """No input or output of any call contains a secret or personal data."""
    scrub = redactor or Redactor()
    for span in run.boundaries:
        for label, payload in (("input", span.input), ("output", span.output)):
            if canonical(scrub(payload)) != canonical(payload):
                raise ContractError(f"{span.name} {label} contains sensitive data")


def assert_replays(run: Run, agent: Callable[[], Any]) -> Run:
    """Strict replay of ``run`` reproduces its outcome without any live calls."""
    replay = execute(agent, plan=Plan.replay(run.tape(), strict=True), name=f"{run.name}:replay")
    if replay.status != "ok":
        raise ContractError(f"replay failed: {replay.error}")
    if canonical(replay.outcome) != canonical(run.outcome):
        raise ContractError("replay produced a different outcome")
    live = [s.name for s in replay.boundaries if s.boundary and s.boundary.source != "tape"]
    if live:
        raise ContractError(f"replay made live calls: {live}")
    return replay
