"""Failure attribution by intervention.

``blame`` answers "which step caused this failure?" by *testing* each candidate:
it re-runs the agent with one step repaired (``do(step := repair)``), every other
step's noise held fixed, and measures how often the outcome flips to success. The
culprit is the earliest step whose repair alone is sufficient, which separates a
root cause from the downstream steps that merely show its symptoms.

``bisect`` needs a passing run of the same task: it splices the failing run's
noise into the passing run's for a prefix of steps and binary-searches for the
first step that turns success into failure, using O(log n) replays.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, TypeVar

from counterfact.model import Run, Tape, TapeEvent
from counterfact.runtime import Intervention, Plan, Resample, execute

R = TypeVar("R")
Repair = Callable[[str, int], Intervention]


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if trials == 0:
        return 0.0, 1.0
    p = successes / trials
    denom = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def resample_repair(key: str, sample: int) -> Intervention:
    return Resample(salt=sample + 1)


@dataclass(frozen=True)
class StepEffect:
    key: str
    name: str
    index: int
    successes: int
    trials: int
    low: float
    high: float
    sufficient: bool

    @property
    def success_rate(self) -> float:
        return self.successes / self.trials if self.trials else 0.0


@dataclass
class BlameReport:
    run_id: str
    culprit: StepEffect | None
    effects: list[StepEffect]
    replays: int
    strategy: str
    baseline_passed: bool | None = None

    def ranking(self) -> list[StepEffect]:
        return sorted(self.effects, key=lambda e: (-e.success_rate, e.index))

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "culprit": asdict(self.culprit) if self.culprit else None,
            "effects": [{**asdict(e), "success_rate": e.success_rate} for e in self.effects],
            "replays": self.replays,
            "strategy": self.strategy,
        }


def suspicion(run: Run) -> list[tuple[str, float]]:
    """Cheap prior over steps: errors first, then empty outputs, then later steps."""
    scores = []
    total = max(len(run.boundaries), 1)
    for span in run.boundaries:
        assert span.boundary is not None
        score = 0.0
        if span.status == "error":
            score += 3.0
        if span.output in (None, "", [], {}):
            score += 1.0
        score += 0.1 * span.boundary.index / total
        scores.append((span.boundary.key, score))
    return sorted(scores, key=lambda kv: -kv[1])


def blame(
    agent: Callable[[], R],
    failing: Run,
    check: Callable[[R], bool],
    *,
    samples: int = 3,
    threshold: float = 0.5,
    strategy: Literal["earliest", "all", "ranked"] = "earliest",
    budget: int | None = None,
    candidates: Sequence[str] | None = None,
    repair: Repair = resample_repair,
    reuse_noise: bool = True,
    match: Literal["content", "position"] = "content",
) -> BlameReport:
    """Find the earliest step whose repair alone makes the run pass.

    Args:
        agent: zero-argument callable that runs the agent (same one that was recorded).
        failing: the recorded failing run.
        check: outcome predicate; ``True`` means success.
        samples: interventions per step (more samples tighten the interval).
        threshold: success rate at which a repair counts as sufficient.
        strategy: ``"earliest"`` stops at the first sufficient step (fewest replays);
            ``"all"`` measures every step; ``"ranked"`` tries steps in
            :func:`suspicion` order within ``budget``.
        candidates: restrict to these boundary keys.
        repair: which intervention to apply (default: resample the step's noise).
        reuse_noise: hold every other step's noise fixed (abduction). ``False`` is an
            ablation in which re-executed steps draw fresh noise.
        match: ``"content"`` lets a re-executed call reuse the recording of the same
            call (same name and inputs) even if it moved position; ``"position"``
            (ablation) matches by occurrence only.
    """
    tape = failing.tape()
    order = [event.key for event in tape.events]
    if candidates is not None:
        allowed = set(candidates)
        order = [key for key in order if key in allowed]
    if strategy == "ranked":
        ranked = [key for key, _ in suspicion(failing) if key in set(order)]
        order = ranked[:budget] if budget is not None else ranked
    events = {event.key: event for event in tape.events}

    effects: list[StepEffect] = []
    replays = 0
    for key in order:
        successes = 0
        for sample in range(samples):
            run = execute(
                agent,
                name=f"{failing.name}:do({key})",
                plan=Plan.counterfactual(
                    tape, {key: repair(key, sample)}, reuse_noise=reuse_noise, match=match
                ),
                check=check,
                parent_run_id=failing.run_id,
                meta={"intervention": key, "sample": sample},
            )
            replays += 1
            successes += int(bool(run.passed))
        low, high = wilson_interval(successes, samples)
        effect = StepEffect(
            key=key,
            name=events[key].name,
            index=events[key].index,
            successes=successes,
            trials=samples,
            low=low,
            high=high,
            sufficient=successes / samples >= threshold,
        )
        effects.append(effect)
        if strategy == "earliest" and effect.sufficient:
            break

    sufficient = sorted((e for e in effects if e.sufficient), key=lambda e: e.index)
    return BlameReport(
        run_id=failing.run_id,
        culprit=sufficient[0] if sufficient else None,
        effects=effects,
        replays=replays,
        strategy=strategy,
        baseline_passed=failing.passed,
    )


@dataclass
class BisectReport:
    culprit_key: str | None
    culprit_index: int | None
    replays: int
    probes: list[tuple[int, bool]] = field(default_factory=list)
    reason: str = ""


def _spliced_plan(failing: Tape, passing: Tape, cut: int) -> Plan:
    """Failing run's tape and noise for steps ``<= cut``, passing run's after."""
    events: list[TapeEvent] = []
    seeds: dict[str, int] = {}
    for position, event in enumerate(failing.events):
        source = event if position <= cut else passing.get(event.key)
        if source is not None:
            events.append(source)
            seeds[source.key] = source.seed
    for event in passing.events:
        if failing.get(event.key) is None:
            events.append(event)
            seeds[event.key] = event.seed
    return Plan(memo=Tape(events), seeds=seeds)


def bisect(
    agent: Callable[[], R],
    failing: Run,
    passing: Run,
    check: Callable[[R], bool],
) -> BisectReport:
    """Binary-search the first failing-run step that turns a passing run into a failure."""
    fail_tape, pass_tape = failing.tape(), passing.tape()
    n = len(fail_tape)
    probes: list[tuple[int, bool]] = []

    def passes(cut: int) -> bool:
        run = execute(
            agent,
            name=f"{failing.name}:splice({cut})",
            plan=_spliced_plan(fail_tape, pass_tape, cut),
            check=check,
            parent_run_id=failing.run_id,
            meta={"splice": cut},
        )
        probes.append((cut, bool(run.passed)))
        return bool(run.passed)

    if n == 0:
        return BisectReport(None, None, 0, probes, "failing run has no boundaries")
    if not passes(-1):
        return BisectReport(None, None, len(probes), probes, "passing run does not reproduce")
    if passes(n - 1):
        return BisectReport(None, None, len(probes), probes, "failing run does not reproduce")
    lo, hi = -1, n - 1  # passes(lo) is True, passes(hi) is False
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if passes(mid):
            lo = mid
        else:
            hi = mid
    event = fail_tape.events[hi]
    return BisectReport(event.key, event.index, len(probes), probes, "ok")
