import asyncio
from dataclasses import dataclass
from pathlib import Path

import pytest

import counterfact.runtime as cf
from counterfact.jsonable import canonical, fingerprint, to_jsonable
from counterfact.model import Run, SpanKind, Tape


class Counter:
    def __init__(self) -> None:
        self.calls = 0


def make_agent(counter: Counter, bias: int = 0):
    @cf.boundary
    def draw(label: str) -> int:
        counter.calls += 1
        return cf.rng().randint(0, 1000)

    @cf.boundary(kind=SpanKind.CHAT)
    def combine(a: int, b: int) -> int:
        counter.calls += 1
        return a + b + bias

    def agent() -> int:
        with cf.span("plan"):
            a = draw("a")
            b = draw("b")
        return combine(a, b)

    return agent


def test_boundaries_get_stable_keys_and_nest_under_spans() -> None:
    run = cf.execute(make_agent(Counter()), name="demo", seed=1)
    assert run.status == "ok"
    assert [s.boundary.key for s in run.boundaries if s.boundary] == [
        "draw#0",
        "draw#1",
        "combine#0",
    ]
    plan_span = next(s for s in run.spans if s.name == "plan")
    draws = [s for s in run.spans if s.name == "draw"]
    assert all(s.parent_id == plan_span.span_id for s in draws)
    agent_span = next(s for s in run.spans if s.kind is SpanKind.AGENT)
    assert plan_span.parent_id == agent_span.span_id


def test_same_seed_same_run_and_different_seed_differs() -> None:
    a = cf.execute(make_agent(Counter()), seed=3)
    b = cf.execute(make_agent(Counter()), seed=3)
    c = cf.execute(make_agent(Counter()), seed=4)
    assert a.outcome == b.outcome
    assert a.outcome != c.outcome


def test_strict_replay_reproduces_without_executing() -> None:
    first = cf.execute(make_agent(Counter()), seed=5)
    counter = Counter()
    replay = cf.execute(make_agent(counter), plan=cf.Plan.replay(first.tape(), strict=True))
    assert replay.outcome == first.outcome
    assert counter.calls == 0
    assert {s.boundary.source for s in replay.boundaries if s.boundary} == {"tape"}


def test_strict_replay_detects_divergence() -> None:
    first = cf.execute(make_agent(Counter()), seed=5)

    @cf.boundary
    def draw(label: str) -> int:
        return 1

    def changed() -> int:
        return draw("different-input")

    run = cf.execute(changed, plan=cf.Plan.replay(first.tape(), strict=True))
    assert run.status == "error"
    assert "diverged at draw#0" in (run.error or "")


def test_memo_miss_reuses_the_recorded_noise() -> None:
    first = cf.execute(make_agent(Counter()), seed=7)
    counter = Counter()
    # bias changes combine's output but not its inputs; change an input instead:
    plan = cf.Plan.counterfactual(first.tape(), {"draw#0": cf.Replace(0)})
    second = cf.execute(make_agent(counter), plan=plan)
    sources = {s.boundary.key: s.boundary.source for s in second.boundaries if s.boundary}
    assert sources == {"draw#0": "intervention", "draw#1": "tape", "combine#0": "live"}
    original_b = first.tape().get("draw#1")
    assert original_b is not None
    assert second.outcome == 0 + original_b.output
    combine_seed = {s.boundary.key: s.boundary.seed for s in second.boundaries if s.boundary}
    assert combine_seed["combine#0"] == first.tape().seeds()["combine#0"]


def test_resample_draws_fresh_noise_only_where_asked() -> None:
    first = cf.execute(make_agent(Counter()), seed=11)
    plan = cf.Plan.counterfactual(first.tape(), {"draw#1": cf.Resample(salt=1)})
    second = cf.execute(make_agent(Counter()), plan=plan)
    old, new = first.tape(), second.tape()
    assert old.get("draw#0") == new.get("draw#0")
    assert old.get("draw#1").output != new.get("draw#1").output  # type: ignore[union-attr]


def test_recorded_errors_replay_as_errors() -> None:
    @cf.boundary
    def flaky() -> int:
        raise ValueError("tool exploded")

    first = cf.execute(lambda: flaky())
    assert first.status == "error"
    assert "tool exploded" in (first.boundaries[0].error or "")
    replay = cf.execute(lambda: flaky(), plan=cf.Plan.replay(first.tape()))
    assert replay.status == "error"
    assert "tool exploded" in (replay.error or "")


def test_check_scores_outcome() -> None:
    run = cf.execute(lambda: 41, check=lambda x: x == 42)
    assert run.passed is False
    assert cf.execute(lambda: 42, check=lambda x: x == 42).passed is True


def test_rng_outside_boundary_raises() -> None:
    with pytest.raises(RuntimeError, match="inside a boundary"):
        cf.rng()


def test_calls_outside_a_run_are_transparent() -> None:
    @cf.boundary
    def double(x: int) -> int:
        return 2 * x

    assert double(4) == 8
    with cf.span("noop") as node:
        assert node is None
    assert cf.current_run() is None


def test_concurrent_async_runs_are_isolated() -> None:
    @cf.boundary
    async def think(i: int, step: int) -> str:
        await asyncio.sleep(0.001 * ((i * 7 + step) % 3))
        return f"agent{i}-step{step}"

    async def agent(i: int) -> list[str]:
        return [await think(i, s) for s in range(3)]

    async def main() -> list[Run]:
        return await asyncio.gather(
            *(cf.aexecute(lambda i=i: agent(i), name=f"agent{i}") for i in range(20))
        )

    runs = asyncio.run(main())
    for i, run in enumerate(runs):
        assert run.outcome == [f"agent{i}-step{s}" for s in range(3)]
        assert [s.boundary.key for s in run.boundaries if s.boundary] == [
            "think#0",
            "think#1",
            "think#2",
        ]
        assert {s.run_id for s in run.spans} == {run.run_id}


def test_tape_and_run_round_trip(tmp_path: Path) -> None:
    run = cf.execute(make_agent(Counter()), seed=2)
    path = tmp_path / "tape.json"
    run.tape().save(path)
    assert Tape.load(path).seeds() == run.tape().seeds()
    restored = Run.from_dict(run.to_dict())
    assert restored.tape().keys() == run.tape().keys()


@dataclass
class Point:
    x: int
    y: tuple[int, ...]


def test_jsonable_and_fingerprint_are_canonical() -> None:
    assert to_jsonable(Point(1, (2, 3))) == {"x": 1, "y": [2, 3]}
    assert to_jsonable({"b": {3, 1, 2}}) == {"b": [1, 2, 3]}
    assert to_jsonable(b"hi") == {"__bytes__": "aGk="}
    assert canonical({"b": 1, "a": 2}) == '{"a":2,"b":1}'
    assert fingerprint({"a": 1, "b": 2}) == fingerprint({"b": 2, "a": 1})
    assert "__repr__" in to_jsonable(object())
