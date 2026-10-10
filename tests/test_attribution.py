import contextlib

import pytest

import counterfact.runtime as cf
from counterfact.attribution import bisect, blame, suspicion, wilson_interval
from counterfact.diff import diff_runs
from counterfact.model import Run

SUBTASKS = [3, 5, 7, 11]
EXPECTED = sum(2 * s for s in SUBTASKS)


@cf.boundary
def plan(task: str, fault_p: float) -> list[int]:
    subtasks = list(SUBTASKS)
    if cf.rng().random() < fault_p:
        subtasks.pop(cf.rng().randrange(len(subtasks)))  # silently drops a subtask
    return subtasks


@cf.boundary
def solve(x: int, fault_p: float) -> int:
    return 2 * x + (1 if cf.rng().random() < fault_p else 0)


@cf.boundary
def aggregate(values: list[int]) -> int:
    return sum(values)


def make_agent(fault_p: float):
    def agent() -> int:
        with cf.span("planner"):
            todo = plan("sum-doubled", fault_p)
        with cf.span("executor"):
            values = [solve(x, fault_p) for x in todo]
        return aggregate(values)

    return agent


def check(answer: int) -> bool:
    return answer == EXPECTED


def erred_keys(run: Run) -> list[str]:
    bad = []
    for span in run.boundaries:
        assert span.boundary is not None
        if span.name == "plan" and span.output != SUBTASKS:
            bad.append(span.boundary.key)
        if span.name == "solve" and span.output != 2 * span.input["args"][0]:
            bad.append(span.boundary.key)
    return bad


def find_run(fault_p: float, want: int, *, first: str | None = None) -> Run:
    for seed in range(5000):
        run = cf.execute(make_agent(fault_p), seed=seed, check=check)
        bad = erred_keys(run)
        if len(bad) == want and (first is None or bad[0].startswith(first)):
            return run
    raise AssertionError("no seed found")


def test_blame_finds_root_cause_not_symptom() -> None:
    failing = find_run(0.2, 1, first="plan")
    assert failing.passed is False
    report = blame(make_agent(0.2), failing, check, samples=4, strategy="all")
    assert report.culprit is not None
    assert report.culprit.key == "plan#0"
    aggregate_effect = next(e for e in report.effects if e.name == "aggregate")
    assert aggregate_effect.successes == 0  # the symptom step is not a cause


@pytest.mark.parametrize("seed_offset", [0, 1, 2])
def test_blame_finds_a_faulty_executor_step(seed_offset: int) -> None:
    runs = [
        r
        for r in (cf.execute(make_agent(0.15), seed=s, check=check) for s in range(3000))
        if len(erred_keys(r)) == 1 and erred_keys(r)[0].startswith("solve")
    ]
    failing = runs[seed_offset]
    report = blame(make_agent(0.15), failing, check, samples=4)
    assert report.culprit is not None
    assert report.culprit.key == erred_keys(failing)[0]


def test_earliest_strategy_uses_fewer_replays_than_all() -> None:
    failing = find_run(0.2, 1, first="solve")
    earliest = blame(make_agent(0.2), failing, check, samples=3, strategy="earliest")
    full = blame(make_agent(0.2), failing, check, samples=3, strategy="all")
    assert earliest.culprit == full.culprit
    assert earliest.replays < full.replays


def test_two_independent_faults_have_no_single_sufficient_step() -> None:
    failing = find_run(0.3, 2, first="solve")
    report = blame(make_agent(0.3), failing, check, samples=3, strategy="all")
    assert report.culprit is None
    assert {e.key for e in report.effects if e.successes == 0} >= set(erred_keys(failing))


def test_bisect_agrees_with_blame() -> None:
    failing = find_run(0.15, 1, first="solve")
    passing = next(
        r
        for r in (cf.execute(make_agent(0.15), seed=s, check=check) for s in range(9000, 9999))
        if r.passed
    )
    split = bisect(make_agent(0.15), failing, passing, check)
    assert split.reason == "ok"
    assert split.culprit_key == erred_keys(failing)[0]
    assert split.replays <= 2 + len(failing.boundaries).bit_length()


def test_bisect_reports_when_runs_do_not_reproduce() -> None:
    failing = find_run(0.15, 1, first="solve")
    report = bisect(make_agent(0.15), failing, failing, check)
    assert report.culprit_key is None
    assert "passing run does not reproduce" in report.reason


def test_wilson_interval_bounds() -> None:
    low, high = wilson_interval(0, 4)
    assert low == 0.0
    assert 0.4 < high < 0.6
    assert wilson_interval(5, 5)[1] == 1.0
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_suspicion_puts_errors_first() -> None:
    @cf.boundary
    def ok() -> int:
        return 1

    @cf.boundary
    def boom() -> int:
        raise RuntimeError("x")

    def agent() -> int:
        ok()
        with contextlib.suppress(RuntimeError):
            boom()
        return ok()

    run = cf.execute(agent)
    assert suspicion(run)[0][0] == "boom#0"


def test_ranked_strategy_respects_budget() -> None:
    failing = find_run(0.2, 1, first="solve")
    report = blame(make_agent(0.2), failing, check, samples=2, strategy="ranked", budget=2)
    assert len(report.effects) <= 2
    assert report.replays <= 4


def test_diff_aligns_runs_with_different_step_counts() -> None:
    passing = next(
        r
        for r in (cf.execute(make_agent(0.2), seed=s, check=check) for s in range(500))
        if r.passed
    )
    dropped = find_run(0.2, 1, first="plan")
    result = diff_runs(passing, dropped)
    statuses = [row.status for row in result.rows]
    assert statuses[0] == "output-changed"  # plan returned a shorter list
    assert "only-left" in statuses  # one solve call is missing on the right
    assert result.first_divergence == 0
    assert sum(result.summary().values()) == len(result.rows)
