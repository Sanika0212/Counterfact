import json
from pathlib import Path

import httpx
import pytest

import counterfact.runtime as cf
from counterfact.bench.evaluate import predict, run_benchmark, summarize
from counterfact.bench.faultbench import FAMILIES, VALID, erred_keys, generate
from counterfact.judge import Judge, load_env, trace_text


@pytest.fixture(scope="module")
def instances():
    return generate(per_family=6, fault_p=0.12)


def test_fault_free_agents_always_solve_their_tasks() -> None:
    for family, make in FAMILIES.items():
        for task in range(1, 15):
            agent, check = make(task, 0.0)
            for seed in range(3):  # benign variation never breaks a correct run
                run = cf.execute(agent, seed=seed, check=check)
                assert run.passed, f"{family} task {task} seed {seed}"
                assert erred_keys(run) == []


def test_every_step_has_a_validator() -> None:
    for make in FAMILIES.values():
        agent, _ = make(1, 0.0)
        run = cf.execute(agent)
        assert {s.name for s in run.boundaries} <= set(VALID)


def test_instances_reproduce_a_single_known_fault(instances) -> None:
    assert {i.family for i in instances} == set(FAMILIES)
    for instance in instances:
        failing = instance.failing_run()
        assert failing.passed is False
        assert erred_keys(failing) == [instance.culprit]
        passing = instance.passing_run()
        assert passing is None or passing.passed


def test_counterfact_beats_baselines(instances) -> None:
    def accuracy(method: str) -> float:
        preds = [predict(method, i, samples=2) for i in instances]
        return sum(p.correct for p in preds) / len(preds)

    assert accuracy("blame") >= 0.8
    assert accuracy("bisect") >= 0.8
    assert accuracy("blame") > accuracy("last")


def test_run_benchmark_writes_results(tmp_path: Path, instances) -> None:
    rows = run_benchmark(instances[:6], tmp_path, ["last", "random", "blame"], samples=1)
    assert {r["method"] for r in rows} == {"last", "random", "blame"}
    saved = json.loads((tmp_path / "results.json").read_text())
    assert len(saved["predictions"]) == 18
    table = (tmp_path / "results.md").read_text()
    assert table.startswith("| Method | Accuracy")


def test_summarize_reports_intervals(instances) -> None:
    preds = [predict("last", i) for i in instances]
    rows, table = summarize(preds)
    assert 0.0 <= rows[0]["low"] <= rows[0]["accuracy"] <= rows[0]["high"] <= 1.0
    assert "Last step" in table


def test_unknown_method_and_missing_judge(instances) -> None:
    with pytest.raises(ValueError, match="unknown"):
        predict("telepathy", instances[0])
    with pytest.raises(ValueError, match="API key"):
        predict("llm_judge", instances[0])


def _judge(provider: str, reply: str, seen: list[httpx.Request]) -> Judge:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if provider == "anthropic":
            return httpx.Response(
                200,
                json={
                    "content": [{"type": "text", "text": reply}],
                    "usage": {"input_tokens": 50, "output_tokens": 5},
                },
            )
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": reply}}],
                "usage": {"prompt_tokens": 40, "completion_tokens": 4},
            },
        )

    return Judge(
        provider=provider,
        model="m",
        api_key="k",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_judge_parses_the_named_step(provider: str, instances) -> None:
    seen: list[httpx.Request] = []
    run = instances[0].failing_run()
    judge = _judge(provider, 'The root cause is {"step": 1}', seen)
    assert judge.blame(run) == run.boundaries[1].boundary.key  # type: ignore[union-attr]
    assert judge.calls == 1
    assert judge.input_tokens > 0
    request = seen[0]
    assert (
        "api.anthropic.com" in str(request.url)
        if provider == "anthropic"
        else "chat/completions" in str(request.url)
    )
    assert "Steps:" in json.loads(request.content)["messages"][0]["content"]


@pytest.mark.parametrize("reply", ["no idea", '{"step": 99}', '{"step": "x"}'])
def test_judge_returns_none_for_unusable_replies(reply: str, instances) -> None:
    assert _judge("openai", reply, []).blame(instances[0].failing_run()) is None


def test_judge_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "COUNTERFACT_JUDGE_PROVIDER",
        "COUNTERFACT_JUDGE_MODEL",
    ):
        monkeypatch.delenv(var, raising=False)
    assert Judge.from_env(tmp_path / "missing.env") is None
    env = tmp_path / ".env"
    env.write_text("# comment\nOPENAI_API_KEY='sk-test-value'\nCOUNTERFACT_JUDGE_MODEL=tiny\n")
    judge = Judge.from_env(env)
    assert judge is not None
    assert (judge.provider, judge.model, judge.api_key) == ("openai", "tiny", "sk-test-value")
    monkeypatch.setenv("OPENAI_API_KEY", "already-set")
    load_env(env)  # existing environment values win
    import os

    assert os.environ["OPENAI_API_KEY"] == "already-set"


def test_trace_text_numbers_steps(instances) -> None:
    text = trace_text(instances[0].failing_run())
    assert text.startswith("[0] ")
    assert "output:" in text


def test_content_matching_follows_reordered_calls() -> None:
    @cf.boundary
    def order(items: list[str]) -> list[str]:
        out = list(items)
        cf.rng().shuffle(out)
        return out

    @cf.boundary
    def fetch(name: str) -> int:
        return len(name) + cf.rng().randint(0, 1000)

    def agent() -> dict[str, int]:
        return {name: fetch(name) for name in order(["a", "bb", "ccc"])}

    first = cf.execute(agent, seed=1)
    reordered = cf.execute(
        agent, plan=cf.Plan.counterfactual(first.tape(), {"order#0": cf.Resample(salt=3)})
    )
    assert reordered.outcome == first.outcome  # same values, found by content
    by_position = cf.execute(
        agent,
        plan=cf.Plan.counterfactual(
            first.tape(), {"order#0": cf.Resample(salt=3)}, match="position"
        ),
    )
    if [s.input for s in by_position.boundaries[1:]] != [s.input for s in first.boundaries[1:]]:
        assert by_position.outcome != first.outcome
