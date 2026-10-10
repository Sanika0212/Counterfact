import json
from pathlib import Path

import httpx
import pytest

import counterfact.runtime as cf
from counterfact.attribution import blame
from counterfact.cli import main
from counterfact.model import SpanKind
from counterfact.otel import from_otlp, send_otlp, to_otlp, write_otlp
from counterfact.report import render, write_report
from counterfact.store import Store
from counterfact.testing import (
    ContractError,
    assert_max_calls,
    assert_no_secrets,
    assert_order,
    assert_replays,
    assert_tools_within,
)
from tests import cli_agent


def failing_run(max_seed: int = 500):
    for seed in range(max_seed):
        run = cf.execute(cli_agent.agent, seed=seed, check=cli_agent.check, name="cli-agent")
        bad = [s for s in run.boundaries if s.output != 10 * s.input["args"][0]]
        if run.passed is False and len(bad) == 1:
            return run, bad[0].boundary.key  # type: ignore[union-attr]
    raise AssertionError


def passing_run():
    return next(
        r
        for r in (cf.execute(cli_agent.agent, seed=s, check=cli_agent.check) for s in range(500))
        if r.passed
    )


def test_report_is_self_contained_and_escaped(tmp_path: Path) -> None:
    @cf.boundary(kind=SpanKind.CHAT)
    def say(text: str) -> str:
        return text

    run = cf.execute(lambda: say("</script><b>hi</b>"), name="<demo & co>")
    page = render(run)
    assert page.startswith("<!doctype html>")
    assert "&lt;demo &amp; co&gt;" in page
    assert "<\\/script>" in page
    assert "http://" not in page.split("<script")[0]
    assert "https://" not in page
    failing, culprit = failing_run()
    report = blame(cli_agent.agent, failing, cli_agent.check, samples=2, strategy="all")
    path = write_report(failing, tmp_path / "r.html", blame=report, compare=passing_run())
    html_text = path.read_text()
    assert f"Root cause: <b>{culprit}</b>" in html_text
    assert "Diff against" in html_text
    assert "All spans as a table" in html_text


def test_otlp_export_round_trip(tmp_path: Path) -> None:
    run, _ = failing_run()
    data = to_otlp(run)
    spans = data["resourceSpans"][0]["scopeSpans"][0]["spans"]
    assert len(spans) == len(run.spans)
    assert all(len(s["traceId"]) == 32 and len(s["spanId"]) == 16 for s in spans)
    tool = next(s for s in spans if s["name"] == "execute_tool fetch")
    keys = {a["key"] for a in tool["attributes"]}
    assert {"gen_ai.operation.name", "gen_ai.tool.name", "counterfact.boundary.key"} <= keys
    path = write_otlp(run, tmp_path / "t.json")
    imported = from_otlp(json.loads(path.read_text()))
    assert len(imported) == 1
    assert len(imported[0].spans) == len(run.spans)
    assert {s.kind for s in imported[0].spans} >= {SpanKind.TOOL, SpanKind.AGENT}


def test_otlp_send_posts_json() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200)

    run, _ = failing_run()
    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert send_otlp(run, "http://collector:4318/v1/traces", client=client) == 200
    assert "resourceSpans" in json.loads(seen[0].content)


def test_contract_assertions() -> None:
    run = passing_run()
    assert_tools_within(run, {"fetch"})
    assert_order(run, "fetch", "fetch")
    assert_max_calls(run, 4, kind=None)
    assert_no_secrets(run)
    assert_replays(run, cli_agent.agent)
    with pytest.raises(ContractError, match="outside the contract"):
        assert_tools_within(run, {"other"})
    with pytest.raises(ContractError, match="exceed the budget"):
        assert_max_calls(run, 2, kind=None)
    with pytest.raises(ContractError, match="does not occur"):
        assert_order(run, "missing")

    @cf.boundary
    def leak() -> str:
        return "contact ada@example.org"

    with pytest.raises(ContractError, match="sensitive"):
        assert_no_secrets(cf.execute(lambda: leak()))


def test_pytest_plugin_records_then_replays(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        test_agent="""
import counterfact.runtime as cf

calls = {"n": 0}

@cf.boundary
def model(q: str) -> str:
    calls["n"] += 1
    return q.upper()

def agent():
    return model("hi")

def test_with_tape(cf_tape):
    run = cf_tape(agent, "greeting", check=lambda out: out == "HI")
    assert run.passed

def test_record(cf_record):
    assert cf_record(agent).outcome == "HI"
"""
    )
    first = pytester.runpytest()  # the plugin loads via its pytest11 entry point
    first.assert_outcomes(passed=2)
    assert (pytester.path / "tests" / "tapes" / "greeting.tape.json").exists()
    second = pytester.runpytest("--counterfact-db", "runs.sqlite")
    second.assert_outcomes(passed=2)
    assert Store(pytester.path / "runs.sqlite").list_runs()


def test_cli_end_to_end(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    db = str(tmp_path / "cli.sqlite")
    store = Store(db)
    failing, culprit = failing_run()
    passing = passing_run()
    store.save_run(failing)
    store.save_run(passing)

    assert main(["runs", "--db", db]) == 0
    assert failing.run_id in capsys.readouterr().out
    assert main(["show", failing.run_id, "--db", db]) == 0
    assert "fetch#0" in capsys.readouterr().out
    assert main(["diff", failing.run_id, passing.run_id, "--db", db]) == 0
    assert "first divergence" in capsys.readouterr().out
    out_html = tmp_path / "r.html"
    assert (
        main(
            [
                "report",
                failing.run_id,
                "--db",
                db,
                "--out",
                str(out_html),
                "--compare",
                passing.run_id,
            ]
        )
        == 0
    )
    assert out_html.exists()
    capsys.readouterr()
    assert (
        main(
            [
                "blame",
                "--run",
                failing.run_id,
                "--db",
                db,
                "--agent",
                "tests.cli_agent:agent",
                "--check",
                "tests.cli_agent:check",
                "--samples",
                "2",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["culprit"]["key"] == culprit
    assert (
        main(
            [
                "bisect",
                "--fail",
                failing.run_id,
                "--pass",
                passing.run_id,
                "--db",
                db,
                "--agent",
                "tests.cli_agent:agent",
                "--check",
                "tests.cli_agent:check",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["culprit"] == culprit
    assert main(["export", failing.run_id, "--db", db, "--out", str(tmp_path / "t.json")]) == 0
    assert (tmp_path / "t.json").exists()
    assert main(["version"]) == 0


def test_cli_demo_and_bench(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Never let a test reach a paid API: no keys, and no .env in the working directory.
    monkeypatch.chdir(tmp_path)
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "COUNTERFACT_JUDGE_PROVIDER"):
        monkeypatch.delenv(var, raising=False)
    assert main(["demo", "--out", str(tmp_path / "demo")]) == 0
    out = capsys.readouterr().out
    assert "blame:" in out
    assert (tmp_path / "demo" / "report.html").exists()
    assert (
        main(
            [
                "bench",
                "--per-family",
                "2",
                "--methods",
                "last",
                "blame",
                "llm_judge",
                "--out",
                str(tmp_path / "bench"),
            ]
        )
        == 0
    )
    assert "skipping llm_judge" in capsys.readouterr().out
    assert (tmp_path / "bench" / "results.md").exists()
