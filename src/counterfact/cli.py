"""Command line: ``counterfact {demo,runs,show,report,diff,blame,bisect,bench,export,version}``."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from counterfact import __version__

DEFAULT_DB = "counterfact.cf.sqlite"


def _load(ref: str) -> Any:
    """Import ``package.module:attribute``."""
    if ":" not in ref:
        raise SystemExit(f"expected module:attribute, got {ref!r}")
    module, attr = ref.split(":", 1)
    sys.path.insert(0, str(Path.cwd()))
    target: Any = importlib.import_module(module)
    for part in attr.split("."):
        target = getattr(target, part)
    return target


def _store(args: argparse.Namespace) -> Any:
    from counterfact.store import Store

    return Store(args.db)


def cmd_demo(args: argparse.Namespace) -> int:
    from counterfact.attribution import bisect, blame
    from counterfact.bench.faultbench import generate
    from counterfact.report import write_report

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    instance = next(
        i
        for i in generate(per_family=12, fault_p=0.12)
        if i.family == "team" and i.culprit.startswith(args.fault)
    )
    agent, check = instance.build()
    failing = instance.failing_run()
    failing.name = "team agent: four-topic report"
    passing = instance.passing_run()
    report = blame(agent, failing, check, samples=3, strategy="all")
    print(f"failing run: {len(failing.boundaries)} calls, outcome passed={failing.passed}")
    print(f"ground truth culprit: {instance.culprit}")
    print(f"blame: {report.culprit.key if report.culprit else None} ({report.replays} replays)")
    if passing is not None:
        split = bisect(agent, failing, passing, check)
        print(f"bisect: {split.culprit_key} ({split.replays} replays)")
    path = write_report(failing, out / "report.html", blame=report, compare=passing)
    print(f"report: {path}")
    return 0


def cmd_runs(args: argparse.Namespace) -> int:
    for r in _store(args).list_runs(args.limit):
        verdict = {True: "pass", False: "FAIL", None: r.status}[r.passed]
        parent = f" <- {r.parent_run_id}" if r.parent_run_id else ""
        print(f"{r.run_id}  {verdict:5s}  {r.spans:4d} spans  {r.name}{parent}")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    run = _store(args).load_run(args.run)
    children: dict[str | None, list[Any]] = {}
    for span in sorted(run.spans, key=lambda s: s.start_ns):
        children.setdefault(span.parent_id, []).append(span)

    def walk(parent: str | None, depth: int) -> None:
        for span in children.get(parent, []):
            mark = f" [{span.boundary.key}]" if span.boundary else ""
            err = f"  !! {span.error}" if span.error else ""
            print(f"{'  ' * depth}{span.kind.value:16s} {span.name}{mark}{err}")
            walk(span.span_id, depth + 1)

    walk(None, 0)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from counterfact.report import write_report

    store = _store(args)
    run = store.load_run(args.run)
    compare = store.load_run(args.compare) if args.compare else None
    print(write_report(run, args.out, compare=compare))
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    from counterfact.diff import diff_runs

    store = _store(args)
    result = diff_runs(store.load_run(args.left), store.load_run(args.right))
    for row in result.rows:
        left = row.left.key if row.left else "-"
        right = row.right.key if row.right else "-"
        print(f"{row.status:15s} {left:28s} {right}")
    print(f"first divergence: {result.first_divergence}")
    return 0


def cmd_blame(args: argparse.Namespace) -> int:
    from counterfact.attribution import blame
    from counterfact.report import write_report

    store = _store(args)
    failing = store.load_run(args.run)
    report = blame(
        _load(args.agent), failing, _load(args.check), samples=args.samples, strategy=args.strategy
    )
    print(json.dumps(report.to_dict(), indent=1))
    if args.report:
        write_report(failing, args.report, blame=report)
    return 0 if report.culprit else 1


def cmd_bisect(args: argparse.Namespace) -> int:
    from counterfact.attribution import bisect

    store = _store(args)
    split = bisect(
        _load(args.agent),
        store.load_run(args.fail),
        store.load_run(args.pass_run),
        _load(args.check),
    )
    print(
        json.dumps(
            {
                "culprit": split.culprit_key,
                "replays": split.replays,
                "reason": split.reason,
                "probes": split.probes,
            },
            indent=1,
        )
    )
    return 0 if split.culprit_key else 1


def cmd_bench(args: argparse.Namespace) -> int:
    from counterfact.bench.evaluate import run_benchmark
    from counterfact.bench.faultbench import generate
    from counterfact.judge import Judge

    judge = Judge.from_env() if "llm_judge" in args.methods else None
    methods = [m for m in args.methods if m != "llm_judge" or judge is not None]
    if "llm_judge" in args.methods and judge is None:
        print("no API key found; skipping llm_judge (see README: Configuring the LLM judge)")
    instances = generate(per_family=args.per_family, fault_p=args.fault_p, seed=args.seed)
    run_benchmark(instances, args.out, methods, samples=args.samples, judge=judge, progress=print)
    print((Path(args.out) / "results.md").read_text())
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    from counterfact.otel import send_otlp, write_otlp

    run = _store(args).load_run(args.run)
    if args.endpoint:
        print(f"collector answered HTTP {send_otlp(run, args.endpoint)}")
    else:
        print(write_otlp(run, args.out))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="counterfact", description="Find the step that broke your agent."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def with_db(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
        p.add_argument("--db", default=DEFAULT_DB, help=f"SQLite store (default {DEFAULT_DB})")
        return p

    demo = sub.add_parser("demo", help="find the culprit in a built-in failing agent")
    demo.add_argument("--out", default="counterfact-demo")
    demo.add_argument("--fault", default="team.research", help="which step type to fault")
    demo.set_defaults(func=cmd_demo)

    runs = with_db(sub.add_parser("runs", help="list recorded runs"))
    runs.add_argument("--limit", type=int, default=30)
    runs.set_defaults(func=cmd_runs)

    show = with_db(sub.add_parser("show", help="print a run's span tree"))
    show.add_argument("run")
    show.set_defaults(func=cmd_show)

    report = with_db(sub.add_parser("report", help="write a single-file HTML report"))
    report.add_argument("run")
    report.add_argument("--out", default="report.html")
    report.add_argument("--compare", help="another run to diff against")
    report.set_defaults(func=cmd_report)

    diff = with_db(sub.add_parser("diff", help="align two runs and show where they diverge"))
    diff.add_argument("left")
    diff.add_argument("right")
    diff.set_defaults(func=cmd_diff)

    blame_p = with_db(sub.add_parser("blame", help="find the root-cause step of a failing run"))
    blame_p.add_argument("--run", required=True)
    blame_p.add_argument("--agent", required=True, help="module:callable that runs the agent")
    blame_p.add_argument("--check", required=True, help="module:callable outcome -> bool")
    blame_p.add_argument("--samples", type=int, default=3)
    blame_p.add_argument("--strategy", choices=["earliest", "all", "ranked"], default="earliest")
    blame_p.add_argument("--report", help="also write an HTML report here")
    blame_p.set_defaults(func=cmd_blame)

    bisect_p = with_db(
        sub.add_parser("bisect", help="binary-search a failing run against a passing one")
    )
    bisect_p.add_argument("--fail", required=True)
    bisect_p.add_argument("--pass", dest="pass_run", required=True)
    bisect_p.add_argument("--agent", required=True)
    bisect_p.add_argument("--check", required=True)
    bisect_p.set_defaults(func=cmd_bisect)

    bench = sub.add_parser("bench", help="run FaultBench")
    bench.add_argument("--out", default="runs/faultbench")
    bench.add_argument("--per-family", type=int, default=60)
    bench.add_argument("--fault-p", type=float, default=0.12)
    bench.add_argument("--samples", type=int, default=2)
    bench.add_argument("--seed", type=int, default=0)
    bench.add_argument(
        "--methods",
        nargs="+",
        default=["last", "first_error", "random", "suspicion", "llm_judge", "blame", "bisect"],
    )
    bench.set_defaults(func=cmd_bench)

    export = with_db(sub.add_parser("export", help="export a run as OpenTelemetry (OTLP JSON)"))
    export.add_argument("run")
    export.add_argument("--out", default="trace.otlp.json")
    export.add_argument("--endpoint", help="POST to an OTLP/HTTP collector instead")
    export.set_defaults(func=cmd_export)

    version = sub.add_parser("version")
    version.set_defaults(func=lambda _: print(__version__) or 0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
