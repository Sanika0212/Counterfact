"""Score failure-attribution methods on FaultBench instances."""

from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from counterfact.attribution import bisect, blame, suspicion, wilson_interval
from counterfact.bench.faultbench import Instance
from counterfact.judge import Judge, JudgeError
from counterfact.model import Run

# Ablations of blame: how calls are matched on replay x whether recorded noise is reused.
ABLATIONS: dict[str, tuple[Literal["content", "position"], bool]] = {
    "blame": ("content", True),
    "blame_position": ("position", True),
    "blame_fresh_noise": ("content", False),
    "blame_position_fresh_noise": ("position", False),
}
METHODS = (
    "last",
    "first_error",
    "random",
    "suspicion",
    "llm_judge",
    "blame",
    "bisect",
    "blame_position",
    "blame_fresh_noise",
    "blame_position_fresh_noise",
)
LABELS = {
    "last": "Last step (symptom)",
    "first_error": "First error-flagged step",
    "random": "Random step",
    "suspicion": "Heuristic suspicion",
    "llm_judge": "LLM judge reads the trace",
    "blame": "Counterfact blame (intervention)",
    "bisect": "Counterfact bisect (needs a passing run)",
    "blame_position": "Blame, positional matching",
    "blame_fresh_noise": "Blame, fresh noise",
    "blame_position_fresh_noise": "Blame, positional matching + fresh noise",
}


@dataclass(frozen=True)
class Prediction:
    method: str
    instance: str
    family: str
    truth: str
    predicted: str | None
    correct: bool
    replays: int
    seconds: float


def _key(span_list: list[Any], position: int) -> str | None:
    if not span_list:
        return None
    span = span_list[position]
    return span.boundary.key if span.boundary else None


def predict(
    method: str, instance: Instance, *, samples: int = 2, judge: Judge | None = None
) -> Prediction:
    agent, check = instance.build()
    failing: Run = instance.failing_run()
    spans = failing.boundaries
    started = time.perf_counter()
    replays = 0
    predicted: str | None
    if method == "last":
        predicted = _key(spans, -1)
    elif method == "first_error":
        errored = [s for s in spans if s.status == "error"]
        predicted = errored[0].boundary.key if errored and errored[0].boundary else _key(spans, -1)
    elif method == "random":
        predicted = _key(spans, random.Random(instance.noise_seed).randrange(len(spans)))
    elif method == "suspicion":
        predicted = suspicion(failing)[0][0]
    elif method == "llm_judge":
        if judge is None:
            raise ValueError("llm_judge needs an API key (see README: Configuring the LLM judge)")
        predicted = judge.blame(failing)
    elif method in ABLATIONS:
        match, reuse_noise = ABLATIONS[method]
        report = blame(agent, failing, check, samples=samples, match=match, reuse_noise=reuse_noise)
        predicted = report.culprit.key if report.culprit else None
        replays = report.replays
    elif method == "bisect":
        passing = instance.passing_run()
        if passing is None:
            predicted = None
        else:
            split = bisect(agent, failing, passing, check)
            predicted, replays = split.culprit_key, split.replays
    else:
        raise ValueError(f"unknown method {method!r}")
    return Prediction(
        method=method,
        instance=instance.label,
        family=instance.family,
        truth=instance.culprit,
        predicted=predicted,
        correct=predicted == instance.culprit,
        replays=replays,
        seconds=time.perf_counter() - started,
    )


def evaluate(
    instances: Sequence[Instance],
    methods: Sequence[str] = ("last", "first_error", "random", "suspicion", "blame", "bisect"),
    *,
    samples: int = 2,
    judge: Judge | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[Prediction]:
    predictions: list[Prediction] = []
    for method in methods:
        try:
            batch = [predict(method, i, samples=samples, judge=judge) for i in instances]
        except JudgeError as exc:
            # A refused API call (no credit, bad key) skips the judge, not the benchmark.
            if progress is not None:
                progress(f"{method}: skipped, {exc}")
            continue
        predictions.extend(batch)
        if progress is not None:
            done = [p for p in predictions if p.method == method]
            acc = sum(p.correct for p in done) / max(len(done), 1)
            progress(f"{method}: accuracy {acc:.3f} on {len(done)} instances")
    return predictions


def summarize(predictions: Sequence[Prediction]) -> tuple[list[dict[str, Any]], str]:
    by_method: dict[str, list[Prediction]] = defaultdict(list)
    for p in predictions:
        by_method[p.method].append(p)
    families = sorted({p.family for p in predictions})
    rows = []
    for method in [m for m in METHODS if m in by_method]:
        preds = by_method[method]
        hits = sum(p.correct for p in preds)
        low, high = wilson_interval(hits, len(preds))
        row: dict[str, Any] = {
            "method": method,
            "label": LABELS[method],
            "n": len(preds),
            "accuracy": hits / len(preds),
            "low": low,
            "high": high,
            "mean_replays": sum(p.replays for p in preds) / len(preds),
            "mean_seconds": sum(p.seconds for p in preds) / len(preds),
        }
        for family in families:
            fam = [p for p in preds if p.family == family]
            row[family] = sum(p.correct for p in fam) / len(fam) if fam else float("nan")
        rows.append(row)
    header = "| Method | Accuracy (95% CI) | " + " | ".join(families) + " | Replays per run |"
    lines = [header, "|" + "---|" * (len(families) + 3)]
    for row in rows:
        fam_cells = " | ".join(f"{row[f]:.0%}" for f in families)
        lines.append(
            f"| {row['label']} | {row['accuracy']:.0%} [{row['low']:.0%}, {row['high']:.0%}] | "
            f"{fam_cells} | {row['mean_replays']:.1f} |"
        )
    return rows, "\n".join(lines) + "\n"


def run_benchmark(
    instances: Sequence[Instance],
    out_dir: str | Path,
    methods: Sequence[str],
    *,
    samples: int = 2,
    judge: Judge | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate ``methods`` and write results.json / results.md to ``out_dir``.

    If ``out_dir`` already holds results for the same instances, predictions of the
    methods not re-run here are kept, so one method (say the LLM judge) can be added
    to an existing run without recomputing the others.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    predictions = evaluate(instances, methods, samples=samples, judge=judge, progress=progress)
    records = [asdict(i) for i in instances]
    judge_info: dict[str, Any] | None = None
    if judge is not None:
        judge_info = {
            "provider": judge.provider,
            "model": judge.model,
            "calls": judge.calls,
            "input_tokens": judge.input_tokens,
            "output_tokens": judge.output_tokens,
        }
    previous = out / "results.json"
    if previous.exists():
        old = json.loads(previous.read_text())
        if old.get("instances") == json.loads(json.dumps(records)):
            kept = [Prediction(**p) for p in old["predictions"] if p["method"] not in methods]
            predictions = kept + predictions
            if judge_info is None and any(p.method == "llm_judge" for p in kept):
                judge_info = old.get("judge")
    rows, table = summarize(predictions)
    payload = {
        "instances": records,
        "predictions": [asdict(p) for p in predictions],
        "summary": rows,
        "judge": judge_info,
    }
    (out / "results.json").write_text(json.dumps(payload, indent=1) + "\n")
    (out / "results.md").write_text(table)
    return rows
