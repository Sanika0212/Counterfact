"""Score failure-attribution methods on FaultBench instances."""

from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from counterfact.attribution import bisect, blame, suspicion, wilson_interval
from counterfact.bench.faultbench import Instance
from counterfact.judge import Judge
from counterfact.model import Run

METHODS = ("last", "first_error", "random", "suspicion", "llm_judge", "blame", "bisect")
LABELS = {
    "last": "Last step (symptom)",
    "first_error": "First error-flagged step",
    "random": "Random step",
    "suspicion": "Heuristic suspicion",
    "llm_judge": "LLM judge reads the trace",
    "blame": "Counterfact blame (intervention)",
    "bisect": "Counterfact bisect (needs a passing run)",
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
    elif method == "blame":
        report = blame(agent, failing, check, samples=samples)
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
    predictions = []
    for method in methods:
        for instance in instances:
            predictions.append(predict(method, instance, samples=samples, judge=judge))
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
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    predictions = evaluate(instances, methods, samples=samples, judge=judge, progress=progress)
    rows, table = summarize(predictions)
    payload = {
        "instances": [asdict(i) for i in instances],
        "predictions": [asdict(p) for p in predictions],
        "summary": rows,
        "judge": None
        if judge is None
        else {
            "provider": judge.provider,
            "model": judge.model,
            "calls": judge.calls,
            "input_tokens": judge.input_tokens,
            "output_tokens": judge.output_tokens,
        },
    }
    (out / "results.json").write_text(json.dumps(payload, indent=1) + "\n")
    (out / "results.md").write_text(table)
    return rows
