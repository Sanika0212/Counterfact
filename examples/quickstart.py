"""Record a flaky agent, find the step that broke it, and write a report.

uv run python examples/quickstart.py
"""

from __future__ import annotations

import counterfact.runtime as cf
from counterfact.attribution import blame
from counterfact.model import SpanKind
from counterfact.report import write_report

PRICES = {"apple": 3, "pear": 5, "plum": 7}


@cf.boundary
def lookup(item: str) -> int:
    """A tool that is usually right. All of its randomness comes from cf.rng()."""
    price = PRICES[item]
    return price + 10 if cf.rng().random() < 0.15 else price


@cf.boundary(kind=SpanKind.CHAT)
def answer(prices: list[int]) -> str:
    """Stands in for the model call that writes the final answer."""
    return f"The basket costs ${sum(prices)}."


def agent() -> str:
    return answer([lookup(item) for item in PRICES])


def check(outcome: str) -> bool:
    return outcome == "The basket costs $15."


if __name__ == "__main__":
    runs = (cf.execute(agent, seed=seed, check=check, name="basket") for seed in range(100))
    failing = next(run for run in runs if not run.passed)
    print(f"recorded run {failing.run_id}: {failing.outcome!r}")

    report = blame(agent, failing, check)
    if report.culprit is not None:
        print(f"root cause: {report.culprit.key} after {report.replays} replays")
    write_report(failing, "quickstart-report.html", blame=report)
    print("wrote quickstart-report.html")
