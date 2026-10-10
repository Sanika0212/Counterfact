"""Align two runs and find where they first diverge.

Boundary calls are aligned with Needleman-Wunsch over their names, so inserted
or skipped steps (for example an extra tool call) do not shift every later
comparison. Each aligned pair is then classified by inputs and outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from counterfact.jsonable import fingerprint
from counterfact.model import Run, TapeEvent

Status = Literal["same", "input-changed", "output-changed", "only-left", "only-right"]


@dataclass(frozen=True)
class DiffRow:
    left: TapeEvent | None
    right: TapeEvent | None
    status: Status


@dataclass(frozen=True)
class RunDiff:
    rows: list[DiffRow]

    @property
    def first_divergence(self) -> int | None:
        return next((i for i, row in enumerate(self.rows) if row.status != "same"), None)

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self.rows:
            counts[row.status] = counts.get(row.status, 0) + 1
        return counts


def _align(left: list[TapeEvent], right: list[TapeEvent]) -> list[tuple[int | None, int | None]]:
    match, mismatch, gap = 2, -1, -1
    n, m = len(left), len(right)
    score = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        score[i][0] = i * gap
    for j in range(1, m + 1):
        score[0][j] = j * gap
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            same = left[i - 1].name == right[j - 1].name
            score[i][j] = max(
                score[i - 1][j - 1] + (match if same else mismatch),
                score[i - 1][j] + gap,
                score[i][j - 1] + gap,
            )
    pairs: list[tuple[int | None, int | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        same = i > 0 and j > 0 and left[i - 1].name == right[j - 1].name
        if i > 0 and j > 0 and score[i][j] == score[i - 1][j - 1] + (match if same else mismatch):
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and score[i][j] == score[i - 1][j] + gap:
            pairs.append((i - 1, None))
            i -= 1
        else:
            pairs.append((None, j - 1))
            j -= 1
    return pairs[::-1]


def diff_runs(left: Run, right: Run) -> RunDiff:
    a, b = left.tape().events, right.tape().events
    rows: list[DiffRow] = []
    for i, j in _align(a, b):
        if i is None:
            rows.append(DiffRow(None, b[j], "only-right"))  # type: ignore[index]
        elif j is None:
            rows.append(DiffRow(a[i], None, "only-left"))
        else:
            x, y = a[i], b[j]
            if x.name != y.name:
                rows.extend([DiffRow(x, None, "only-left"), DiffRow(None, y, "only-right")])
            elif x.fingerprint != y.fingerprint:
                rows.append(DiffRow(x, y, "input-changed"))
            elif fingerprint(x.output) != fingerprint(y.output) or x.error != y.error:
                rows.append(DiffRow(x, y, "output-changed"))
            else:
                rows.append(DiffRow(x, y, "same"))
    return RunDiff(rows)
