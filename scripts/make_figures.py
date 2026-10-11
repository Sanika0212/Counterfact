"""Build every README figure.

    uv run --extra viz python scripts/make_figures.py

Inputs: ``runs/faultbench/results.json`` (from ``counterfact bench``) for the two
charts; the banner, the blame animation and the report screenshots are computed
here from the built-in demo agent. Outputs go to ``docs/assets/`` as
``<name>-light.*`` and ``<name>-dark.*``.

Colours come from a fixed palette checked for colour-vision deficiencies:
Counterfact's methods always use the first categorical slot, baselines a neutral.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch, Rectangle

from counterfact.attribution import BlameReport, blame
from counterfact.bench.faultbench import Instance, generate
from counterfact.model import Run
from counterfact.report import write_report

ASSETS = Path("docs/assets")
CSS_PX = 96.0
DPI = 200
CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    ink: str
    ink2: str
    muted: str
    grid: str
    axis: str
    empty: str
    neutral: str
    series: tuple[str, ...]


THEMES = {
    "light": Theme(
        name="light",
        surface="#fcfcfb",
        ink="#0b0b0b",
        ink2="#52514e",
        muted="#898781",
        grid="#e1e0d9",
        axis="#c3c2b7",
        empty="#ecebe6",
        neutral="#a8a79f",
        series=("#2a78d6", "#eb6834", "#1baf7a", "#eda100"),
    ),
    "dark": Theme(
        name="dark",
        surface="#1a1a19",
        ink="#ffffff",
        ink2="#c3c2b7",
        muted="#898781",
        grid="#2c2c2a",
        axis="#383835",
        empty="#262624",
        neutral="#64635e",
        series=("#3987e5", "#d95926", "#199e70", "#c98500"),
    ),
}


def use_theme(mode: str) -> Theme:
    theme = THEMES[mode]
    plt.rcParams.update(
        {
            "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9.5,
            "figure.facecolor": theme.surface,
            "axes.facecolor": theme.surface,
            "savefig.facecolor": theme.surface,
            "text.color": theme.ink,
            "axes.labelcolor": theme.ink2,
            "axes.edgecolor": theme.axis,
            "xtick.color": theme.muted,
            "ytick.color": theme.muted,
            "xtick.labelcolor": theme.ink2,
            "ytick.labelcolor": theme.ink2,
            "grid.color": theme.grid,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "legend.frameon": False,
            "legend.fontsize": 9,
        }
    )
    return theme


def _title(fig: Figure, title: str, subtitle: str, theme: Theme) -> None:
    fig.text(0.012, 0.975, title, ha="left", va="top", fontsize=12.5, weight="bold")
    fig.text(0.012, 0.905, subtitle, ha="left", va="top", fontsize=9, color=theme.ink2)


def _px_per_unit(ax: Axes) -> tuple[float, float]:
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    bbox = ax.get_window_extent()
    return bbox.width / abs(x1 - x0), bbox.height / abs(y1 - y0)


def _rounded_bar(
    ax: Axes, start: float, thickness: float, length: float, color: str, *, horizontal: bool
) -> None:
    """Bar with a 4px rounded data end and a square baseline (axis limits must be final)."""
    if length <= 0:
        return
    ppx, ppy = _px_per_unit(ax)
    radius_px = 4.0 * ax.figure.dpi / CSS_PX
    aspect = ppx / ppy
    if horizontal:
        rounding = min(radius_px / ppx, length / 2, thickness / 2 / aspect)
        xy, width, height = (0.0, start), length, thickness
        cover = Rectangle((0.0, start), max(length - rounding, 0.0), thickness)
    else:
        rounding = min(radius_px / ppx, thickness / 2, length / 2 / aspect)
        xy, width, height = (start, 0.0), thickness, length
        cover = Rectangle((start, 0.0), thickness, max(length - rounding * aspect, 0.0))
    ax.add_patch(
        FancyBboxPatch(
            xy,
            width,
            height,
            boxstyle=f"round,pad=0,rounding_size={rounding}",
            mutation_aspect=aspect,
            facecolor=color,
            edgecolor="none",
            linewidth=0,
        )
    )
    cover.set_facecolor(color)
    cover.set_linewidth(0)
    ax.add_patch(cover)


def _save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(path)
    return path


# ------------------------------------------------------------------------ banner


def banner(mode: str) -> str:
    t = THEMES[mode]
    blue, orange, green = t.series[0], t.series[1], t.series[2]
    font = "system-ui, -apple-system, 'Segoe UI', sans-serif"
    width, height = 1200, 260
    x0, step, w, h = 600, 60, 42, 30
    top, bottom, culprit, n = 64, 166, 3, 8
    shapes = []
    for i in range(n):
        x = x0 + i * step
        color, opacity = (orange, 1.0) if i == culprit else (blue, 1.0 if i < culprit else 0.35)
        shapes.append(
            f'<rect x="{x}" y="{top}" width="{w}" height="{h}" rx="7" '
            f'fill="{color}" fill-opacity="{opacity}"/>'
        )
        if i < n - 1:
            shapes.append(
                f'<line x1="{x + w}" y1="{top + h / 2}" x2="{x + step}" y2="{top + h / 2}" '
                f'stroke="{t.axis}" stroke-width="2"/>'
            )
    for i in range(culprit, n):
        x = x0 + i * step
        color = green if i == culprit else blue
        shapes.append(
            f'<rect x="{x}" y="{bottom}" width="{w}" height="{h}" rx="7" fill="{color}"/>'
        )
        if i < n - 1:
            shapes.append(
                f'<line x1="{x + w}" y1="{bottom + h / 2}" x2="{x + step}" '
                f'y2="{bottom + h / 2}" stroke="{t.axis}" stroke-width="2"/>'
            )
    cx = x0 + culprit * step + w / 2
    shapes.append(
        f'<path d="M {cx} {top + h + 4} C {cx} {top + h + 40}, {cx} {bottom - 40}, '
        f'{cx} {bottom - 4}" fill="none" stroke="{t.ink2}" stroke-width="2" '
        f'stroke-dasharray="5 5"/>'
    )
    end = x0 + n * step - step + w + 14
    shapes.append(
        f'<text x="{end}" y="{top + 21}" font-family="{font}" font-size="17" '
        f'fill="{t.ink2}">✗ fail</text>'
    )
    shapes.append(
        f'<text x="{end}" y="{bottom + 21}" font-family="{font}" font-size="17" '
        f'font-weight="600" fill="{t.ink}">✓ pass</text>'
    )
    shapes.append(
        f'<text x="{x0}" y="{top - 14}" font-family="{font}" font-size="14" '
        f'fill="{t.ink2}">recorded run</text>'
    )
    shapes.append(
        f'<text x="{cx + 14}" y="{(top + bottom + h) / 2 + 5}" font-family="{font}" '
        f'font-size="14" fill="{t.ink2}">what if only this step had gone right?</text>'
    )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Counterfact: find the step that broke your agent">
  <rect width="{width}" height="{height}" rx="18" fill="{t.surface}"/>
  <text x="56" y="118" font-family="{font}" font-size="64" font-weight="700" fill="{t.ink}">Counterfact</text>
  <text x="58" y="162" font-family="{font}" font-size="22" fill="{t.ink2}">Find the step that broke your agent,</text>
  <text x="58" y="192" font-family="{font}" font-size="22" fill="{t.ink2}">by replaying what would have happened</text>
  {"".join(shapes)}
</svg>
"""


# ---------------------------------------------------------------- FaultBench chart

BASELINES = ("random", "last", "first_error", "suspicion", "llm_judge")
OURS = ("blame", "bisect")
SHORT = {
    "random": "Random step",
    "last": "Last step (the symptom)",
    "first_error": "First step that raised",
    "suspicion": "Heuristic suspicion score",
    "llm_judge": "LLM judge (gpt-4.1-mini)",
    "blame": "Counterfact blame",
    "bisect": "Counterfact bisect",
}


def plot_faultbench(summary: list[dict[str, Any]], n: int, out: Path, mode: str) -> Path:
    theme = use_theme(mode)
    rows = {r["method"]: r for r in summary}
    order = [m for m in (*BASELINES, *OURS) if m in rows]
    fig = plt.figure(figsize=(7.6, 0.42 * len(order) + 1.55))
    _title(
        fig,
        "Which step caused the failure?",
        f"{n} failing runs of three agent types in FaultBench. Share of runs where each method "
        "names the true root-cause step;\nwhiskers are 95% Wilson intervals.",
        theme,
    )
    top = 1 - 0.86 / fig.get_figheight()
    ax = fig.add_axes((0.30, 0.1, 0.58, top - 0.1))
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_yticks(range(len(order)), [SHORT[m] for m in order])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.grid(True, axis="x")
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=6)
    ax.spines["bottom"].set_visible(False)
    fig.canvas.draw()
    for y, method in enumerate(order):
        row = rows[method]
        ours = method in OURS
        color = theme.series[0] if ours else theme.neutral
        _rounded_bar(ax, y - 0.3, 0.6, row["accuracy"], color, horizontal=True)
        ax.plot([row["low"], row["high"]], [y, y], color=theme.ink2, lw=1.2, solid_capstyle="butt")
        for edge in (row["low"], row["high"]):
            ax.plot([edge, edge], [y - 0.12, y + 0.12], color=theme.ink2, lw=1.2)
        ax.text(
            1.04,
            y,
            f"{row['accuracy']:.0%}",
            va="center",
            ha="left",
            fontsize=9.5,
            weight="bold" if ours else "normal",
            color=theme.ink if ours else theme.ink2,
            clip_on=False,
        )
    for label in ax.get_yticklabels():
        if any(SHORT[m] == label.get_text() for m in OURS):
            label.set_color(theme.ink)
            label.set_fontweight("bold")
    return _save(fig, out)


def plot_ablation(summary: list[dict[str, Any]], out: Path, mode: str) -> Path:
    theme = use_theme(mode)
    rows = {r["method"]: r for r in summary}
    groups = [
        ("Calls matched by content", "blame", "blame_fresh_noise"),
        ("Calls matched by position", "blame_position", "blame_position_fresh_noise"),
    ]
    series = [("Recorded noise reused", 0), ("Fresh noise", 1)]
    fig = plt.figure(figsize=(7.6, 3.6))
    _title(
        fig,
        "What makes blame accurate",
        "Blame accuracy on the same runs with each ingredient switched off. Reusing the recorded\n"
        "randomness keeps the rest of the run fixed; content matching follows calls that moved.",
        theme,
    )
    ax = fig.add_axes((0.09, 0.12, 0.88, 0.6))
    ax.set_ylim(0, 1.08)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xticks(range(len(groups)), [g[0] for g in groups])
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=6)
    ax.spines["bottom"].set_color(theme.axis)
    fig.canvas.draw()
    width, gap = 0.3, 0.02
    for g, (_, *methods) in enumerate(groups):
        for s, (method, (_, slot)) in enumerate(zip(methods, series, strict=True)):
            if method not in rows:
                continue
            row = rows[method]
            x = g - width - gap / 2 if s == 0 else g + gap / 2
            _rounded_bar(ax, x, width, row["accuracy"], theme.series[slot], horizontal=False)
            ax.text(
                x + width / 2,
                row["accuracy"] + 0.025,
                f"{row['accuracy']:.0%}",
                ha="center",
                va="bottom",
                fontsize=9.5,
                color=theme.ink,
            )
    handles = [
        Rectangle((0, 0), 1, 1, facecolor=theme.series[slot], edgecolor="none")
        for _, slot in series
    ]
    fig.legend(
        handles,
        [name for name, _ in series],
        loc="upper left",
        bbox_to_anchor=(0.008, 0.8),
        ncol=2,
        handlelength=1.0,
        handleheight=1.0,
        columnspacing=1.6,
    )
    return _save(fig, out)


# ---------------------------------------------------------------- blame animation


def demo_case() -> tuple[Instance, Run, BlameReport, Run | None]:
    instance = next(
        i
        for i in generate(per_family=12, fault_p=0.12)
        if i.family == "team" and i.culprit.startswith("team.research")
    )
    agent, check = instance.build()
    failing = instance.failing_run()
    failing.name = "team agent: four-topic report"
    report = blame(agent, failing, check, samples=3, strategy="all")
    return instance, failing, report, instance.passing_run()


def _short(key: str) -> tuple[str, str]:
    name, occurrence = key.split("#")
    return name.split(".")[-1], f"#{occurrence}"


def animate_blame(failing: Run, report: BlameReport, out: Path, mode: str) -> Path:
    theme = use_theme(mode)
    effects = {e.key: e for e in report.effects}
    keys = [s.boundary.key for s in failing.boundaries if s.boundary is not None]
    culprit = report.culprit.key if report.culprit else None
    frames: list[tuple[int, bool]] = []
    for i in range(len(keys)):
        frames += [(i, False), (i, True), (i, True)]
    frames += [(len(keys), True)] * 8  # hold the verdict

    fig = plt.figure(figsize=(9.6, 4.2))
    n = len(keys)

    def draw(frame: int) -> None:
        fig.clear()
        probe, shown = frames[frame]
        done = probe >= n
        fig.text(
            0.012, 0.975, "How blame finds the root cause", ha="left", va="top",
            fontsize=13, weight="bold",
        )  # fmt: skip
        fig.text(
            0.012,
            0.905,
            "The run failed. For each step, repair only that step, keep every other step's "
            "recorded randomness,\nand replay the agent three times. The root cause is the "
            "step whose repair alone makes the run pass.",
            ha="left",
            va="top",
            fontsize=9.3,
            color=theme.ink2,
        )
        ax = fig.add_axes((0.085, 0.2, 0.9, 0.56))
        ax.set_xlim(0, n)
        ax.set_ylim(0, 1)
        ax.axis("off")
        for i, key in enumerate(keys):
            effect = effects.get(key)
            revealed = effect is not None and (i < probe or (i == probe and shown) or done)
            active = i == probe and not done
            is_culprit = key == culprit and (done or (revealed and effect is not None))
            edge = theme.series[1] if is_culprit else (theme.ink if active else theme.axis)
            ax.add_patch(
                FancyBboxPatch(
                    (i + 0.08, 0.66),
                    0.84,
                    0.3,
                    boxstyle="round,pad=0,rounding_size=0.06",
                    facecolor=theme.series[1] if is_culprit else theme.surface,
                    edgecolor=edge,
                    linewidth=2.2 if active or is_culprit else 1.2,
                    mutation_aspect=1 / 3,
                )
            )
            name, occurrence = _short(key)
            text_color = "#ffffff" if is_culprit else theme.ink
            ax.text(i + 0.5, 0.835, name, ha="center", va="center", fontsize=8.6, color=text_color)
            ax.text(i + 0.5, 0.735, occurrence, ha="center", va="center", fontsize=8.6,
                    color=text_color)  # fmt: skip
            ax.add_patch(Rectangle((i + 0.3, 0.08), 0.4, 0.46, facecolor=theme.empty, lw=0))
            if revealed and effect is not None:
                rate = effect.success_rate
                ax.add_patch(
                    Rectangle(
                        (i + 0.3, 0.08),
                        0.4,
                        0.46 * rate,
                        facecolor=theme.series[1] if key == culprit else theme.series[0],
                        lw=0,
                    )
                )
                ax.text(
                    i + 0.5, 0.0, f"{effect.successes}/{effect.trials}", ha="center",
                    va="top", fontsize=8.6, color=theme.ink2,
                )  # fmt: skip
        ax.text(-0.02, 0.31, "replays\nthat pass", ha="right", va="center", fontsize=8,
                color=theme.muted, clip_on=False)  # fmt: skip
        if done and culprit is not None:
            name, occurrence = _short(culprit)
            effect = effects[culprit]
            message = (
                f"Root cause: {name} {occurrence}. Repairing it alone makes "
                f"{effect.successes}/{effect.trials} replays pass; no other single repair does."
            )
        elif not done:
            name, occurrence = _short(keys[probe])
            effect = effects.get(keys[probe])
            if shown and effect is not None:
                message = (
                    f"Repair {name} {occurrence}, replay the rest: "
                    f"{effect.successes} of {effect.trials} replays pass."
                )
            else:
                message = f"Repair {name} {occurrence}, replay the rest ..."
        else:
            message = "No single step explains this failure."
        fig.text(0.02, 0.05, message, ha="left", va="bottom", fontsize=10.5, color=theme.ink)

    anim = FuncAnimation(fig, draw, frames=len(frames))
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out, writer=PillowWriter(fps=3), dpi=130, savefig_kwargs={"facecolor": theme.surface})
    plt.close(fig)
    print(out)
    return out


# ------------------------------------------------------------- report screenshots


def screenshot_reports(failing: Run, report: BlameReport, passing: Run | None) -> None:
    if not CHROME.exists():
        print(f"skipping report screenshots: {CHROME} not found")
        return
    with tempfile.TemporaryDirectory() as tmp:
        page = write_report(failing, Path(tmp) / "report.html", blame=report, compare=passing)
        html_text = page.read_text()
        for mode in ("light", "dark"):
            themed = Path(tmp) / f"report-{mode}.html"
            themed.write_text(html_text.replace("<html", f'<html data-theme="{mode}"', 1))
            target = ASSETS / f"report-{mode}.png"
            subprocess.run(
                [
                    str(CHROME),
                    "--headless=new",
                    "--hide-scrollbars",
                    "--force-device-scale-factor=2",
                    "--window-size=1100,860",
                    f"--screenshot={target.resolve()}",
                    themed.as_uri(),
                ],
                check=True,
                capture_output=True,
            )
            print(target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", default="runs/faultbench/results.json")
    args = parser.parse_args()
    ASSETS.mkdir(parents=True, exist_ok=True)
    for mode in THEMES:
        (ASSETS / f"banner-{mode}.svg").write_text(banner(mode))
        print(ASSETS / f"banner-{mode}.svg")

    bench = Path(args.bench)
    if bench.exists():
        data = json.loads(bench.read_text())
        n = len(data["instances"])
        for mode in THEMES:
            plot_faultbench(data["summary"], n, ASSETS / f"faultbench-{mode}.png", mode)
            plot_ablation(data["summary"], ASSETS / f"ablation-{mode}.png", mode)
    else:
        print(f"skipping charts: {bench} not found (run `counterfact bench` first)")

    _, failing, report, passing = demo_case()
    for mode in THEMES:
        animate_blame(failing, report, ASSETS / f"blame-{mode}.gif", mode)
    screenshot_reports(failing, report, passing)
    if shutil.which("gifsicle"):
        for gif in ASSETS.glob("*.gif"):
            subprocess.run(["gifsicle", "-O3", "-b", str(gif)], check=False)


if __name__ == "__main__":
    main()
