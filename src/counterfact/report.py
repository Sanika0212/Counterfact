"""Single-file HTML report: timeline, inspector, blame chart and run diff.

No server, no build step, no external assets: the report is one HTML file with
inline CSS, SVG and a few lines of JavaScript, so it can be attached to an issue,
opened offline, or committed next to a failing test.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from counterfact.attribution import BlameReport
from counterfact.diff import RunDiff, diff_runs
from counterfact.model import Run, Span, SpanKind

KIND_SLOT = {
    SpanKind.AGENT: "--s1",
    SpanKind.CHAT: "--s2",
    SpanKind.TOOL: "--s3",
    SpanKind.WORKFLOW: "--s4",
    SpanKind.STEP: "--muted",
}

_CSS = """
:root { color-scheme: light;
  --bg:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7;
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100; --good:#0ca30c; --bad:#d03b3b; --hl:rgba(42,120,214,.12); }
@media (prefers-color-scheme: dark) { :root:where(:not([data-theme="light"])) { color-scheme: dark;
  --bg:#0d0d0d; --surface:#1a1a19; --ink:#ffffff; --ink2:#c3c2b7; --muted:#898781; --grid:#2c2c2a; --axis:#383835;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --hl:rgba(57,135,229,.18); } }
:root[data-theme="dark"] { color-scheme: dark; --bg:#0d0d0d; --surface:#1a1a19; --ink:#ffffff; --ink2:#c3c2b7;
  --grid:#2c2c2a; --axis:#383835; --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --hl:rgba(57,135,229,.18); }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }
header { padding:20px 24px 8px; }
h1 { font-size:20px; margin:0; } h2 { font-size:15px; margin:0 0 10px; }
.sub { color:var(--ink2); font-size:13px; }
.chip { display:inline-block; padding:1px 8px; border-radius:10px; font-size:12px; border:1px solid var(--axis); color:var(--ink2); }
.chip.pass { border-color:var(--good); } .chip.fail { border-color:var(--bad); }
main { display:grid; grid-template-columns:minmax(0,1fr) 360px; gap:16px; padding:8px 24px 24px; }
@media (max-width:900px) { main { grid-template-columns:1fr; } }
section { background:var(--surface); border:1px solid var(--grid); border-radius:10px; padding:14px 16px; margin-bottom:16px; }
svg text { fill:var(--ink2); font-size:11px; }
.bar { cursor:pointer; } .bar:hover { opacity:.85; } .bar.sel { stroke:var(--ink); stroke-width:1.5; }
.legend span { margin-right:14px; font-size:12px; color:var(--ink2); }
.legend i { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:5px; vertical-align:-1px; }
pre { background:var(--bg); border:1px solid var(--grid); border-radius:6px; padding:8px; font-size:12px; max-height:260px; overflow:auto; white-space:pre-wrap; word-break:break-word; }
table { width:100%; border-collapse:collapse; font-size:12.5px; } th, td { text-align:left; padding:4px 6px; border-bottom:1px solid var(--grid); }
th { color:var(--muted); font-weight:500; } td.num { text-align:right; font-variant-numeric:tabular-nums; }
tr.culprit td { background:var(--hl); font-weight:600; }
details summary { cursor:pointer; color:var(--ink2); }
"""

_JS = """
const spans = JSON.parse(document.getElementById('cf-spans').textContent);
const panel = document.getElementById('inspector');
function show(id) {
  document.querySelectorAll('.bar').forEach(b => b.classList.toggle('sel', b.dataset.id === id));
  const s = spans[id]; if (!s) return;
  const esc = t => String(t).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
  const b = s.boundary ? `<div class="sub">boundary <b>${esc(s.boundary.key)}</b>, source ${esc(s.boundary.source)}, seed ${s.boundary.seed}</div>` : '';
  panel.innerHTML = `<h2>${esc(s.name)}</h2><div class="sub">${esc(s.kind)}, ${s.ms.toFixed(1)} ms, ${esc(s.status)}</div>${b}` +
    (s.error ? `<pre>${esc(s.error)}</pre>` : '') +
    `<h2 style="margin-top:10px">Input</h2><pre>${esc(JSON.stringify(s.input, null, 2))}</pre>` +
    `<h2>Output</h2><pre>${esc(JSON.stringify(s.output, null, 2))}</pre>` +
    (Object.keys(s.attributes).length ? `<h2>Attributes</h2><pre>${esc(JSON.stringify(s.attributes, null, 2))}</pre>` : '');
}
document.querySelectorAll('.bar').forEach(b => b.addEventListener('click', () => show(b.dataset.id)));
const first = document.querySelector('.bar[data-boundary="1"]') || document.querySelector('.bar');
if (first) show(first.dataset.id);
"""


def _depths(spans: list[Span]) -> dict[str, int]:
    parents = {s.span_id: s.parent_id for s in spans}
    depths: dict[str, int] = {}
    for span in spans:
        depth, node = 0, span.parent_id
        while node is not None and node in parents and depth < 50:
            depth, node = depth + 1, parents[node]
        depths[span.span_id] = depth
    return depths


def _timeline(run: Run) -> str:
    spans = sorted(run.spans, key=lambda s: (s.start_ns, -(s.end_ns or s.start_ns)))
    if not spans:
        return "<p class='sub'>No spans recorded.</p>"
    t0 = min(s.start_ns for s in spans)
    t1 = max((s.end_ns or s.start_ns) for s in spans)
    total = max(t1 - t0, 1)
    depths = _depths(spans)
    row_h, label_w, width = 24, 230, 900
    plot_w = width - label_w - 10
    rows = []
    for i, span in enumerate(spans):
        y = i * row_h
        start = (span.start_ns - t0) / total * plot_w
        dur = max(((span.end_ns or span.start_ns) - span.start_ns) / total * plot_w, 3)
        slot = KIND_SLOT.get(span.kind, "--muted")
        indent = depths[span.span_id] * 12
        label = html.escape(span.name if len(span.name) < 28 else span.name[:26] + "…")
        tip = html.escape(f"{span.kind.value} {span.name} ({span.status})")
        boundary = "1" if span.boundary else "0"
        stroke = ' stroke="var(--bad)" stroke-width="1.5"' if span.status == "error" else ""
        rows.append(
            f'<text x="{8 + indent}" y="{y + 16}">{label}</text>'
            f'<rect class="bar" data-id="{span.span_id}" data-boundary="{boundary}" x="{label_w + start:.1f}" '
            f'y="{y + 5}" width="{dur:.1f}" height="14" rx="3" fill="var({slot})"{stroke}><title>{tip}</title></rect>'
        )
    height = len(spans) * row_h + 6
    grid = "".join(
        f'<line x1="{label_w + plot_w * f:.1f}" x2="{label_w + plot_w * f:.1f}" y1="0" y2="{height}" '
        f'stroke="var(--grid)"/>'
        for f in (0, 0.25, 0.5, 0.75, 1)
    )
    legend = "".join(
        f'<span><i style="background:var({slot})"></i>{kind.value}</span>'
        for kind, slot in KIND_SLOT.items()
    )
    return (
        f'<div class="legend">{legend}</div>'
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="Span timeline">'
        f"{grid}{''.join(rows)}</svg>"
    )


def _blame_section(report: BlameReport) -> str:
    effects = sorted(report.effects, key=lambda e: e.index)
    if not effects:
        return ""
    culprit = report.culprit.key if report.culprit else None
    row_h, label_w, width = 24, 230, 900
    plot_w = width - label_w - 130
    bars = []
    for i, e in enumerate(effects):
        y = i * row_h
        w = e.success_rate * plot_w
        lo, hi = label_w + e.low * plot_w, label_w + e.high * plot_w
        color = "var(--s1)" if e.key == culprit else "var(--axis)"
        bars.append(
            f'<text x="0" y="{y + 17}">{html.escape(e.key)}</text>'
            f'<rect x="{label_w}" y="{y + 6}" width="{max(w, 2):.1f}" height="14" rx="3" fill="{color}">'
            f"<title>{html.escape(e.key)}: {e.successes}/{e.trials} repairs fixed the run</title></rect>"
            f'<line x1="{lo:.1f}" x2="{hi:.1f}" y1="{y + 13}" y2="{y + 13}" stroke="var(--muted)" '
            f'stroke-width="1" opacity="0.7"/>'
            f'<text x="{label_w + plot_w + 8}" y="{y + 17}">{e.success_rate:.0%}'
            f"{' ← culprit' if e.key == culprit else ''}</text>"
        )
    height = len(effects) * row_h + 4
    rows = "".join(
        f"<tr class='{'culprit' if e.key == culprit else ''}'><td>{html.escape(e.key)}</td>"
        f"<td class='num'>{e.successes}/{e.trials}</td><td class='num'>{e.success_rate:.0%}</td>"
        f"<td class='num'>{e.low:.0%} to {e.high:.0%}</td></tr>"
        for e in effects
    )
    verdict = (
        f"Root cause: <b>{html.escape(culprit)}</b>, the earliest step whose repair alone fixes the run."
        if culprit
        else "No single step's repair fixes the run (multiple interacting faults or an unsolvable task)."
    )
    return (
        f"<section><h2>Blame</h2><p class='sub'>{verdict} {report.replays} counterfactual replays.</p>"
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="Repair success by step">'
        f"{''.join(bars)}</svg><details><summary>Table</summary><table><tr><th>Step</th><th>Fixed</th>"
        f"<th>Rate</th><th>95% CI</th></tr>{rows}</table></details></section>"
    )


def _diff_section(diff: RunDiff, right: Run) -> str:
    rows = []
    for row in diff.rows:
        left_key = html.escape(row.left.key) if row.left else "-"
        right_key = html.escape(row.right.key) if row.right else "-"
        rows.append(f"<tr><td>{left_key}</td><td>{right_key}</td><td>{row.status}</td></tr>")
    first = diff.first_divergence
    note = "Runs are identical." if first is None else f"First divergence at row {first}."
    return (
        f"<section><h2>Diff against {html.escape(right.name)}</h2><p class='sub'>{note}</p>"
        f"<table><tr><th>This run</th><th>Other run</th><th>Status</th></tr>{''.join(rows)}</table></section>"
    )


def _span_payload(span: Span) -> dict[str, Any]:
    end = span.end_ns or span.start_ns
    return {
        "name": span.name,
        "kind": span.kind.value,
        "status": span.status,
        "error": span.error,
        "ms": (end - span.start_ns) / 1e6,
        "input": span.input,
        "output": span.output,
        "attributes": span.attributes,
        "boundary": None
        if span.boundary is None
        else {"key": span.boundary.key, "source": span.boundary.source, "seed": span.boundary.seed},
    }


def render(run: Run, *, blame: BlameReport | None = None, compare: Run | None = None) -> str:
    status = "pass" if run.passed else ("fail" if run.passed is False else "")
    verdict = {"pass": "passed", "fail": "failed", "": run.status}[status]
    payload = json.dumps({s.span_id: _span_payload(s) for s in run.spans}).replace("</", "<\\/")
    sections = [f"<section><h2>Timeline</h2>{_timeline(run)}</section>"]
    if blame is not None:
        sections.append(_blame_section(blame))
    if compare is not None:
        sections.append(_diff_section(diff_runs(run, compare), compare))
    spans_table = "".join(
        f"<tr><td>{html.escape(s.name)}</td><td>{s.kind.value}</td><td>{s.status}</td>"
        f"<td>{html.escape(s.boundary.key) if s.boundary else ''}</td></tr>"
        for s in sorted(run.spans, key=lambda s: s.start_ns)
    )
    sections.append(
        "<section><details><summary>All spans as a table</summary><table><tr><th>Name</th><th>Kind</th>"
        f"<th>Status</th><th>Boundary</th></tr>{spans_table}</table></details></section>"
    )
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>Counterfact: {html.escape(run.name)}</title><style>{_CSS}</style></head><body>"
        f"<header><h1>{html.escape(run.name)} <span class='chip {status}'>{verdict}</span></h1>"
        f"<div class='sub'>run {run.run_id}, {len(run.boundaries)} boundary calls, "
        f"{len(run.spans)} spans, seed {run.seed}</div></header>"
        f"<main><div>{''.join(sections)}</div><aside><section id='inspector'>"
        "<p class='sub'>Select a span.</p></section></aside></main>"
        f"<script id='cf-spans' type='application/json'>{payload}</script><script>{_JS}</script>"
        "</body></html>"
    )


def write_report(
    run: Run, path: str | Path, *, blame: BlameReport | None = None, compare: Run | None = None
) -> Path:
    out = Path(path)
    out.write_text(render(run, blame=blame, compare=compare), encoding="utf-8")
    return out
