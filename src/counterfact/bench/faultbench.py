"""FaultBench: executable agents with known root-cause faults.

Three agent families (a tool-using calculator, a planner/executor/reviewer team,
and a retrieval QA pipeline) are built from boundaries whose behaviour depends only
on their inputs and their noise (``counterfact.rng()``). Like real LLM calls, correct
steps vary harmlessly (a plan lists facts in any order, a rewrite picks one of
several paraphrases), and faults are silent: no step raises an error, and the damage
surfaces later, usually in the final answer.

Ground truth comes from validators that check each step's output against the
*inputs it actually received*. Every task is first checked to be solvable by a
fault-free agent, and benchmark instances are failing runs in which exactly one
step was invalid, so that step is the unambiguous root cause.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import counterfact.runtime as cf
from counterfact.model import Run, SpanKind

Validator = Callable[..., bool]
VALID: dict[str, Validator] = {}


def _valid(name: str) -> Callable[[Validator], Validator]:
    def register(fn: Validator) -> Validator:
        VALID[name] = fn
        return fn

    return register


def erred_keys(run: Run) -> list[str]:
    """Boundary keys whose output is invalid for the inputs they received."""
    bad = []
    for span in run.boundaries:
        assert span.boundary is not None
        check = VALID.get(span.name)
        if check is None or span.status == "error":
            continue
        if not check(span.output, *span.input["args"], **span.input["kwargs"]):
            bad.append(span.boundary.key)
    return bad


def _flip(p: float) -> bool:
    return cf.rng().random() < p


# ------------------------------------------------------------------ calc family
FACTS = ["apples", "pears", "plums", "figs", "limes", "dates", "kiwis", "mangos"]
OPS = ["sum", "max-min", "sum-of-two-largest"]


@cf.boundary(name="calc.plan", kind=SpanKind.CHAT)
def calc_plan(question: dict[str, Any], p: float) -> dict[str, Any]:
    keys = list(question["keys"])
    cf.rng().shuffle(keys)  # benign: any order is a correct plan
    plan = {"keys": keys, "op": question["op"]}
    if _flip(p):
        if cf.rng().random() < 0.5:
            wrong = [f for f in FACTS if f not in keys]
            plan["keys"][cf.rng().randrange(len(keys))] = cf.rng().choice(wrong)
        else:
            plan["op"] = cf.rng().choice([o for o in OPS if o != plan["op"]])
    return plan


@_valid("calc.plan")
def _plan_ok(out: dict[str, Any], question: dict[str, Any], p: float) -> bool:
    return sorted(out["keys"]) == sorted(question["keys"]) and out["op"] == question["op"]


@cf.boundary(name="calc.lookup")
def calc_lookup(kb: dict[str, int], name: str, p: float) -> int:
    value = kb[name]
    return value + cf.rng().choice([-3, -2, -1, 1, 2, 3]) if _flip(p) else value


@_valid("calc.lookup")
def _lookup_ok(out: int, kb: dict[str, int], name: str, p: float) -> bool:
    return out == kb[name]


def _apply(op: str, values: list[int]) -> int:
    if op == "sum":
        return sum(values)
    if op == "max-min":
        return max(values) - min(values)
    return sum(sorted(values)[-2:])


@cf.boundary(name="calc.compute")
def calc_compute(op: str, values: list[int]) -> int:
    return _apply(op, values)


@_valid("calc.compute")
def _compute_ok(out: int, op: str, values: list[int]) -> bool:
    return out == _apply(op, values)


@cf.boundary(name="calc.respond", kind=SpanKind.CHAT)
def calc_respond(value: int, p: float) -> str:
    text = str(value)
    if _flip(p):
        if len(text) >= 2 and text[0] != text[1]:
            text = text[1] + text[0] + text[2:]
        else:
            text = str(value + 1)
    return f"The answer is {text}."


@_valid("calc.respond")
def _respond_ok(out: str, value: int, p: float) -> bool:
    return out == f"The answer is {value}."


def make_calc(task_seed: int, p: float) -> tuple[Callable[[], str], Callable[[str], bool]]:
    rnd = random.Random(task_seed)
    kb = {f: rnd.randint(5, 60) for f in FACTS}
    keys = rnd.sample(FACTS, 3)
    op = rnd.choice(OPS)
    expected = _apply(op, [kb[k] for k in keys])
    question = {"keys": keys, "op": op}

    def agent() -> str:
        with cf.span("calculator", SpanKind.AGENT):
            plan = calc_plan(question, p)
            values = [calc_lookup(kb, k, p) for k in plan["keys"]]
            value = calc_compute(plan["op"], values)
            return calc_respond(value, p)

    return agent, lambda answer: answer == f"The answer is {expected}."


# ------------------------------------------------------------------ team family
TOPICS = ["history", "climate", "economy", "culture", "transport", "health"]


@cf.boundary(name="team.planner", kind=SpanKind.CHAT)
def team_planner(topics: list[str], p: float) -> list[str]:
    plan = list(topics)
    cf.rng().shuffle(plan)  # benign: topics can be researched in any order
    if _flip(p):
        plan.pop(cf.rng().randrange(len(plan)))
    return plan


@_valid("team.planner")
def _planner_ok(out: list[str], topics: list[str], p: float) -> bool:
    return sorted(out) == sorted(topics)


@cf.boundary(name="team.research")
def team_research(facts: dict[str, str], topic: str, p: float) -> str:
    if _flip(p):
        return facts[cf.rng().choice([t for t in facts if t != topic])]
    return facts[topic]


@_valid("team.research")
def _research_ok(out: str, facts: dict[str, str], topic: str, p: float) -> bool:
    return out == facts[topic]


@cf.boundary(name="team.summarize", kind=SpanKind.CHAT)
def team_summarize(topic: str, fact: str, p: float) -> str:
    return f"{topic}: (details omitted)" if _flip(p) else f"{topic}: {fact}"


@_valid("team.summarize")
def _summarize_ok(out: str, topic: str, fact: str, p: float) -> bool:
    return out == f"{topic}: {fact}"


@cf.boundary(name="team.review", kind=SpanKind.CHAT)
def team_review(summaries: list[str], p: float) -> str:
    kept = list(summaries)
    if _flip(p) and len(kept) > 1:
        kept.pop(cf.rng().randrange(len(kept)))
    return " | ".join(kept)


@_valid("team.review")
def _review_ok(out: str, summaries: list[str], p: float) -> bool:
    return out == " | ".join(summaries)


def make_team(task_seed: int, p: float) -> tuple[Callable[[], str], Callable[[str], bool]]:
    rnd = random.Random(task_seed)
    topics = rnd.sample(TOPICS, 4)
    facts = {t: f"fact-{rnd.randint(100, 999)}" for t in TOPICS}
    expected = sorted(f"{t}: {facts[t]}" for t in topics)

    def agent() -> str:
        with cf.span("planner", SpanKind.AGENT):
            plan = team_planner(topics, p)
        summaries = []
        for topic in plan:
            with cf.span(f"executor:{topic}", SpanKind.AGENT):
                fact = team_research(facts, topic, p)
                summaries.append(team_summarize(topic, fact, p))
        with cf.span("reviewer", SpanKind.AGENT):
            return team_review(summaries, p)

    return agent, lambda report: sorted(report.split(" | ")) == expected


# ------------------------------------------------------------------- rag family
TERMS = [
    "orbit",
    "enzyme",
    "tariff",
    "glacier",
    "sonnet",
    "voltage",
    "harbor",
    "pollen",
    "lattice",
    "falcon",
    "census",
    "canyon",
]
TEMPLATES = ["what do the notes say about {t}", "notes on {t}", "find details about {t}"]


def _content(text: str) -> set[str]:
    return set(text.split()) & set(TERMS)


@cf.boundary(name="rag.rewrite", kind=SpanKind.CHAT)
def rag_rewrite(question: str, p: float) -> str:
    term = sorted(_content(question))[0]
    query = cf.rng().choice(TEMPLATES).format(t=term)  # benign paraphrase
    if _flip(p):
        query = query.replace(term, "it")
    return query


@_valid("rag.rewrite")
def _rewrite_ok(out: str, question: str, p: float) -> bool:
    return _content(out) == _content(question)


def _retrieve(corpus: dict[str, str], query: str) -> list[str]:
    words = _content(query)
    return sorted(corpus, key=lambda d: (-len(words & _content(corpus[d])), d))[:4]


@cf.boundary(name="rag.retrieve")
def rag_retrieve(corpus: dict[str, str], query: str) -> list[str]:
    return _retrieve(corpus, query)


@_valid("rag.retrieve")
def _retrieve_ok(out: list[str], corpus: dict[str, str], query: str) -> bool:
    return out == _retrieve(corpus, query)


def _rerank(corpus: dict[str, str], query: str, ids: list[str]) -> list[str]:
    words = _content(query)
    return sorted(ids, key=lambda d: (-len(words & _content(corpus[d])), d))


@cf.boundary(name="rag.rerank", kind=SpanKind.CHAT)
def rag_rerank(corpus: dict[str, str], query: str, ids: list[str], p: float) -> list[str]:
    ranked = _rerank(corpus, query, ids)
    if _flip(p):
        ranked = ranked[1:] + ranked[:1]
    return ranked


@_valid("rag.rerank")
def _rerank_ok(
    out: list[str], corpus: dict[str, str], query: str, ids: list[str], p: float
) -> bool:
    return out == _rerank(corpus, query, ids)


@cf.boundary(name="rag.answer", kind=SpanKind.CHAT)
def rag_answer(answers: dict[str, str], top: list[str], p: float) -> str:
    if _flip(p) and len(top) > 1:
        return answers[top[1]]
    return answers[top[0]]


@_valid("rag.answer")
def _answer_ok(out: str, answers: dict[str, str], top: list[str], p: float) -> bool:
    return out == answers[top[0]]


def make_rag(task_seed: int, p: float) -> tuple[Callable[[], str], Callable[[str], bool]]:
    rnd = random.Random(task_seed)
    corpus = {}
    answers = {}
    for i, term in enumerate(TERMS):
        filler = " ".join(rnd.sample(["the", "a", "of", "and", "in", "notes", "record"], 4))
        corpus[f"doc{i}"] = f"{term} {filler}"
        answers[f"doc{i}"] = f"answer-{rnd.randint(1000, 9999)}"
    gold = rnd.randrange(len(TERMS))
    question = f"what is said about {TERMS[gold]} in the notes"

    def agent() -> str:
        with cf.span("rag", SpanKind.WORKFLOW):
            query = rag_rewrite(question, p)
            ids = rag_retrieve(corpus, query)
            top = rag_rerank(corpus, query, ids, p)
            return rag_answer(answers, top, p)

    return agent, lambda answer: answer == answers[f"doc{gold}"]


FAMILIES: dict[str, Callable[[int, float], tuple[Callable[[], str], Callable[[str], bool]]]] = {
    "calc": make_calc,
    "team": make_team,
    "rag": make_rag,
}


@dataclass(frozen=True)
class Instance:
    family: str
    task_seed: int
    noise_seed: int
    pass_seed: int | None
    culprit: str
    culprit_index: int
    steps: int
    fault_p: float

    def build(self) -> tuple[Callable[[], str], Callable[[str], bool]]:
        return FAMILIES[self.family](self.task_seed, self.fault_p)

    def failing_run(self) -> Run:
        agent, check = self.build()
        return cf.execute(agent, seed=self.noise_seed, check=check, name=self.label)

    def passing_run(self) -> Run | None:
        if self.pass_seed is None:
            return None
        agent, check = self.build()
        return cf.execute(agent, seed=self.pass_seed, check=check, name=self.label)

    @property
    def label(self) -> str:
        return f"{self.family}/{self.task_seed}/{self.noise_seed}"


def generate(
    per_family: int = 60,
    *,
    families: tuple[str, ...] = ("calc", "team", "rag"),
    fault_p: float = 0.12,
    seed: int = 0,
    max_noise_seeds: int = 400,
) -> list[Instance]:
    """Single-fault failing instances, one per task, with a passing run when one exists."""
    instances: list[Instance] = []
    for family in families:
        task_seed = seed * 100_000
        found = 0
        while found < per_family:
            task_seed += 1
            ideal_agent, ideal_check = FAMILIES[family](task_seed, 0.0)
            if not cf.execute(ideal_agent, check=ideal_check).passed:
                continue  # only tasks that a fault-free agent solves
            agent, check = FAMILIES[family](task_seed, fault_p)
            candidates: list[Run] = []
            pass_seed: int | None = None
            for noise in range(max_noise_seeds):
                run = cf.execute(agent, seed=task_seed * 10_000 + noise, check=check)
                if run.passed and pass_seed is None:
                    pass_seed = run.seed
                elif not run.passed and len(erred_keys(run)) == 1:
                    candidates.append(run)
                if len(candidates) >= 12 and pass_seed is not None:
                    break
            if not candidates:
                continue
            failing = random.Random(task_seed).choice(candidates)
            culprit = erred_keys(failing)[0]
            index = next(
                s.boundary.index
                for s in failing.boundaries
                if s.boundary and s.boundary.key == culprit
            )
            instances.append(
                Instance(
                    family,
                    task_seed,
                    failing.seed,
                    pass_seed,
                    culprit,
                    index,
                    len(failing.boundaries),
                    fault_p,
                )
            )
            found += 1
    return instances
