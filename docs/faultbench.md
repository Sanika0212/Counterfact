# FaultBench

FaultBench measures whether a method can name the step that caused an agent run to
fail. It is built so that the true answer is known exactly.

## Agents

There are three families. Each is an ordinary Python agent whose steps are Counterfact
boundaries, and all randomness comes from `cf.rng()`.

| Family | Steps | Task |
|---|---|---|
| `calc` | plan → lookup × 3 → compute → respond | Look up three values in a small knowledge base and combine them with one arithmetic operation |
| `team` | planner → (research → summarize) × 4 → review | A planner splits a report into four topics; executors research and summarize each; a reviewer assembles the report |
| `rag` | rewrite → retrieve → rerank → answer | Rewrite a question, retrieve candidate notes, rerank them and answer from the top note |

Each step can fail with probability `fault_p` (0.12 in the reported results). Failures
are **silent**: the step returns a plausible but wrong value and raises nothing. Some
examples:

1. a lookup returns a value that is off by a few;
2. a planner drops a topic;
3. a rerank puts the wrong note first;
4. a respond step garbles the number.

Correct steps also vary harmlessly, the way model outputs do. A plan may list facts in
any order, and a rewrite picks one of several paraphrases. A method therefore can't
spot the culprit by checking which output differs from a reference run.

## Ground truth

Each step type has a **validator** that checks an output against the inputs the step
actually received. For example, a summarize step is valid if it faithfully summarizes
the fact it was given, even when that fact is wrong. Validators are used only to label
instances. The methods under test never see them; blame uses only the outcome check.

`generate()` builds instances as follows:

1. For each task (a task seed), check that a fault-free agent (`fault_p = 0`) solves
   it. Unsolvable tasks are skipped.
2. Run the agent with `fault_p` under up to 400 noise seeds. Keep the failing runs in
   which exactly one step is invalid; that step is the root cause. Also remember a
   passing run for `bisect`.
3. Pick one single-fault failing run per task at random.

The reported results use 60 tasks per family (180 instances, `seed=0`).

## Methods

| Method | Uses |
|---|---|
| `random` | A uniformly random step |
| `last` | The last step, where the failure becomes visible |
| `first_error` | The first step whose span has an error status (falls back to `last`) |
| `suspicion` | A score from error status, empty output and position |
| `llm_judge` | A model reads every step's inputs and outputs and names the root cause |
| `blame` | Counterfactual replay; see [how-it-works.md](how-it-works.md) |
| `bisect` | Splicing a failing run with a passing run of the same task |
| `blame_position`, `blame_fresh_noise`, `blame_position_fresh_noise` | Ablations of blame |

A prediction is correct when it names exactly the culprit's key, for example
`team.research#1`.

## Running it

```bash
uv run counterfact bench --per-family 60 --out runs/faultbench \
  --methods last first_error random suspicion blame bisect \
            blame_position blame_fresh_noise blame_position_fresh_noise
```

Results go to `runs/faultbench/results.json` (every instance and prediction) and
`results.md` (the summary table). Running the command again with other methods on the
same instances adds them to the existing results; this is how the `llm_judge` row is
added once a key is configured.

## What it does and doesn't show

FaultBench shows that intervention identifies silent root causes in cases where
reading a trace for errors or symptoms cannot. It has known limits:

1. it is synthetic;
2. every instance has a single fault;
3. agents are fully replayable;
4. noise is explicit.

Real agents built on hosted models give up some of that last property (see Limits in
the README). Multi-fault runs are a natural extension. With `strategy="all"`, blame
already reports every step's effect, but the accuracy metric here assumes a single
culprit.
