# How Counterfact works

This page describes the runtime that makes counterfactual replay possible, then the
two attribution methods built on it.

## Runs, boundaries and keys

`cf.execute(agent, ...)` runs a zero-argument callable and records a `Run`: a tree of
spans plus the agent's outcome and, if a `check` is given, whether it passed.

A **boundary** is a call whose result Counterfact records and can later replay or
change. It can be a function decorated with `@cf.boundary`, a call wrapped with
`cf.call(name, fn, ...)`, or any HTTP request made through an instrumented client. Each
boundary call gets:

| Field | Meaning |
|---|---|
| `key` | `"<name>#<occurrence>"`, e.g. `research#1` for the second `research` call. It stays stable across replays as long as the agent makes the same calls. |
| `fingerprint` | Hash of the call's canonicalised inputs. |
| `seed` | The call's own noise seed, derived from the run seed and the key. |
| `source` | `live`, `tape` (served from a recording) or `intervention`. |

Spans that are not boundaries (`cf.span(...)`, agents, workflows) only structure the
timeline. They are never replayed.

## Noise

Inside a boundary, `cf.rng()` returns a `random.Random` seeded with that call's seed.
Routing all randomness through it gives each call an explicit, recorded noise term:
the call's output is a function of its inputs and its seed. Counterfactuals rely on
exactly this. Holding a seed fixed asks "what would this call have done with the same
luck?"

Model APIs that sample on the server have no seed you can hold. Counterfact records
their responses instead, so they replay exactly when their inputs are unchanged. When a
counterfactual changes their inputs, they are sampled again live (see Limits in the
README).

## Plans

A `Plan` decides how each boundary call is resolved, in this order:

1. **Intervention on this key.** `Replace(value)` returns `value` without running the
   call. `Resample(salt)` runs the call live with a new seed derived from the recorded
   one.
2. **Recorded call with the same key and fingerprint.** The recording is returned
   (`source="tape"`). A recorded exception is raised again.
3. **Recorded call that moved.** With `match="content"` (the default), a recording with
   the same name and fingerprint at another position is reused, along with its seed.
   This happens when an intervention changes control flow, for example when a planner
   reorders its steps.
4. **Strict mode.** If none of the above applies, the call raises `Divergence`. Tests
   use this to fail loudly when an agent's behaviour changes.
5. **Otherwise the call runs live** with its recorded seed (`reuse_noise=True`, the
   default) or with fresh noise (`reuse_noise=False`, an ablation).

`Plan.replay(tape)` reproduces a run. `Plan.counterfactual(tape, {key: intervention})`
changes one or more steps and keeps everything else.

## Blame

For each candidate step, in recorded order, `blame` runs `samples` counterfactual
replays in which only that step is repaired. By default the repair is `Resample`, so
the step gets a different draw of its own noise. The step's effect is the fraction of
replays that pass, with a Wilson score interval. A step is **sufficient** if at least
`threshold` (default 0.5) of its replays pass. The culprit is the earliest sufficient
step.

Strategies:

1. `earliest` (default) stops at the first sufficient step, using the fewest replays.
2. `all` measures every step. The report and the animation use this mode.
3. `ranked` tries steps in order of a cheap suspicion score (errors, empty outputs,
   position), within an optional `budget`.

Why "earliest sufficient" rather than "most effective"? A wrong value usually flows
downstream, so the root cause comes first. Repairing a later step rarely helps,
because that step faithfully processes the bad input it received.

## Bisect

Given a failing run and a passing run of the same task, `bisect` builds spliced plans.
Steps up to a cut point take the failing run's recordings and seeds; steps after it
take the passing run's. It checks that the two ends reproduce (all-passing passes,
all-failing fails), then binary-searches the cut. The answer is the first step whose
failing version turns a passing run into a failing one. It needs no repair function and
about log₂(steps) + 2 replays.

## Diff

`diff_runs` aligns two runs' boundary calls by name with Needleman-Wunsch alignment,
so inserted, removed or repeated calls don't shift every later row. Each aligned pair is
then classified as `same`, `input-changed`, `output-changed`, `only-left` or
`only-right`. The first row that isn't `same` is where the runs diverge.

## Storage

`Store` is a SQLite database with three tables: runs, spans and content-addressed
blobs. Inputs and outputs are canonicalised, redacted, and stored once per distinct
value. Chat message lists are stored as hash chains, one node per message, so a
conversation whose prompt grows by one message per turn adds one node per turn rather
than a full copy.
