# Contributing

Thanks for your interest in Counterfact.

## Development setup

```bash
git clone https://github.com/Sanika0212/Counterfact.git
cd Counterfact
uv sync --extra viz
uv run pre-commit install
```

## Checks

Every pull request must pass the same checks CI runs:

```bash
uv run ruff check src tests scripts examples
uv run ruff format --check src tests scripts examples
uv run mypy
uv run pytest --cov --cov-fail-under=85
uv run counterfact demo --out demo
```

## Ground rules

- **Tests never touch the network or a paid API.** Model SDKs are exercised against
  `httpx.MockTransport`; the LLM judge is tested with a fake client.
- **Replay is exact.** A recorded run replayed with `Plan.replay(tape, strict=True)` must
  reproduce its outcome without a single live call (`assert_replays`).
- **Results come from runs.** Numbers in the README must be reproducible with
  `counterfact bench` and `scripts/make_figures.py`.
- **No secrets in recordings.** New capture paths go through `Redactor` before
  anything reaches the store.
- Use [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`,
  `docs:`, `test:`, `chore:`, `ci:`).
