"""Pytest plugin: record agents inside tests and keep tapes for offline regression tests.

Fixtures:

- ``cf_record(agent, check=None, name=None)``: run an agent under recording and
  return the :class:`~counterfact.model.Run`. With ``--counterfact-db PATH`` every
  recorded run is also saved for later inspection (``counterfact report``).
- ``cf_tape(agent, name, check=None)``: replay the tape ``tests/tapes/<name>.tape.json``
  (or ``--counterfact-tapes DIR``), recording it first if it does not exist yet. After
  the first recording the test runs offline, with no API keys. Use
  ``--counterfact-record`` to re-record.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:  # imported lazily so the plugin adds no import cost to every test session
    from counterfact.model import Run
    from counterfact.runtime import Plan


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("counterfact")
    group.addoption("--counterfact-db", default=None, help="save recorded runs to this SQLite file")
    group.addoption("--counterfact-tapes", default="tests/tapes", help="directory of tape files")
    group.addoption(
        "--counterfact-record", action="store_true", help="re-record tapes instead of replaying"
    )


@pytest.fixture
def cf_record(request: pytest.FixtureRequest) -> Callable[..., Run]:
    from counterfact.runtime import execute
    from counterfact.store import Store

    db = request.config.getoption("--counterfact-db")
    store = Store(db) if db else None

    def record(
        agent: Callable[[], Any],
        check: Callable[[Any], bool] | None = None,
        name: str | None = None,
        plan: Plan | None = None,
    ) -> Run:
        run = execute(agent, check=check, name=name or request.node.name, plan=plan)
        if store is not None:
            store.save_run(run)
        return run

    return record


@pytest.fixture
def cf_tape(request: pytest.FixtureRequest) -> Callable[..., Run]:
    from counterfact.model import Tape
    from counterfact.runtime import Plan, execute

    directory = Path(request.config.getoption("--counterfact-tapes"))
    rerecord = bool(request.config.getoption("--counterfact-record"))

    def run_with_tape(
        agent: Callable[[], Any], name: str, check: Callable[[Any], bool] | None = None
    ) -> Run:
        path = directory / f"{name}.tape.json"
        if rerecord or not path.exists():
            run = execute(agent, check=check, name=name)
            directory.mkdir(parents=True, exist_ok=True)
            run.tape().save(path)
            return run
        plan = Plan.replay(Tape.load(path), strict=True)
        return execute(agent, check=check, name=name, plan=plan)

    return run_with_tape
