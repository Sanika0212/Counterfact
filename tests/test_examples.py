import runpy
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_quickstart_finds_the_planted_fault(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(EXAMPLES / "quickstart.py"), run_name="__main__")
    out = capsys.readouterr().out
    assert "root cause: lookup#0 after 3 replays" in out
    assert (tmp_path / "quickstart-report.html").exists()
