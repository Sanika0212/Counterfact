import string
import threading
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import counterfact.runtime as cf
from counterfact.model import SpanKind
from counterfact.redact import Redactor
from counterfact.store import Store

SECRETS = {
    "openai_key": "sk-proj-" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4",
    "anthropic_key": "sk-ant-api03-" + "Zz9Yy8Xx7Ww6Vv5Uu4Tt3",
    "google_key": "AIza" + "B" * 35,
    "aws_access_key": "AKIA" + "ABCDEFGHIJKLMNOP",
    "github_token": "ghp_" + "a" * 36,
    "jwt": "eyJhbGciOiJIUzI1.eyJzdWIiOiIxMjM0.SflKxwRJSMeKKF2QT4",
    "email": "ada.lovelace@example.org",
    "phone": "(716) 555-0199",
    "card_number": "4111 1111 1111 1111",
}


@pytest.mark.parametrize(("rule", "secret"), list(SECRETS.items()))
def test_each_rule_redacts_its_secret(rule: str, secret: str) -> None:
    redactor = Redactor()
    out = redactor.text(f"before {secret} after")
    assert secret not in out
    assert f"[REDACTED:{rule}]" in out
    assert out.startswith("before ")
    assert out.endswith(" after")


def test_invalid_card_number_is_kept() -> None:
    assert "4111 1111 1111 1112" in Redactor().text("card 4111 1111 1111 1112")


def test_sensitive_keys_and_nesting() -> None:
    redactor = Redactor()
    payload = {
        "headers": {"Authorization": "Bearer abc", "accept": "json"},
        "messages": [{"content": "mail me at a@b.co"}],
        "n": 3,
    }
    out = redactor(payload)
    assert out["headers"]["Authorization"] == "[REDACTED:sensitive_key]"
    assert out["headers"]["accept"] == "json"
    assert out["messages"][0]["content"] == "mail me at [REDACTED:email]"
    assert out["n"] == 3
    assert redactor.counts["email"] == 1


def test_rules_can_be_disabled() -> None:
    assert "a@b.co" in Redactor.without(["email"]).text("a@b.co")


@settings(max_examples=150)
@given(
    prefix=st.text(alphabet=string.ascii_letters + " .,", max_size=40),
    body=st.text(alphabet=string.ascii_letters + string.digits, min_size=24, max_size=48),
    suffix=st.text(alphabet=string.ascii_letters + " .,", max_size=40),
)
def test_generated_keys_never_survive(prefix: str, body: str, suffix: str) -> None:
    secret = "sk-ant-" + body
    assert secret not in Redactor().text(f"{prefix} {secret} {suffix}")


@settings(max_examples=150)
@given(st.text(alphabet=string.ascii_letters + " .,;:!?'-", max_size=200))
def test_plain_prose_is_untouched(text: str) -> None:
    assert Redactor().text(text) == text


def _chat_agent(turns: int):
    @cf.boundary(kind=SpanKind.CHAT)
    def chat(messages: list[dict[str, str]]) -> str:
        return f"reply {len(messages)} " + "x" * 200

    def agent() -> int:
        messages = [{"role": "system", "content": "You are helpful. " * 40}]
        for turn in range(turns):
            messages.append({"role": "user", "content": f"question {turn} " + "y" * 200})
            messages.append({"role": "assistant", "content": chat(messages=list(messages))})
        return len(messages)

    return agent


def test_round_trip_preserves_runs_and_redacts_secrets() -> None:
    store = Store()

    @cf.boundary
    def lookup(key: str) -> str:
        return f"value for {key}"

    run = cf.execute(lambda: lookup(SECRETS["openai_key"]), name="secret", check=lambda _: True)
    store.save_run(run)
    loaded = store.load_run(run.run_id)
    assert loaded.name == "secret"
    assert loaded.passed is True
    assert [s.boundary for s in loaded.boundaries] == [s.boundary for s in run.boundaries]
    stored_input = loaded.boundaries[0].input
    assert SECRETS["openai_key"] not in str(stored_input)
    assert "[REDACTED:openai_key]" in str(stored_input)
    assert loaded.tape().keys() == run.tape().keys()


def test_message_level_dedup_shrinks_conversations() -> None:
    store = Store()
    run = cf.execute(_chat_agent(30), name="chat")
    store.save_run(run)
    stats = store.stats()
    logical = sum(len(str(s.input)) + len(str(s.output)) for s in run.spans)
    assert stats["unique_raw_bytes"] * 5 < logical  # each message stored once
    assert stats["stored_bytes"] * 10 < logical  # and compressed on disk
    loaded = store.load_run(run.run_id)
    assert loaded.boundaries[-1].input == run.boundaries[-1].input


def test_listing_and_parent_filter(tmp_path: Path) -> None:
    store = Store(tmp_path / "runs.cf.sqlite")
    parent = cf.execute(lambda: 1, name="parent")
    child = cf.execute(lambda: 2, name="child", parent_run_id=parent.run_id)
    store.save_run(parent)
    store.save_run(child)
    assert {r.name for r in store.list_runs()} == {"parent", "child"}
    assert [r.name for r in store.list_runs(parent=parent.run_id)] == ["child"]
    with pytest.raises(KeyError):
        store.load_run("missing")


def test_streaming_writer_and_threads(tmp_path: Path) -> None:
    store = Store(tmp_path / "stream.cf.sqlite")
    on_span, close = store.span_writer()
    run = cf.execute(_chat_agent(3), name="streamed", on_span=on_span)
    close()
    store.save_run(run)
    assert store.load_run(run.run_id).tape().keys() == run.tape().keys()

    errors: list[BaseException] = []

    def save_many(k: int) -> None:
        try:
            for i in range(5):
                store.save_run(cf.execute(lambda i=i: i, name=f"t{k}-{i}"))
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=save_many, args=(k,)) for k in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert store.stats()["runs"] == 21
