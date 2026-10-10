"""Capture and replay with the real OpenAI and Anthropic SDK clients (no network)."""

import asyncio
import json
from collections.abc import Iterator

import anthropic
import httpx
import httpx2
import openai
import pytest

import counterfact.runtime as cf
from counterfact import http as cfhttp
from counterfact.genai import classify, response_attributes
from counterfact.model import SpanKind


@pytest.fixture(autouse=True)
def instrumented() -> Iterator[None]:
    cfhttp.instrument()
    yield
    cfhttp.uninstrument()


def openai_body(text: str) -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-test",
        "choices": [
            {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14},
    }


class Server:
    """Mock provider that counts calls and answers with the request's last message."""

    def __init__(self, lib=httpx) -> None:
        self.calls = 0
        self.lib = lib

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        body = json.loads(request.content)
        last = body["messages"][-1]["content"]
        if request.url.path.endswith("/messages"):
            return self.lib.Response(
                200,
                json={
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": body["model"],
                    "content": [{"type": "text", "text": f"echo:{last}"}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 5, "output_tokens": 2},
                },
            )
        if body.get("stream"):
            chunks = [
                {
                    "id": "c1",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "gpt-test",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": part},
                            "finish_reason": None,
                        }
                    ],
                }
                for part in ("ec", "ho:", last)
            ]
            sse = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
            return self.lib.Response(200, text=sse, headers={"content-type": "text/event-stream"})
        return self.lib.Response(200, json=openai_body(f"echo:{last}"))


def offline_handler(request: httpx.Request) -> httpx.Response:
    raise AssertionError("replay must not reach the network")


def openai_client(handler) -> openai.OpenAI:
    return openai.OpenAI(
        api_key="sk-test",
        base_url="https://api.openai.com/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def ask(client: openai.OpenAI, question: str) -> str:
    response = client.chat.completions.create(
        model="gpt-test", messages=[{"role": "user", "content": question}], temperature=0
    )
    return response.choices[0].message.content or ""


def test_records_real_openai_sdk_calls() -> None:
    server = Server()
    client = openai_client(server.handler)
    run = cf.execute(lambda: [ask(client, "hi"), ask(client, "bye")], name="openai")
    assert run.outcome == ["echo:hi", "echo:bye"]
    calls = run.boundaries
    assert [s.boundary.key for s in calls if s.boundary] == ["openai.chat#0", "openai.chat#1"]
    assert calls[0].kind is SpanKind.CHAT
    assert calls[0].attributes["gen_ai.request.model"] == "gpt-test"
    assert calls[0].attributes["gen_ai.provider.name"] == "openai"
    assert response_attributes(calls[0].output["body"])["gen_ai.usage.input_tokens"] == 11


def test_replay_serves_the_tape_without_network() -> None:
    first = cf.execute(lambda: ask(openai_client(Server().handler), "hi"))
    replay = cf.execute(
        lambda: ask(openai_client(offline_handler), "hi"),
        plan=cf.Plan.replay(first.tape(), strict=True),
    )
    assert replay.status == "ok"
    assert replay.outcome == "echo:hi"


def test_intervention_replaces_a_model_reply() -> None:
    server = Server()
    first = cf.execute(lambda: ask(openai_client(server.handler), "hi"))
    plan = cf.Plan.counterfactual(
        first.tape(), {"openai.chat#0": cf.Replace(cfhttp.chat_reply("patched"))}
    )
    patched = cf.execute(lambda: ask(openai_client(offline_handler), "hi"), plan=plan)
    assert patched.outcome == "patched"


def test_streaming_responses_record_and_replay() -> None:
    def stream(client: openai.OpenAI) -> str:
        chunks = client.chat.completions.create(
            model="gpt-test", messages=[{"role": "user", "content": "s"}], stream=True
        )
        return "".join(c.choices[0].delta.content or "" for c in chunks)

    first = cf.execute(lambda: stream(openai_client(Server().handler)))
    assert first.outcome == "echo:s"
    assert first.boundaries[0].attributes.get("counterfact.stream") is True
    replay = cf.execute(
        lambda: stream(openai_client(offline_handler)),
        plan=cf.Plan.replay(first.tape(), strict=True),
    )
    assert replay.outcome == "echo:s"


def test_real_anthropic_sdk_calls() -> None:
    def call(handler) -> str:
        client = anthropic.Anthropic(
            api_key="sk-ant-test",
            http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
        )
        message = client.messages.create(
            model="claude-test", max_tokens=16, messages=[{"role": "user", "content": "yo"}]
        )
        return message.content[0].text  # type: ignore[union-attr]

    first = cf.execute(lambda: call(Server(httpx2).handler))
    assert first.outcome == "echo:yo"
    span = first.boundaries[0]
    assert span.name == "anthropic.chat"
    assert span.attributes["gen_ai.request.max_tokens"] == 16
    replay = cf.execute(lambda: call(offline_handler), plan=cf.Plan.replay(first.tape()))
    assert replay.outcome == "echo:yo"


def test_async_client_and_concurrency() -> None:
    server = Server()

    async def agent(i: int) -> str:
        client = openai.AsyncOpenAI(
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(server.handler)),
        )
        response = await client.chat.completions.create(
            model="gpt-test", messages=[{"role": "user", "content": f"q{i}"}]
        )
        return response.choices[0].message.content or ""

    async def main() -> list:
        return await asyncio.gather(*(cf.aexecute(lambda i=i: agent(i)) for i in range(8)))

    runs = asyncio.run(main())
    assert [r.outcome for r in runs] == [f"echo:q{i}" for i in range(8)]
    assert all(len(r.boundaries) == 1 for r in runs)


def test_calls_outside_a_run_pass_through() -> None:
    server = Server()
    assert ask(openai_client(server.handler), "plain") == "echo:plain"
    assert server.calls == 1


def test_instrument_covers_both_http_libraries_and_restores() -> None:
    cfhttp.uninstrument()
    originals = (httpx.Client.send, httpx2.Client.send)
    assert cfhttp.instrument() == ["httpx", "httpx2"]
    assert httpx.Client.send is not originals[0]
    assert httpx2.Client.send is not originals[1]
    cfhttp.uninstrument()
    assert (httpx.Client.send, httpx2.Client.send) == originals
    cfhttp.instrument()


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://api.openai.com/v1/chat/completions", ("openai", "chat")),
        ("https://api.anthropic.com/v1/messages", ("anthropic", "chat")),
        ("http://localhost:11434/v1/chat/completions", ("openai_compatible", "chat")),
        ("https://api.openai.com/v1/embeddings", ("openai", "embeddings")),
        ("https://example.com/weather", ("", "")),
    ],
)
def test_classify(url: str, expected: tuple[str, str]) -> None:
    assert classify(url) == expected


def test_response_attributes_cover_tool_calls() -> None:
    openai_tools = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {"tool_calls": [{"function": {"name": "search"}}]},
            }
        ]
    }
    anthropic_tools = {
        "content": [{"type": "tool_use", "name": "lookup"}],
        "stop_reason": "tool_use",
    }
    assert response_attributes(openai_tools)["counterfact.tool_calls"] == ["search"]
    assert response_attributes(anthropic_tools)["gen_ai.response.finish_reasons"] == ["tool_use"]
    assert response_attributes("not a dict") == {}
