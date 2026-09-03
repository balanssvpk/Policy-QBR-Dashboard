from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from policy_dashboard import gen_bi


class _OllamaResponse:
    status_code = 200

    def json(self) -> dict[str, str]:
        return {"response": "**Executive answer** Reproducible response."}


def test_ollama_uses_a_fixed_greedy_generation_profile(monkeypatch) -> None:
    payloads: list[dict[str, object]] = []

    def fake_post(url: str, *, json: dict[str, object], timeout: object) -> _OllamaResponse:
        assert url == "http://127.0.0.1:11434/api/generate"
        assert timeout == (3.0, 45.0)
        payloads.append(json)
        return _OllamaResponse()

    monkeypatch.setattr(gen_bi.requests, "post", fake_post)
    request = {
        "host": "http://127.0.0.1:11434",
        "model": "llama3.2:1b",
        "question": "Which payer needs action?",
        "context_json": '{"scope":"All portfolio"}',
    }

    first = gen_bi.ask_ollama(**request)
    second = gen_bi.ask_ollama(**request)

    assert first == second == "**Executive answer** Reproducible response."
    assert payloads[0] == payloads[1]
    assert payloads[0]["options"] == {
        "seed": 42,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "min_p": 0.0,
        "mirostat": 0,
        "num_predict": 220,
        "num_ctx": 4096,
    }
