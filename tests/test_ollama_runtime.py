from __future__ import annotations

from pathlib import Path
import sys

import requests


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import policy_dashboard.gen_bi as gen_bi


class _FakeResponse:
    def __init__(self, payload: dict[str, object], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> dict[str, object]:
        return self._payload


def test_ollama_config_reads_dotenv_and_allows_process_overrides(
    tmp_path: Path, monkeypatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OLLAMA_HOST=http://127.0.0.1:11435\nOLLAMA_MODEL=test-model:1b\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)

    config = gen_bi.get_ollama_config(env_file)
    assert config.host == "http://127.0.0.1:11435"
    assert config.model == "test-model:1b"

    monkeypatch.setenv("OLLAMA_MODEL", "process-model:3b")
    overridden = gen_bi.get_ollama_config(env_file)
    assert overridden.host == "http://127.0.0.1:11435"
    assert overridden.model == "process-model:3b"


def test_model_response_probe_uses_a_real_minimal_generation(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_post(url: str, **kwargs):
        calls.append({"url": url, **kwargs})
        return _FakeResponse({"response": "READY"})

    monkeypatch.setattr(gen_bi.requests, "post", fake_post)

    status = gen_bi.check_ollama_model_response(
        "http://127.0.0.1:11434", "test-model:1b"
    )

    assert status.is_responding
    assert status.sample_response == "READY"
    assert calls[0]["url"] == "http://127.0.0.1:11434/api/generate"
    payload = calls[0]["json"]
    assert isinstance(payload, dict)
    assert payload["model"] == "test-model:1b"
    assert payload["prompt"] == "Reply with READY only."
    assert payload["stream"] is False
    assert payload["options"]["num_predict"] == 4


def test_model_response_probe_reports_connection_failure(monkeypatch) -> None:
    def fake_post(*_args, **_kwargs):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(gen_bi.requests, "post", fake_post)

    status = gen_bi.check_ollama_model_response(
        "http://127.0.0.1:11434", "test-model:1b"
    )

    assert not status.is_responding
    assert "No response from configured Ollama endpoint" in status.message


def test_narration_is_aggregate_bound_and_has_no_server_launcher(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_post(url: str, **kwargs):
        calls.append({"url": url, **kwargs})
        return _FakeResponse({"response": "**Executive answer**\n\nReady."})

    monkeypatch.setattr(gen_bi.requests, "post", fake_post)

    answer = gen_bi.ask_ollama(
        host="http://127.0.0.1:11434",
        model="test-model:1b",
        question="Which payer needs attention?",
        context_json='{"evidence":{"payer_review":[{"payer_name":"Allianz"}]}}',
    )

    assert answer.startswith("**Executive answer**")
    payload = calls[0]["json"]
    assert isinstance(payload, dict)
    assert "Which payer needs attention?" in payload["prompt"]
    assert "payer_review" in payload["prompt"]
    assert payload["options"]["num_predict"] == 220
    assert not hasattr(gen_bi, "ensure_ollama_server")
