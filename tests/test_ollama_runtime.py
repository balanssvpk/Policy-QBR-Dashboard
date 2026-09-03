from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from policy_dashboard import ollama_runtime


class _RunningProcess:
    pid = 4321

    def poll(self) -> None:
        return None


def test_get_ollama_serve_status_only_checks_availability(monkeypatch) -> None:
    monkeypatch.setattr(
        ollama_runtime,
        "_is_ollama_ready",
        lambda host, timeout_seconds: True,
    )

    def no_start(*args, **kwargs):
        raise AssertionError("A passive status check must not start Ollama.")

    monkeypatch.setattr(ollama_runtime.subprocess, "Popen", no_start)

    status = ollama_runtime.get_ollama_serve_status("http://127.0.0.1:11434")

    assert status.is_ready
    assert not status.started
    assert status.process_id is None
    assert "available" in status.message.lower()


def test_ensure_ollama_leaves_a_ready_endpoint_untouched(monkeypatch) -> None:
    monkeypatch.setattr(
        ollama_runtime,
        "_is_ollama_ready",
        lambda host, timeout_seconds: True,
    )

    def no_start(*args, **kwargs):
        raise AssertionError("A healthy endpoint must not be restarted.")

    monkeypatch.setattr(ollama_runtime.subprocess, "Popen", no_start)

    status = ollama_runtime.ensure_ollama("http://127.0.0.1:11434")

    assert status.is_ready
    assert not status.started
    assert status.process_id is None


def test_ensure_ollama_never_starts_a_remote_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(
        ollama_runtime,
        "_is_ollama_ready",
        lambda host, timeout_seconds: False,
    )

    def no_start(*args, **kwargs):
        raise AssertionError("Remote endpoints must not be managed locally.")

    monkeypatch.setattr(ollama_runtime.subprocess, "Popen", no_start)

    status = ollama_runtime.ensure_ollama("https://ollama.example.test:11434")

    assert not status.is_ready
    assert not status.started
    assert "localhost" in status.message


def test_ensure_ollama_does_not_start_an_https_loopback_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(
        ollama_runtime,
        "_is_ollama_ready",
        lambda host, timeout_seconds: False,
    )

    def no_start(*args, **kwargs):
        raise AssertionError("An HTTPS loopback endpoint cannot be served by Ollama.")

    monkeypatch.setattr(ollama_runtime.subprocess, "Popen", no_start)

    status = ollama_runtime.ensure_ollama("https://127.0.0.1:11434")

    assert not status.is_ready
    assert not status.started
    assert "HTTP OLLAMA_HOST" in status.message


def test_ensure_ollama_starts_a_local_runtime_without_a_shell(monkeypatch) -> None:
    readiness = iter((False, True))
    monkeypatch.setattr(
        ollama_runtime,
        "_is_ollama_ready",
        lambda host, timeout_seconds: next(readiness),
    )
    monkeypatch.setattr(
        ollama_runtime.shutil,
        "which",
        lambda executable: "C:\\Program Files\\Ollama\\ollama.exe",
    )
    calls: list[tuple[list[str], dict[str, object]]] = []

    def start(command, **kwargs):
        calls.append((command, kwargs))
        return _RunningProcess()

    monkeypatch.setattr(ollama_runtime.subprocess, "Popen", start)

    status = ollama_runtime.ensure_ollama("http://127.0.0.1:11434")

    assert status.is_ready
    assert status.started
    assert status.process_id == 4321
    assert calls[0][0] == ["C:\\Program Files\\Ollama\\ollama.exe", "serve"]
    assert "shell" not in calls[0][1]
    assert calls[0][1]["env"]["OLLAMA_HOST"] == "127.0.0.1:11434"
