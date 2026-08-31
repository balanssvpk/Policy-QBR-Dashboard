from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import policy_dashboard.gen_bi as gen_bi


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


def test_ollama_autostart_skips_unreachable_remote_hosts(monkeypatch) -> None:
    monkeypatch.setattr(gen_bi, "ollama_server_available", lambda *_: False)

    status = gen_bi.ensure_ollama_server("http://ollama.example.internal:11434")

    assert not status.is_available
    assert not status.launch_attempted
    assert "limited to local hosts" in status.message


def test_ollama_autostart_launches_missing_local_service(monkeypatch) -> None:
    checks = iter([False, False, True])
    launched: dict[str, object] = {}
    gen_bi._ollama_launch_attempts.clear()
    monkeypatch.setattr(
        gen_bi,
        "ollama_server_available",
        lambda *_: next(checks),
    )
    monkeypatch.setattr(gen_bi.shutil, "which", lambda _: "C:\\Tools\\ollama.exe")

    def fake_popen(command, **kwargs):
        launched["command"] = command
        launched["options"] = kwargs
        return object()

    monkeypatch.setattr(gen_bi.subprocess, "Popen", fake_popen)

    status = gen_bi.ensure_ollama_server(
        "http://127.0.0.1:11434", startup_timeout_seconds=0.1
    )

    assert status.is_available
    assert status.launch_attempted
    assert launched["command"] == ["C:\\Tools\\ollama.exe", "serve"]
    assert launched["options"]["env"]["OLLAMA_HOST"] == "127.0.0.1:11434"
