"""Safe, on-demand lifecycle support for a local Ollama runtime."""

from __future__ import annotations

from dataclasses import dataclass
import ipaddress
import os
import shutil
import subprocess
import time
from typing import Any
from urllib.parse import urlparse

import requests


DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"


@dataclass(frozen=True)
class OllamaRuntimeStatus:
    """Availability result for the configured Ollama HTTP endpoint."""

    is_ready: bool
    started: bool
    host: str
    message: str
    process_id: int | None = None


def _normalise_host(host: str) -> tuple[str, str, int]:
    raw_host = str(host or DEFAULT_OLLAMA_HOST).strip().rstrip("/")
    if "://" not in raw_host:
        raw_host = f"http://{raw_host}"
    parsed = urlparse(raw_host)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("OLLAMA_HOST must be an HTTP URL with a hostname.")
    try:
        port = parsed.port or 11434
    except ValueError as exc:
        raise ValueError("OLLAMA_HOST contains an invalid port.") from exc

    hostname = parsed.hostname
    formatted_hostname = f"[{hostname}]" if ":" in hostname else hostname
    return f"{parsed.scheme}://{formatted_hostname}:{port}", hostname, port


def _is_local_host(hostname: str) -> bool:
    if hostname.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _is_ollama_ready(host: str, timeout_seconds: float) -> bool:
    try:
        response = requests.get(
            f"{host.rstrip('/')}/api/tags",
            timeout=max(0.1, float(timeout_seconds)),
        )
    except requests.RequestException:
        return False
    return response.status_code == 200


def _popen_options(environment: dict[str, str]) -> dict[str, Any]:
    options: dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "env": environment,
    }
    if os.name == "nt":
        options["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return options


def get_ollama_serve_status(
    host: str = DEFAULT_OLLAMA_HOST,
) -> OllamaRuntimeStatus:
    """Read the configured Ollama serve availability without managing it.

    This deliberately performs only the lightweight HTTP health check.  It
    never starts, stops, or otherwise changes a local or remote runtime.
    """

    try:
        endpoint, _, _ = _normalise_host(host)
    except ValueError as exc:
        return OllamaRuntimeStatus(
            is_ready=False,
            started=False,
            host=str(host),
            message=str(exc),
        )

    if _is_ollama_ready(endpoint, timeout_seconds=0.25):
        return OllamaRuntimeStatus(
            is_ready=True,
            started=False,
            host=endpoint,
            message="The configured endpoint is available.",
        )

    return OllamaRuntimeStatus(
        is_ready=False,
        started=False,
        host=endpoint,
        message="The configured endpoint is not responding.",
    )


def ensure_ollama(
    host: str = DEFAULT_OLLAMA_HOST,
    *,
    startup_timeout_seconds: float = 8.0,
    poll_interval_seconds: float = 0.25,
    executable: str = "ollama",
) -> OllamaRuntimeStatus:
    """Ensure a local Ollama endpoint is serving before a Gen BI request.

    An already healthy endpoint is left untouched. Automatic startup is limited
    to loopback endpoints; remote endpoints are never started, stopped, or
    otherwise managed. The configured model is not pulled automatically.
    """

    try:
        endpoint, hostname, port = _normalise_host(host)
    except ValueError as exc:
        return OllamaRuntimeStatus(
            is_ready=False,
            started=False,
            host=str(host),
            message=str(exc),
        )

    if _is_ollama_ready(endpoint, timeout_seconds=0.5):
        return OllamaRuntimeStatus(
            is_ready=True,
            started=False,
            host=endpoint,
            message="Configured Ollama endpoint is already available.",
        )

    if endpoint.startswith("https://") and _is_local_host(hostname):
        return OllamaRuntimeStatus(
            is_ready=False,
            started=False,
            host=endpoint,
            message=(
                "Automatic local startup requires an HTTP OLLAMA_HOST; Ollama's "
                "local server does not provide HTTPS."
            ),
        )

    if not _is_local_host(hostname):
        return OllamaRuntimeStatus(
            is_ready=False,
            started=False,
            host=endpoint,
            message=(
                "Configured Ollama endpoint is unavailable. Automatic startup is "
                "limited to localhost endpoints."
            ),
        )

    executable_path = shutil.which(executable)
    if executable_path is None:
        return OllamaRuntimeStatus(
            is_ready=False,
            started=False,
            host=endpoint,
            message=(
                "Ollama is not installed or is not available on PATH. Install Ollama "
                "and pull the configured model before retrying."
            ),
        )

    environment = os.environ.copy()
    formatted_hostname = f"[{hostname}]" if ":" in hostname else hostname
    environment["OLLAMA_HOST"] = f"{formatted_hostname}:{port}"
    try:
        process = subprocess.Popen(
            [executable_path, "serve"],
            **_popen_options(environment),
        )
    except OSError as exc:
        return OllamaRuntimeStatus(
            is_ready=False,
            started=False,
            host=endpoint,
            message=f"Unable to start the local Ollama service: {exc}",
        )

    deadline = time.monotonic() + max(0.0, float(startup_timeout_seconds))
    while True:
        if _is_ollama_ready(endpoint, timeout_seconds=0.5):
            return OllamaRuntimeStatus(
                is_ready=True,
                started=True,
                host=endpoint,
                message="Started the local Ollama service for this Gen BI request.",
                process_id=process.pid,
            )
        exit_code = process.poll()
        if exit_code is not None:
            return OllamaRuntimeStatus(
                is_ready=False,
                started=True,
                host=endpoint,
                message=f"Local Ollama service exited during startup (exit code {exit_code}).",
                process_id=process.pid,
            )
        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0:
            break
        time.sleep(min(max(0.05, float(poll_interval_seconds)), remaining_seconds))

    return OllamaRuntimeStatus(
        is_ready=False,
        started=True,
        host=endpoint,
        message=(
            "Local Ollama service was started but did not become ready within "
            f"{startup_timeout_seconds:g} seconds."
        ),
        process_id=process.pid,
    )
