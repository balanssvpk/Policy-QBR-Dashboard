"""Safe, aggregate-only Ollama narration for the Gen BI sheet."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import requests


DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "llama3.2:1b"
_DOTENV_LINE = re.compile(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")
_LOCAL_OLLAMA_HOSTS = {"127.0.0.1", "localhost", "::1"}
_OLLAMA_LAUNCH_COOLDOWN_SECONDS = 10.0
_ollama_launch_lock = threading.Lock()
_ollama_launch_attempts: dict[str, float] = {}


@dataclass(frozen=True)
class OllamaConfig:
    host: str
    model: str


@dataclass(frozen=True)
class OllamaServiceStatus:
    is_available: bool
    launch_attempted: bool
    message: str


def _read_dotenv(path: Path) -> dict[str, str]:
    """Read simple KEY=VALUE pairs without adding a runtime dependency."""

    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _DOTENV_LINE.match(line)
        if not match:
            continue
        key, raw_value = match.groups()
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        else:
            value = value.split(" #", maxsplit=1)[0].strip()
        values[key] = value
    return values


def get_ollama_config(env_path: str | Path) -> OllamaConfig:
    """Resolve local .env values, allowing explicit process env overrides."""

    values = _read_dotenv(Path(env_path))
    host = os.getenv("OLLAMA_HOST") or values.get("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST
    model = os.getenv("OLLAMA_MODEL") or values.get("OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL
    return OllamaConfig(host=host.strip().rstrip("/"), model=model.strip())


def ollama_server_available(host: str, timeout_seconds: float = 0.25) -> bool:
    """Return whether an Ollama API is already serving at the configured endpoint."""

    try:
        response = requests.get(
            f"{host.rstrip('/')}/api/version", timeout=timeout_seconds
        )
        response.raise_for_status()
        return True
    except requests.RequestException:
        return False


def _local_ollama_bind_address(host: str) -> str | None:
    parsed = urlparse(host if "://" in host else f"http://{host}")
    if parsed.hostname not in _LOCAL_OLLAMA_HOSTS or not parsed.netloc:
        return None
    return parsed.netloc


def ensure_ollama_server(
    host: str, startup_timeout_seconds: float = 2.0
) -> OllamaServiceStatus:
    """Check Ollama, launching a missing local service once per process window."""

    if ollama_server_available(host):
        return OllamaServiceStatus(True, False, "Ollama is ready.")

    bind_address = _local_ollama_bind_address(host)
    if bind_address is None:
        return OllamaServiceStatus(
            False,
            False,
            f"Ollama is not reachable at {host}. Automatic startup is limited to local hosts.",
        )

    executable = shutil.which("ollama")
    if executable is None:
        return OllamaServiceStatus(
            False,
            False,
            "Ollama is not installed or is not available on PATH.",
        )

    with _ollama_launch_lock:
        if ollama_server_available(host):
            return OllamaServiceStatus(True, False, "Ollama is ready.")

        now = time.monotonic()
        last_attempt = _ollama_launch_attempts.get(host)
        if (
            last_attempt is not None
            and now - last_attempt < _OLLAMA_LAUNCH_COOLDOWN_SECONDS
        ):
            return OllamaServiceStatus(
                False,
                True,
                "Ollama is still starting. Refresh shortly if it remains unavailable.",
            )

        _ollama_launch_attempts[host] = now
        launch_env = os.environ.copy()
        launch_env["OLLAMA_HOST"] = bind_address
        launch_options: dict[str, Any] = {
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "env": launch_env,
        }
        if os.name == "nt":
            launch_options["creationflags"] = getattr(
                subprocess, "CREATE_NO_WINDOW", 0
            )
        else:
            launch_options["start_new_session"] = True

        try:
            subprocess.Popen([executable, "serve"], **launch_options)
        except OSError as exc:
            return OllamaServiceStatus(
                False,
                False,
                f"Ollama could not be started: {exc}",
            )

    deadline = time.monotonic() + startup_timeout_seconds
    while time.monotonic() < deadline:
        if ollama_server_available(host):
            return OllamaServiceStatus(True, True, "Ollama started successfully.")
        time.sleep(0.1)

    return OllamaServiceStatus(
        False,
        True,
        "Ollama was started and is still warming up. Refresh shortly to continue.",
    )


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

def _number(value: Any) -> float:
    try:
        if value is None or pd.isna(value):
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def _pct(value: float) -> str:
    return f"{value:.1%}" if pd.notna(value) else "n/a"


def _money(value: Any) -> str:
    amount = _number(value)
    for threshold, suffix in ((1_000_000_000, "bn"), (1_000_000, "m"), (1_000, "k")):
        if abs(amount) >= threshold:
            return f"${amount / threshold:,.1f}{suffix}"
    return f"${amount:,.0f}"


def _records(frame: pd.DataFrame, columns: list[str], limit: int = 8) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    cols = [c for c in columns if c in frame.columns]
    return frame.loc[:, cols].head(limit).round(4).to_dict(orient="records")


# ---------------------------------------------------------------------------
# Deterministic insight (no LLM)
# ---------------------------------------------------------------------------

def deterministic_insight(snapshot: dict[str, Any]) -> str:
    summary = snapshot["summary"].iloc[0]
    annual = snapshot["premium_by_year"].sort_values("uw_year")
    payer_review = snapshot["payer_review"]
    mobile = snapshot["mobile_by_payer"]

    gp = _number(summary.get("gross_premium_usd"))
    np = _number(summary.get("net_premium_usd"))
    tpa = _number(summary.get("tpa_fee_usd"))
    beneficiaries = int(_number(summary.get("unique_beneficiaries")))
    registered_users = int(_number(summary.get("registered_users")))

    margin = (gp - np) / gp if gp else float("nan")
    tpa_ratio = tpa / gp if gp else float("nan")

    bullets = [
        f"The selected portfolio covers {beneficiaries:,.0f} unique beneficiaries, with "
        f"{_money(gp)} GP, {_money(np)} NP, and {_money(tpa)} TPA fees.",
        f"Net-to-gross economics imply a {_pct(margin)} gross-to-net spread; "
        f"TPA fees are {_pct(tpa_ratio)} of GP.",
    ]

    if len(annual) >= 2:
        previous = _number(annual.iloc[-2].get("gross_premium_usd"))
        latest = _number(annual.iloc[-1].get("gross_premium_usd"))
        change = (latest / previous - 1) if previous else float("nan")
        bullets.append(
            f"GP in {int(annual.iloc[-1]['uw_year'])} is {_pct(change)} versus the prior underwriting year."
        )

    if not payer_review.empty and gp:
        top = payer_review.iloc[0]
        share = _number(top["gross_premium_usd"]) / gp
        bullets.append(
            f"{top['payer_name']} is the largest selected payer at {_pct(share)} of GP; "
            "review concentration and renewal leverage."
        )

    if beneficiaries:
        app_rate = registered_users / beneficiaries
        bullets.append(
            f"There are {registered_users:,.0f} unique registered users, equal to {_pct(app_rate)} "
            "of unique beneficiaries."
        )

    if not mobile.empty:
        lowest = mobile.sort_values("registered_user_penetration", na_position="last").iloc[0]
        bullets.append(
            f"A practical adoption focus is {lowest['payer_name']}, currently at "
            f"{_pct(_number(lowest['registered_user_penetration']))} registered-user penetration."
        )

    return "\n".join(f"- {b}" for b in bullets[:5])


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def build_context(question: str, snapshot: dict[str, Any], filter_summary: str) -> str:
    summary = snapshot["summary"].iloc[0].to_dict()

    headline_keys = {
        "policy_records",
        "unique_beneficiaries",
        "master_contracts",
        "registered_users",
        "registered_beneficiaries",
        "gross_premium_usd",
        "net_premium_usd",
        "tpa_fee_usd",
        "latest_active_population",
        "latest_active_month",
    }

    context = {
        "question": question.strip()[:800],
        "scope": filter_summary,
        "headline_metrics": {
            k: (None if pd.isna(v) else v)
            for k, v in summary.items()
            if k in headline_keys
        },
        "annual_premium": _records(
            snapshot["premium_by_year"],
            ["uw_year", "beneficiaries", "gross_premium_usd", "net_premium_usd", "tpa_fee_usd"],
        ),
        "top_payers": _records(
            snapshot["payer_review"],
            [
                "payer_name",
                "beneficiaries",
                "registered_users",
                "gross_premium_usd",
                "net_premium_usd",
                "tpa_fee_usd",
                "app_penetration_rate",
                "net_to_gross_ratio",
                "tpa_to_gross_ratio",
            ],
        ),
        "mobile_adoption": _records(
            snapshot["mobile_by_payer"],
            [
                "payer_name",
                "unique_beneficiaries",
                "unique_registered_users",
                "registered_user_penetration",
                "linked_beneficiary_coverage",
            ],
        ),
    }

    return str(context)


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

def available_models(host: str, timeout_seconds: float = 1.0) -> list[str]:
    try:
        r = requests.get(f"{host.rstrip('/')}/api/tags", timeout=timeout_seconds)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Ollama call
# ---------------------------------------------------------------------------

def ask_ollama(
    *,
    host: str,
    model: str,
    question: str,
    context: str,
    timeout_seconds: int = 90,
) -> str:

    system = (
        "You are a precise insurance portfolio strategy analyst. Use only the aggregate "
        "evidence in the supplied briefing. Never invent data, never expose or request "
        "individual member data, and do not write SQL. Answer in an executive consulting "
        "style: a direct answer, 2-4 quantified insights, then 2-3 practical actions. "
        "State a data limitation when the briefing cannot support a conclusion. Keep the "
        "response below 180 words."
    )

    payload = {
        "model": model,
        "stream": False,
        "keep_alive": "30m",
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": f"Question: {question.strip()}\n\nAggregate briefing: {context}",
            },
        ],
        "options": {"temperature": 0.1, "num_predict": 180, "num_ctx": 3072},
    }

    r = requests.post(
        f"{host.rstrip('/')}/api/chat",
        json=payload,
        timeout=timeout_seconds,
    )
    r.raise_for_status()

    content = r.json().get("message", {}).get("content", "")
    content = content.strip()

    if not content:
        raise RuntimeError("Ollama returned an empty response.")

    return content


