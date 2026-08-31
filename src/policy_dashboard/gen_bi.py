"""Safe, aggregate-only Ollama narration for the Gen BI sheet."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse
from uuid import uuid4

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


@dataclass
class QuestionEvidence:
    """Question-specific aggregate evidence prepared for a Gen BI response."""

    question: str
    scope: str
    focus: str
    intents: tuple[str, ...]
    requested_metrics: tuple[str, ...]
    matched_entities: dict[str, tuple[str, ...]]
    source_tables: tuple[str, ...]
    tables: dict[str, pd.DataFrame]
    limitations: tuple[str, ...]
    context_json: str
    all_metrics_json: str


_METRIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "gross_premium_usd": ("gross premium", " gross ", " gp "),
    "net_premium_usd": ("net premium", " net ", " np "),
    "tpa_fee_usd": ("tpa", "tpa fee", "fee", "fees"),
    "app_penetration_rate": (
        "app penetration",
        "mobile penetration",
        "penetration",
        "adoption",
    ),
    "active_population": ("active population", "population", "active members"),
    "active_registered_users": (
        "active registered",
        "registered users",
        "registered user",
    ),
    "unique_beneficiaries": ("beneficiary", "beneficiaries", "member", "members"),
    "net_to_gross_ratio": ("net to gross", "gross to net", "margin", "ratio"),
    "tpa_to_gross_ratio": ("tpa to gross", "tpa ratio"),
}

_METRIC_CATALOG: dict[str, dict[str, str]] = {
    "summary": {
        "policy_records": "Policy-year records in the applied scope.",
        "unique_beneficiaries": "Distinct beneficiaries in the applied scope.",
        "master_contracts": "Distinct master contracts in the applied scope.",
        "registered_users": "Distinct registered application users.",
        "registered_beneficiaries": "Beneficiaries linked to a registered user.",
        "gross_premium_usd": "Gross premium in USD.",
        "net_premium_usd": "Net premium in USD.",
        "tpa_fee_usd": "TPA fee in USD.",
        "latest_active_population": "Distinct active beneficiaries at the latest month-end.",
        "latest_active_month": "Latest active-population month-end.",
    },
    "monthly_kpis": {
        "active_population": "Distinct active beneficiaries at each month-end.",
        "active_registered_users": "Distinct active registered users at each month-end.",
        "gross_premium_usd": "Allocated gross premium in USD by active month.",
        "net_premium_usd": "Allocated net premium in USD by active month.",
        "tpa_fee_usd": "Allocated TPA fee in USD by active month.",
    },
    "monthly_country_kpis": {
        "active_population": "Distinct active beneficiaries by payer country and month-end.",
        "active_registered_users": "Distinct active registered users by payer country and month-end.",
    },
    "premium_by_year": {
        "beneficiaries": "Distinct beneficiaries by underwriting year.",
        "gross_premium_usd": "Gross premium in USD by underwriting year.",
        "net_premium_usd": "Net premium in USD by underwriting year.",
        "tpa_fee_usd": "TPA fee in USD by underwriting year.",
    },
    "payer_review": {
        "beneficiaries": "Distinct beneficiaries by payer.",
        "registered_users": "Distinct registered users by payer.",
        "gross_premium_usd": "Gross premium in USD by payer.",
        "net_premium_usd": "Net premium in USD by payer.",
        "tpa_fee_usd": "TPA fee in USD by payer.",
        "app_penetration_rate": "Registered users divided by beneficiaries by payer.",
        "net_to_gross_ratio": "Net premium divided by gross premium by payer.",
        "tpa_to_gross_ratio": "TPA fee divided by gross premium by payer.",
    },
    "policy_type_review": {
        "beneficiaries": "Distinct beneficiaries by policy type.",
        "gross_premium_usd": "Gross premium in USD by policy type.",
        "net_premium_usd": "Net premium in USD by policy type.",
        "tpa_fee_usd": "TPA fee in USD by policy type.",
    },
    "mobile_by_payer": {
        "unique_beneficiaries": "Distinct beneficiaries by payer.",
        "unique_registered_users": "Distinct registered users by payer.",
        "registered_user_penetration": "Registered users divided by beneficiaries by payer.",
        "linked_beneficiary_coverage": "Beneficiaries linked to a registered user by payer.",
    },
}

_TABLE_LABELS = {
    "summary": "Portfolio summary",
    "monthly_kpis": "Monthly operating metrics",
    "monthly_country_kpis": "Country operating metrics",
    "premium_by_year": "Premium by underwriting year",
    "payer_review": "Payer comparison",
    "policy_type_review": "Policy type comparison",
    "mobile_by_payer": "Mobile adoption by payer",
}


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
# Question-to-evidence semantic layer
# ---------------------------------------------------------------------------

def _normalise_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).strip()


def _contains_phrase(normalised_question: str, phrase: str) -> bool:
    normalised_phrase = _normalise_text(phrase)
    if not normalised_phrase:
        return False
    return f" {normalised_phrase} " in f" {normalised_question} "


def _unique_text(values: Sequence[Any]) -> list[str]:
    return list(
        dict.fromkeys(
            str(value).strip()
            for value in values
            if value is not None and str(value).strip()
        )
    )


def _aggregate_table(snapshot: dict[str, Any], source: str) -> pd.DataFrame:
    full_source = snapshot.get(f"{source}_all")
    if source in {"payer_review", "mobile_by_payer"} and isinstance(
        full_source, pd.DataFrame
    ):
        return full_source
    frame = snapshot.get(source)
    return frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()


def _entity_values(
    snapshot: dict[str, Any],
    entity_catalog: Mapping[str, Sequence[Any]] | None,
    option_key: str,
    table_name: str,
    column: str,
) -> list[str]:
    values: list[Any] = []
    frame = _aggregate_table(snapshot, table_name)
    if column in frame:
        values.extend(frame[column].dropna().tolist())
    if entity_catalog:
        values.extend(entity_catalog.get(option_key, ()))
    return _unique_text(values)


def _match_entities(question: str, values: Sequence[str]) -> tuple[str, ...]:
    normalised_question = f" {_normalise_text(question)} "
    matched: list[str] = []
    for value in sorted(values, key=lambda item: len(_normalise_text(item)), reverse=True):
        normalised_value = _normalise_text(value)
        if normalised_value and f" {normalised_value} " in normalised_question:
            matched.append(value)
    return tuple(_unique_text(matched))


def _infer_intents(
    question: str, matched_entities: Mapping[str, tuple[str, ...]]
) -> tuple[str, ...]:
    normalised_question = _normalise_text(question)

    intent_phrases = {
        "economics": (
            "premium",
            "gross",
            "net",
            "tpa",
            "fee",
            "cost",
            "economics",
            "margin",
            "ratio",
        ),
        "adoption": (
            "app",
            "mobile",
            "penetration",
            "adoption",
            "registered",
            "registration",
            "digital",
        ),
        "population": ("population", "beneficiary", "member", "active"),
        "policy": ("policy", "product", "network", "contract", "underwriting"),
        "payer": ("payer", "insurer", "carrier"),
        "country": ("country", "countries", "market", "markets"),
        "trend": (
            "trend",
            "growth",
            "change",
            "increase",
            "decrease",
            "yoy",
            "mom",
            "month",
            "year",
            "over time",
            "compare",
            "versus",
            " vs ",
        ),
    }
    selected = [
        intent
        for intent, phrases in intent_phrases.items()
        if any(_contains_phrase(normalised_question, phrase) for phrase in phrases)
    ]
    if matched_entities["payers"] and "payer" not in selected:
        selected.append("payer")
    if matched_entities["payer_countries"] and "country" not in selected:
        selected.append("country")
    if matched_entities["policy_types"] and "policy" not in selected:
        selected.append("policy")
    return tuple(selected or ["portfolio"])


def _requested_metrics(question: str, intents: Sequence[str]) -> tuple[str, ...]:
    normalised_question = _normalise_text(question)
    selected = [
        metric
        for metric, phrases in _METRIC_KEYWORDS.items()
        if any(_contains_phrase(normalised_question, phrase) for phrase in phrases)
    ]
    if selected:
        return tuple(selected)

    defaults: list[str] = []
    if "economics" in intents:
        defaults.extend(["gross_premium_usd", "net_premium_usd", "tpa_fee_usd"])
    if "adoption" in intents:
        defaults.extend(["app_penetration_rate", "active_registered_users"])
    if "population" in intents:
        defaults.append("active_population")
    if "payer" in intents or "policy" in intents:
        defaults.extend(["gross_premium_usd", "net_premium_usd", "tpa_fee_usd"])
    return tuple(_unique_text(defaults) or ["gross_premium_usd", "net_premium_usd"])


def _source_tables(
    intents: Sequence[str], matched_entities: Mapping[str, tuple[str, ...]]
) -> tuple[str, ...]:
    sources = ["summary"]
    if "economics" in intents or "trend" in intents:
        sources.append("premium_by_year")
    if "payer" in intents or matched_entities["payers"]:
        sources.append("payer_review")
    if "adoption" in intents:
        sources.extend(["mobile_by_payer", "monthly_kpis"])
    if "population" in intents:
        sources.append("monthly_kpis")
    if "country" in intents or matched_entities["payer_countries"]:
        sources.append("monthly_country_kpis")
    if "policy" in intents or matched_entities["policy_types"]:
        sources.append("policy_type_review")
    if len(sources) == 1:
        sources.extend(["premium_by_year", "payer_review"])
    return tuple(_unique_text(sources))


def _question_years(question: str) -> set[int]:
    return {int(value) for value in re.findall(r"\b20\d{2}\b", question)}


def _time_slice(frame: pd.DataFrame, question: str, max_rows: int = 24) -> pd.DataFrame:
    if frame.empty or "month_end" not in frame:
        return frame.copy()
    result = frame.copy()
    result["month_end"] = pd.to_datetime(result["month_end"])
    result = result.sort_values("month_end")
    requested_years = _question_years(question)
    if requested_years:
        selected = result.loc[result["month_end"].dt.year.isin(requested_years)]
        if not selected.empty:
            return selected
    normalised_question = _normalise_text(question)
    time_terms = ("trend", "growth", "change", "month", "yoy", "mom", "over time")
    if any(_contains_phrase(normalised_question, term) for term in time_terms):
        return result.tail(max_rows)
    return result.tail(1)


def _filter_rows(frame: pd.DataFrame, column: str, values: Sequence[str]) -> pd.DataFrame:
    if frame.empty or not values or column not in frame:
        return frame.copy()
    return frame.loc[frame[column].isin(values)].copy()


def _rank_entity_rows(frame: pd.DataFrame, question: str) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    normalised_question = _normalise_text(question)
    ascending = any(
        _contains_phrase(normalised_question, term)
        for term in ("lowest", "least", "worst", "lagging", "weakest")
    )
    if any(
        _contains_phrase(normalised_question, term)
        for term in ("app", "mobile", "penetration", "adoption", "registered")
    ):
        metric = (
            "app_penetration_rate"
            if "app_penetration_rate" in frame
            else "registered_user_penetration"
        )
    elif "tpa_to_gross_ratio" in frame and any(
        _contains_phrase(normalised_question, term)
        for term in ("tpa", "fee", "cost", "margin", "ratio")
    ):
        metric = "tpa_to_gross_ratio"
    else:
        metric = "gross_premium_usd"
    if metric not in frame:
        return frame.copy()
    return frame.sort_values(metric, ascending=ascending, na_position="last")


def _evidence_table(
    source: str,
    snapshot: dict[str, Any],
    question: str,
    matched_entities: Mapping[str, tuple[str, ...]],
) -> pd.DataFrame:
    frame = _aggregate_table(snapshot, source)
    if frame.empty and source not in snapshot and f"{source}_all" not in snapshot:
        return pd.DataFrame()
    if source == "summary":
        return frame.head(1).copy()
    if source == "premium_by_year":
        return frame.sort_values("uw_year").copy()
    if source == "monthly_kpis":
        return _time_slice(frame, question)
    if source == "monthly_country_kpis":
        result = _filter_rows(frame, "payer_country", matched_entities["payer_countries"])
        if matched_entities["payer_countries"]:
            return _time_slice(result, question)
        result = _time_slice(result, question, max_rows=72)
        if len(result) > 12 and "month_end" in result:
            latest_month = result["month_end"].max()
            return result.loc[result["month_end"] == latest_month].sort_values(
                "active_population", ascending=False
            )
        return result
    if source == "payer_review":
        result = _filter_rows(frame, "payer_name", matched_entities["payers"])
        if not matched_entities["payers"]:
            result = _rank_entity_rows(result, question).head(12)
        return result
    if source == "mobile_by_payer":
        result = _filter_rows(frame, "payer_name", matched_entities["payers"])
        if not matched_entities["payers"]:
            result = _rank_entity_rows(result, question).head(12)
        return result
    if source == "policy_type_review":
        result = _filter_rows(frame, "policy_type", matched_entities["policy_types"])
        return result if matched_entities["policy_types"] else result.head(12)
    return frame.head(12).copy()


def _project_evidence_columns(
    source: str, frame: pd.DataFrame, requested_metrics: Sequence[str]
) -> pd.DataFrame:
    """Keep the model briefing limited to the columns needed by the question."""

    identity_columns = {
        "summary": ("policy_records",),
        "monthly_kpis": ("month_end",),
        "monthly_country_kpis": ("month_end", "payer_country"),
        "premium_by_year": ("uw_year", "beneficiaries"),
        "payer_review": ("payer_name",),
        "policy_type_review": ("policy_type", "beneficiaries"),
        "mobile_by_payer": ("payer_name",),
    }
    selected = list(identity_columns.get(source, ()))
    common_metrics = {
        "gross_premium_usd": ("gross_premium_usd",),
        "net_premium_usd": ("net_premium_usd",),
        "tpa_fee_usd": ("tpa_fee_usd",),
        "net_to_gross_ratio": ("gross_premium_usd", "net_premium_usd", "net_to_gross_ratio"),
        "tpa_to_gross_ratio": ("gross_premium_usd", "tpa_fee_usd", "tpa_to_gross_ratio"),
    }
    for metric in requested_metrics:
        selected.extend(common_metrics.get(metric, ()))
        if metric == "app_penetration_rate":
            if source == "payer_review":
                selected.extend(
                    ("beneficiaries", "registered_users", "app_penetration_rate")
                )
            elif source == "mobile_by_payer":
                selected.extend(
                    (
                        "unique_beneficiaries",
                        "unique_registered_users",
                        "registered_user_penetration",
                        "linked_beneficiary_coverage",
                    )
                )
            elif source == "monthly_kpis":
                selected.extend(("active_population", "active_registered_users"))
            elif source == "summary":
                selected.extend(
                    ("unique_beneficiaries", "registered_users", "registered_beneficiaries")
                )
        elif metric == "active_population":
            selected.extend(
                ("latest_active_population", "latest_active_month", "active_population")
            )
        elif metric == "active_registered_users":
            selected.extend(("active_registered_users", "registered_users"))
        elif metric == "unique_beneficiaries":
            selected.extend(("unique_beneficiaries", "beneficiaries", "unique_beneficiaries"))

    columns = [column for column in _unique_text(selected) if column in frame]
    return frame.loc[:, columns].copy() if columns else frame.copy()


def _frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    return json.loads(
        frame.to_json(
            orient="records",
            date_format="iso",
            date_unit="s",
            double_precision=10,
        )
    )


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _all_aggregate_metrics_json(snapshot: dict[str, Any]) -> str:
    table_names = (
        "summary",
        "active_population",
        "monthly_kpis",
        "monthly_country_kpis",
        "premium_by_year",
        "payer_review",
        "policy_type_review",
        "mobile_by_payer",
    )
    payload = {
        "metric_catalog": _METRIC_CATALOG,
        "values": {
            table_name: _frame_records(_aggregate_table(snapshot, table_name))
            for table_name in table_names
            if not _aggregate_table(snapshot, table_name).empty
        },
        "metadata": snapshot.get("metadata", {}),
        "dashboard_query_ms": snapshot.get("query_ms"),
    }
    return _json_dumps(payload)


def _focus_label(
    intents: Sequence[str], matched_entities: Mapping[str, tuple[str, ...]]
) -> str:
    if matched_entities["payers"] and "adoption" in intents:
        return "Payer digital-adoption comparison"
    if matched_entities["payers"]:
        return "Named payer review"
    if matched_entities["payer_countries"]:
        return "Named country review"
    if matched_entities["policy_types"]:
        return "Named policy-type review"
    if "adoption" in intents:
        return "Digital-adoption review"
    if "economics" in intents:
        return "Premium and economics review"
    if "population" in intents:
        return "Active-population review"
    if "policy" in intents:
        return "Policy mix review"
    return "Portfolio review"


def evidence_table_label(table_name: str) -> str:
    return _TABLE_LABELS.get(table_name, table_name.replace("_", " ").title())


def build_question_evidence(
    question: str,
    snapshot: dict[str, Any],
    filter_summary: str,
    entity_catalog: Mapping[str, Sequence[Any]] | None = None,
) -> QuestionEvidence:
    """Map one executive question to the smallest relevant aggregate evidence pack."""

    cleaned_question = " ".join(str(question).split())[:800]
    if not cleaned_question:
        raise ValueError("Enter a business question before generating an answer.")

    payer_values = _entity_values(
        snapshot, entity_catalog, "payer_name", "payer_review", "payer_name"
    )
    country_values = _entity_values(
        snapshot,
        entity_catalog,
        "payer_country",
        "monthly_country_kpis",
        "payer_country",
    )
    policy_type_values = _entity_values(
        snapshot,
        entity_catalog,
        "policy_type",
        "policy_type_review",
        "policy_type",
    )
    matched_entities = {
        "payers": _match_entities(cleaned_question, payer_values),
        "payer_countries": _match_entities(cleaned_question, country_values),
        "policy_types": _match_entities(cleaned_question, policy_type_values),
    }
    intents = _infer_intents(cleaned_question, matched_entities)
    requested_metrics = _requested_metrics(cleaned_question, intents)
    source_tables = _source_tables(intents, matched_entities)
    tables = {
        source: _project_evidence_columns(
            source,
            _evidence_table(source, snapshot, cleaned_question, matched_entities),
            requested_metrics,
        )
        for source in source_tables
    }
    tables = {source: frame for source, frame in tables.items() if not frame.empty}

    limitations: list[str] = []
    if matched_entities["payers"] and tables.get("payer_review", pd.DataFrame()).empty:
        limitations.append("No payer-level aggregate row was available for the payer named in the question.")
    if (
        matched_entities["payer_countries"]
        and tables.get("monthly_country_kpis", pd.DataFrame()).empty
    ):
        limitations.append("No country-level active-population row was available for the country named in the question.")
    if (
        matched_entities["policy_types"]
        and tables.get("policy_type_review", pd.DataFrame()).empty
    ):
        limitations.append("No policy-type aggregate row was available for the policy type named in the question.")
    if "country" in intents and "economics" in intents:
        limitations.append(
            "Country-level activity metrics are available; premium economics remain aggregated at payer and policy-type level."
        )
    if len(tables) == 1:
        limitations.append("The selected scope only returned portfolio-level aggregate evidence for this question.")

    focus = _focus_label(intents, matched_entities)
    context_payload = {
        "question": cleaned_question,
        "scope": filter_summary or "All portfolio",
        "question_focus": focus,
        "intents": list(intents),
        "requested_metrics": list(requested_metrics),
        "matched_entities": {
            name: list(values) for name, values in matched_entities.items()
        },
        "evidence": {
            source: _frame_records(frame) for source, frame in tables.items()
        },
        "limitations": limitations,
    }
    return QuestionEvidence(
        question=cleaned_question,
        scope=filter_summary or "All portfolio",
        focus=focus,
        intents=intents,
        requested_metrics=requested_metrics,
        matched_entities=matched_entities,
        source_tables=tuple(tables),
        tables=tables,
        limitations=tuple(limitations),
        context_json=_json_dumps(context_payload),
        all_metrics_json=_all_aggregate_metrics_json(snapshot),
    )


def build_context(question: str, snapshot: dict[str, Any], filter_summary: str) -> str:
    """Compatibility wrapper returning a question-specific aggregate briefing."""

    return build_question_evidence(question, snapshot, filter_summary).context_json


def record_evaluation(
    *,
    evaluation_dir: str | Path,
    evidence: QuestionEvidence,
    answer: str | None,
    model: str,
    response_status: str,
    filter_spec_json: str,
    planning_ms: float | None,
    model_ms: float | None,
    dashboard_query_ms: float | None,
    error_message: str | None = None,
) -> Path:
    """Persist one aggregate-only Gen BI interaction as a Parquet dataset row."""

    timestamp = datetime.now(timezone.utc)
    output_dir = Path(evaluation_dir) / f"date={timestamp:%Y-%m-%d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    interaction_id = uuid4().hex
    output_path = output_dir / (
        f"gen_bi_{timestamp:%Y%m%dT%H%M%S%fZ}_{interaction_id}.parquet"
    )
    record = {
        "schema_version": 1,
        "interaction_id": interaction_id,
        "timestamp_utc": pd.Timestamp(timestamp),
        "question": evidence.question,
        "question_focus": evidence.focus,
        "intents_json": _json_dumps(list(evidence.intents)),
        "requested_metrics_json": _json_dumps(list(evidence.requested_metrics)),
        "matched_entities_json": _json_dumps(
            {name: list(values) for name, values in evidence.matched_entities.items()}
        ),
        "source_tables_json": _json_dumps(list(evidence.source_tables)),
        "scope": evidence.scope,
        "filter_spec_json": filter_spec_json,
        "evidence_context_json": evidence.context_json,
        "all_aggregate_metrics_json": evidence.all_metrics_json,
        "answer": answer or "",
        "model": model,
        "response_status": response_status,
        "error_message": error_message or "",
        "planning_ms": planning_ms,
        "model_ms": model_ms,
        "dashboard_query_ms": dashboard_query_ms,
    }
    pd.DataFrame([record]).to_parquet(output_path, index=False, compression="zstd")
    return output_path


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
        "You are a senior MBB engagement manager preparing a concise CXO briefing for an "
        "insurance portfolio. Answer only the business question using the supplied "
        "question-specific aggregate evidence. Never invent data, never expose or request "
        "individual member data, and do not write SQL. Use exactly this Markdown structure: "
        "**Executive answer** (one direct, decision-oriented paragraph); **Evidence** "
        "(2-4 quantified bullets that answer the question); **Recommended actions** "
        "(2-3 numbered, practical actions). Cite values precisely, distinguish facts from "
        "recommendations, and state any listed limitation plainly. Keep the response below "
        "220 words."
    )

    payload = {
        "model": model,
        "stream": False,
        "keep_alive": "30m",
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    f"Business question: {question.strip()}\n\n"
                    f"Question-specific aggregate evidence (JSON): {context}"
                ),
            },
        ],
        "options": {"temperature": 0.1, "num_predict": 220, "num_ctx": 4096},
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
