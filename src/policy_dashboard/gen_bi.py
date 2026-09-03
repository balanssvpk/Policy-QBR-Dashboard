"""Fast, deterministic, aggregate-only Gen BI responses for the CXO sheet."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from time import perf_counter
from typing import Any, Mapping, Sequence
from uuid import uuid4

import pandas as pd
import requests


DETERMINISTIC_ENGINE = "deterministic-cxo-engine-v1"
OLLAMA_ENGINE = "ollama"
OLLAMA_REPRODUCIBILITY_PROFILE = "greedy-seed-42-v1"
_DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
_DEFAULT_OLLAMA_MODEL = "llama3.2:1b"
_MAX_MODEL_CONTEXT_CHARS = 18_000
_OLLAMA_REPRODUCIBILITY_OPTIONS = {
    "seed": 42,
    "temperature": 0.0,
    "top_k": 1,
    "top_p": 1.0,
    "min_p": 0.0,
    "mirostat": 0,
}


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


@dataclass(frozen=True)
class OllamaConfig:
    """Configured endpoint and model for the optional local narrator."""

    host: str
    model: str


@dataclass(frozen=True)
class OllamaModelStatus:
    """Result of a real, minimal model-generation probe."""

    is_responding: bool
    message: str
    latency_ms: float | None = None
    sample_response: str | None = None


class OllamaResponseError(RuntimeError):
    """Raised when Ollama cannot return a usable model response."""


def _read_dotenv(path: Path) -> dict[str, str]:
    """Read the few configuration values needed without a dotenv dependency."""

    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        key, separator, value = line.partition("=")
        if not separator or not key.strip():
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key.strip()] = value
    return values


def get_ollama_config(env_file: str | Path | None = None) -> OllamaConfig:
    """Read `.env` defaults, with process environment taking precedence."""

    path = (
        Path(env_file)
        if env_file is not None
        else Path(__file__).resolve().parents[2] / ".env"
    )
    dotenv = _read_dotenv(path)
    host = (
        os.getenv("OLLAMA_HOST")
        or dotenv.get("OLLAMA_HOST")
        or _DEFAULT_OLLAMA_HOST
    ).strip().rstrip("/")
    model = (
        os.getenv("OLLAMA_MODEL")
        or dotenv.get("OLLAMA_MODEL")
        or _DEFAULT_OLLAMA_MODEL
    ).strip()
    return OllamaConfig(
        host=host or _DEFAULT_OLLAMA_HOST,
        model=model or _DEFAULT_OLLAMA_MODEL,
    )


def _ollama_generate(
    *,
    host: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout_seconds: float,
    keep_alive: str,
) -> str:
    """Make one non-streaming generation request; never starts a server."""

    endpoint = f"{host.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": keep_alive,
        "options": {
            **_OLLAMA_REPRODUCIBILITY_OPTIONS,
            "num_predict": max_tokens,
            "num_ctx": 4096,
        },
    }
    try:
        response = requests.post(
            endpoint,
            json=payload,
            timeout=(3.0, max(1.0, float(timeout_seconds))),
        )
    except requests.RequestException as exc:
        raise OllamaResponseError(
            f"No response from configured Ollama endpoint {host}: {exc}"
        ) from exc

    try:
        body = response.json()
    except ValueError as exc:
        raise OllamaResponseError(
            f"Ollama returned a non-JSON response (HTTP {response.status_code})."
        ) from exc

    if response.status_code >= 400:
        detail = body.get("error") if isinstance(body, dict) else None
        raise OllamaResponseError(
            f"Ollama rejected model '{model}' (HTTP {response.status_code}): "
            f"{detail or 'unknown error'}"
        )

    answer = body.get("response") if isinstance(body, dict) else None
    answer = str(answer or "").strip()
    if not answer:
        detail = body.get("error") if isinstance(body, dict) else None
        raise OllamaResponseError(
            f"Model '{model}' returned no text{f': {detail}' if detail else ''}."
        )
    return answer


# def build_ollama_prompt(question: str, context_json: str) -> str:
#     """Create a compact, evidence-bound CXO prompt from aggregate data only."""

#     context = str(context_json or "").strip()
#     if len(context) > _MAX_MODEL_CONTEXT_CHARS:
#         context = context[:_MAX_MODEL_CONTEXT_CHARS] + "\n[Evidence truncated after aggregate rows.]"
#     return (
#         "You are an insurance strategy consultant preparing a concise CXO answer.\n"
#         "Use only the aggregate evidence below. Do not infer missing figures, write SQL, "
#         "or mention data you were not given.\n"
#         "Return Markdown with exactly these sections: Executive answer, Evidence, and "
#         "Recommended actions. Make the answer decision-oriented, factual, and under 190 words.\n\n"
#         f"Executive question:\n{question.strip()}\n\n"
#         f"Aggregate evidence (JSON):\n{context}"
#     )

def build_ollama_prompt(question: str, context_json: str) -> str:
    """
    Create a governed CXO‑grade prompt for aggregate‑only Generative BI.
    Ensures: no hallucination, no SQL, no invented metrics, no raw‑data inference.
    """

    context = str(context_json or "").strip()
    if len(context) > _MAX_MODEL_CONTEXT_CHARS:
        context = (
            context[:_MAX_MODEL_CONTEXT_CHARS]
            + "\n[Aggregate evidence truncated due to size.]"
        )

    return (
        "You are a senior analytics strategist preparing a concise CXO‑grade answer. "
        "Use ONLY the aggregate metrics and tables provided. Do not infer missing values, "
        "do not reference raw data, and do not generate SQL.\n\n"
        "Your response MUST be under 190 words and MUST contain exactly three Markdown "
        "sections in this order:\n"
        "1. **Executive answer** — A direct, decision‑ready conclusion grounded only in the evidence.\n"
        "2. **Evidence used** — Cite the specific aggregate fields and values that support your answer.\n"
        "3. **Recommended actions** — 2–4 practical next steps.\n\n"
        "Guidelines:\n"
        "- Be factual, concise, and quantitative.\n"
        "- If evidence is insufficient, state the limitation explicitly.\n"
        "- Do not speculate or invent metrics.\n"
        "- Maintain an executive consulting tone.\n\n"
        f"Executive question:\n{question.strip()}\n\n"
        f"Aggregate evidence (JSON):\n{context}"
    )



def ask_ollama(
    *,
    host: str,
    model: str,
    question: str,
    context_json: str,
    timeout_seconds: float = 45.0,
) -> str:
    """Generate a concise narrative from a question-specific aggregate briefing."""

    return _ollama_generate(
        host=host,
        model=model,
        prompt=build_ollama_prompt(question, context_json),
        max_tokens=220,
        timeout_seconds=timeout_seconds,
        keep_alive="10m",
    )


def check_ollama_model_response(
    host: str,
    model: str,
    *,
    timeout_seconds: float = 20.0,
) -> OllamaModelStatus:
    """Confirm that the configured model can generate text with a tiny probe.

    This intentionally performs a real model call instead of checking only the
    HTTP endpoint. It does not launch, stop, or otherwise manage Ollama.
    """

    started = perf_counter()
    try:
        sample = _ollama_generate(
            host=host,
            model=model,
            prompt="Reply with READY only.",
            max_tokens=4,
            timeout_seconds=timeout_seconds,
            keep_alive="1m",
        )
    except OllamaResponseError as exc:
        return OllamaModelStatus(
            is_responding=False,
            message=str(exc),
            latency_ms=round((perf_counter() - started) * 1000, 1),
        )
    return OllamaModelStatus(
        is_responding=True,
        message=f"Configured model '{model}' responded successfully.",
        latency_ms=round((perf_counter() - started) * 1000, 1),
        sample_response=sample,
    )


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
    "master_contract_network_premium": {
        "gross_premium_usd": "Gross premium in USD by master contract and network type.",
    },
    "age_bucket_review": {
        "beneficiaries": "Distinct beneficiaries by age bucket.",
        "registered_users": "Distinct registered users by age bucket.",
        "gross_premium_usd": "Gross premium in USD by age bucket.",
        "net_premium_usd": "Net premium in USD by age bucket.",
        "tpa_fee_usd": "TPA fee in USD by age bucket.",
        "app_penetration_rate": "Registered users divided by beneficiaries by age bucket.",
        "net_to_gross_ratio": "Net premium divided by gross premium by age bucket.",
        "tpa_to_gross_ratio": "TPA fee divided by gross premium by age bucket.",
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
    "master_contract_network_premium": "Master-contract network mix",
    "age_bucket_review": "Age-bucket demographic comparison",
    "policy_type_review": "Policy type comparison",
    "mobile_by_payer": "Mobile adoption by payer",
}


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
        "policy": ("policy", "product", "network", "underwriting"),
        "master_contract": ("master contract", "contract"),
        "demographic": (
            "demographic",
            "age bucket",
            "age band",
            "age group",
            "age profile",
        ),
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
    if matched_entities["master_contracts"] and "master_contract" not in selected:
        selected.append("master_contract")
    if matched_entities["age_buckets"] and "demographic" not in selected:
        selected.append("demographic")
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
    if ("economics" in intents or "trend" in intents) and "master_contract" not in intents:
        sources.append("premium_by_year")
    if "payer" in intents or matched_entities["payers"]:
        sources.append("payer_review")
    if "adoption" in intents and "demographic" not in intents:
        sources.extend(["mobile_by_payer", "monthly_kpis"])
    if "population" in intents and "demographic" not in intents:
        sources.append("monthly_kpis")
    if "country" in intents or matched_entities["payer_countries"]:
        sources.append("monthly_country_kpis")
    if (
        ("policy" in intents or matched_entities["policy_types"])
        and "master_contract" not in intents
    ):
        sources.append("policy_type_review")
    if "master_contract" in intents or matched_entities["master_contracts"]:
        sources.append("master_contract_network_premium")
    if "demographic" in intents or matched_entities["age_buckets"]:
        sources.append("age_bucket_review")
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


def _master_contract_totals(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse network rows into one gross-premium total per master contract."""

    required_columns = {"master_contract", "gross_premium_usd"}
    if frame.empty or not required_columns.issubset(frame.columns):
        return pd.DataFrame(columns=["master_contract", "gross_premium_usd"])
    return (
        frame.groupby("master_contract", as_index=False, dropna=False)["gross_premium_usd"]
        .sum()
        .sort_values("gross_premium_usd", ascending=False, na_position="last")
    )


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
    if source == "master_contract_network_premium":
        result = _filter_rows(frame, "master_contract", matched_entities["master_contracts"])
        if matched_entities["master_contracts"]:
            return result.sort_values(
                ["master_contract", "gross_premium_usd"],
                ascending=[True, False],
                na_position="last",
            )
        top_contracts = _master_contract_totals(result).head(12)["master_contract"]
        return result.loc[result["master_contract"].isin(top_contracts)].sort_values(
            ["master_contract", "gross_premium_usd"],
            ascending=[True, False],
            na_position="last",
        )
    if source == "age_bucket_review":
        result = _filter_rows(frame, "age_bucket", matched_entities["age_buckets"])
        if not matched_entities["age_buckets"]:
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
        "master_contract_network_premium": ("master_contract", "network_type"),
        "age_bucket_review": ("age_bucket",),
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
            if source in {"payer_review", "age_bucket_review"}:
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
            elif source == "monthly_country_kpis":
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
        "age_bucket_review",
        "policy_type_review",
        "mobile_by_payer",
    )
    values: dict[str, list[dict[str, Any]]] = {}
    for table_name in table_names:
        frame = _aggregate_table(snapshot, table_name)
        if not frame.empty:
            values[table_name] = _frame_records(frame)

    payload = {
        "metric_catalog": _METRIC_CATALOG,
        "values": values,
        "metadata": snapshot.get("metadata", {}),
        "dashboard_query_ms": snapshot.get("query_ms"),
    }
    return _json_dumps(payload)


def _focus_label(
    intents: Sequence[str], matched_entities: Mapping[str, tuple[str, ...]]
) -> str:
    if matched_entities["master_contracts"]:
        return "Named master-contract review"
    if "master_contract" in intents:
        return "Master-contract review"
    if matched_entities["age_buckets"]:
        return "Named age-bucket demographic review"
    if "demographic" in intents:
        return "Age-bucket demographic review"
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
    master_contract_values = _entity_values(
        snapshot,
        entity_catalog,
        "contracts",
        "master_contract_network_premium",
        "master_contract",
    )
    age_bucket_values = _entity_values(
        snapshot,
        entity_catalog,
        "age_buckets",
        "age_bucket_review",
        "age_bucket",
    )
    matched_entities = {
        "payers": _match_entities(cleaned_question, payer_values),
        "payer_countries": _match_entities(cleaned_question, country_values),
        "policy_types": _match_entities(cleaned_question, policy_type_values),
        "master_contracts": _match_entities(cleaned_question, master_contract_values),
        "age_buckets": _match_entities(cleaned_question, age_bucket_values),
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
    if (
        matched_entities["master_contracts"]
        and tables.get("master_contract_network_premium", pd.DataFrame()).empty
    ):
        limitations.append(
            "No master-contract network aggregate row was available for the contract named in the question."
        )
    if (
        matched_entities["age_buckets"]
        and tables.get("age_bucket_review", pd.DataFrame()).empty
    ):
        limitations.append("No age-bucket aggregate row was available for the demographic segment named in the question.")
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


# ---------------------------------------------------------------------------
# Deterministic CXO response engine
# ---------------------------------------------------------------------------

_METRIC_LABELS = {
    "gross_premium_usd": "gross premium",
    "net_premium_usd": "net premium",
    "tpa_fee_usd": "TPA fee",
    "app_penetration_rate": "app penetration",
    "active_population": "active population",
    "active_registered_users": "active registered users",
    "unique_beneficiaries": "unique beneficiaries",
    "net_to_gross_ratio": "net-to-gross ratio",
    "tpa_to_gross_ratio": "TPA-to-GP ratio",
}
_RATE_METRICS = {
    "app_penetration_rate",
    "net_to_gross_ratio",
    "tpa_to_gross_ratio",
}
_COUNT_METRICS = {
    "active_population",
    "active_registered_users",
    "unique_beneficiaries",
}


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if pd.notna(number) else None


def _first_number(row: Mapping[str, Any], columns: Sequence[str]) -> float | None:
    for column in columns:
        if column not in row:
            continue
        value = _finite_number(row.get(column))
        if value is not None:
            return value
    return None


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in {None, 0.0}:
        return None
    return numerator / denominator


def _metric_value(row: Mapping[str, Any], metric: str) -> float | None:
    if metric == "app_penetration_rate":
        direct = _first_number(
            row, ("app_penetration_rate", "registered_user_penetration")
        )
        return direct if direct is not None else _ratio(
            _first_number(
                row,
                (
                    "registered_users",
                    "unique_registered_users",
                    "active_registered_users",
                ),
            ),
            _first_number(
                row,
                (
                    "beneficiaries",
                    "unique_beneficiaries",
                    "active_population",
                    "latest_active_population",
                ),
            ),
        )
    if metric == "net_to_gross_ratio":
        direct = _first_number(row, ("net_to_gross_ratio",))
        return direct if direct is not None else _ratio(
            _first_number(row, ("net_premium_usd",)),
            _first_number(row, ("gross_premium_usd",)),
        )
    if metric == "tpa_to_gross_ratio":
        direct = _first_number(row, ("tpa_to_gross_ratio",))
        return direct if direct is not None else _ratio(
            _first_number(row, ("tpa_fee_usd",)),
            _first_number(row, ("gross_premium_usd",)),
        )
    columns = {
        "gross_premium_usd": ("gross_premium_usd",),
        "net_premium_usd": ("net_premium_usd",),
        "tpa_fee_usd": ("tpa_fee_usd",),
        "active_population": ("active_population", "latest_active_population"),
        "active_registered_users": ("active_registered_users", "registered_users"),
        "unique_beneficiaries": ("unique_beneficiaries", "beneficiaries"),
    }
    return _first_number(row, columns.get(metric, ()))


def _primary_metric(evidence: QuestionEvidence) -> str:
    priority = (
        "app_penetration_rate",
        "tpa_to_gross_ratio",
        "net_to_gross_ratio",
        "active_population",
        "active_registered_users",
        "gross_premium_usd",
        "net_premium_usd",
        "tpa_fee_usd",
        "unique_beneficiaries",
    )
    requested = set(evidence.requested_metrics)
    for metric in priority:
        if metric in requested:
            return metric
    if "adoption" in evidence.intents:
        return "app_penetration_rate"
    if "population" in evidence.intents:
        return "active_population"
    return "gross_premium_usd"


def _format_metric(metric: str, value: float) -> str:
    if metric in _RATE_METRICS:
        return _pct(value)
    if metric in _COUNT_METRICS:
        return f"{value:,.0f}"
    return _money(value)


def _metric_gap(metric: str, first: float, second: float) -> str:
    difference = abs(first - second)
    if metric in _RATE_METRICS:
        return f"{difference * 100:.1f} percentage points"
    return _format_metric(metric, difference)


def _latest_entity_rows(frame: pd.DataFrame, entity_column: str) -> pd.DataFrame:
    if frame.empty or "month_end" not in frame or entity_column not in frame:
        return frame
    result = frame.copy()
    result["month_end"] = pd.to_datetime(result["month_end"], errors="coerce")
    return result.sort_values("month_end").groupby(entity_column, as_index=False).tail(1)


def _entity_source(evidence: QuestionEvidence) -> tuple[pd.DataFrame, str] | None:
    if (
        evidence.matched_entities["master_contracts"]
        or "master_contract" in evidence.intents
    ):
        frame = evidence.tables.get("master_contract_network_premium", pd.DataFrame())
        totals = _master_contract_totals(frame)
        if not totals.empty:
            return totals, "master_contract"
    if evidence.matched_entities["age_buckets"] or "demographic" in evidence.intents:
        frame = evidence.tables.get("age_bucket_review", pd.DataFrame())
        if not frame.empty:
            return frame, "age_bucket"
    if evidence.matched_entities["payer_countries"]:
        frame = evidence.tables.get("monthly_country_kpis", pd.DataFrame())
        return (frame, "payer_country") if not frame.empty else None
    if evidence.matched_entities["policy_types"]:
        frame = evidence.tables.get("policy_type_review", pd.DataFrame())
        return (frame, "policy_type") if not frame.empty else None
    if evidence.matched_entities["payers"] or "payer" in evidence.intents:
        frame = evidence.tables.get("payer_review", pd.DataFrame())
        if not frame.empty:
            return frame, "payer_name"
    if "adoption" in evidence.intents:
        frame = evidence.tables.get("mobile_by_payer", pd.DataFrame())
        if not frame.empty:
            return frame, "payer_name"
    return None


def _entity_finding(
    evidence: QuestionEvidence, metric: str
) -> tuple[str | None, str | None, str | None]:
    source = _entity_source(evidence)
    if source is None:
        return None, None, None

    frame, entity_column = source
    frame = _latest_entity_rows(frame, entity_column)
    candidates: list[tuple[str, float]] = []
    for _, row in frame.iterrows():
        if entity_column not in row or pd.isna(row[entity_column]):
            continue
        value = _metric_value(row, metric)
        if value is not None:
            candidates.append((str(row[entity_column]), value))
    if not candidates:
        return None, None, None

    label = _METRIC_LABELS[metric]
    lower_is_better = metric == "tpa_to_gross_ratio"
    if len(candidates) == 1:
        entity, value = candidates[0]
        statement = f"{entity} is at {_format_metric(metric, value)} for {label}."
        return statement, statement, entity if lower_is_better else None

    ordered = sorted(candidates, key=lambda item: item[1])
    low, high = ordered[0], ordered[-1]
    if lower_is_better:
        headline = (
            f"{high[0]} has the highest {label} at {_format_metric(metric, high[1])}, "
            f"{_metric_gap(metric, high[1], low[1])} above {low[0]}."
        )
        evidence_line = (
            f"The peer range runs from {low[0]} at {_format_metric(metric, low[1])} "
            f"to {high[0]} at {_format_metric(metric, high[1])}."
        )
        return headline, evidence_line, high[0]

    if metric in _RATE_METRICS:
        headline = (
            f"{high[0]} leads {low[0]} on {label}: {_format_metric(metric, high[1])} "
            f"versus {_format_metric(metric, low[1])}, a {_metric_gap(metric, high[1], low[1])} gap."
        )
    else:
        headline = (
            f"{high[0]} is largest on {label} at {_format_metric(metric, high[1])}; "
            f"{low[0]} is at {_format_metric(metric, low[1])}."
        )
    evidence_line = (
        f"The observed {label} range is {_format_metric(metric, low[1])} to "
        f"{_format_metric(metric, high[1])} across the selected entities."
    )
    return headline, evidence_line, low[0] if metric in _RATE_METRICS else None


def _time_label(value: Any, source: str) -> str:
    if source == "monthly_kpis":
        timestamp = pd.to_datetime(value, errors="coerce")
        return timestamp.strftime("%b %Y") if pd.notna(timestamp) else str(value)
    return str(int(value)) if _finite_number(value) is not None else str(value)


def _trend_finding(evidence: QuestionEvidence, metric: str) -> str | None:
    candidates = (
        ("monthly_kpis", "month_end", "monthly"),
        ("premium_by_year", "uw_year", "underwriting-year"),
    )
    for source, period_column, cadence in candidates:
        frame = evidence.tables.get(source, pd.DataFrame())
        if frame.empty or period_column not in frame:
            continue
        values: list[tuple[Any, float]] = []
        for _, row in frame.sort_values(period_column).iterrows():
            value = _metric_value(row, metric)
            if value is not None:
                values.append((row[period_column], value))
        if len(values) < 2:
            continue
        previous, latest = values[-2], values[-1]
        if metric in _RATE_METRICS:
            change = f"{(latest[1] - previous[1]) * 100:+.1f} percentage points"
        elif previous[1] == 0:
            change = "from a zero base"
        else:
            change = _pct(latest[1] / previous[1] - 1)
        return (
            f"{cadence.capitalize()} { _METRIC_LABELS[metric] } is "
            f"{_format_metric(metric, latest[1])} in {_time_label(latest[0], source)}, "
            f"{change} versus {_time_label(previous[0], source)}."
        )
    return None


def _scope_summary(evidence: QuestionEvidence) -> str | None:
    frame = evidence.tables.get("summary", pd.DataFrame())
    if frame.empty:
        return None
    row = frame.iloc[0]
    beneficiaries = _metric_value(row, "unique_beneficiaries")
    gross_premium = _metric_value(row, "gross_premium_usd")
    net_premium = _metric_value(row, "net_premium_usd")
    if beneficiaries is None or gross_premium is None:
        return None
    text = (
        f"The applied scope covers {beneficiaries:,.0f} unique beneficiaries and "
        f"{_money(gross_premium)} GP"
    )
    if net_premium is not None:
        text += f" / {_money(net_premium)} NP"
    return text + "."


def _recommended_actions(
    evidence: QuestionEvidence, priority_entity: str | None
) -> list[str]:
    target = priority_entity or "the lowest-performing segment"
    if "adoption" in evidence.intents:
        return [
            f"Set a 90-day registration activation plan for {target}, focused on enrolment and renewal touchpoints.",
            "Track active registered users and app penetration monthly, with a named owner for the conversion gap.",
        ]
    if "economics" in evidence.intents:
        return [
            f"Review pricing, benefit design, and TPA terms for {target} before the next renewal decision.",
            "Use the premium and ratio trend as a monthly value-for-money control, not a one-off year-end review.",
        ]
    if "population" in evidence.intents:
        return [
            "Align service capacity and member communications to the latest active-population trend.",
            "Review the country and payer mix behind material movement before changing the operating plan.",
        ]
    return [
        "Prioritise the largest evidence-backed variance in the next portfolio review.",
        "Assign a payer or policy owner, action date, and monthly outcome metric before the next QBR.",
    ]


def generate_executive_answer(evidence: QuestionEvidence) -> str:
    """Generate an instant, evidence-bound executive response without an external model."""

    metric = _primary_metric(evidence)
    headline, entity_evidence, priority_entity = _entity_finding(evidence, metric)
    trend = _trend_finding(evidence, metric)
    summary = _scope_summary(evidence)
    executive_answer = headline or trend or summary
    if executive_answer is None:
        executive_answer = (
            "The selected scope does not contain enough aggregate evidence to answer this question reliably."
        )

    evidence_lines = [line for line in (entity_evidence, trend, summary) if line]
    if evidence.limitations:
        evidence_lines.append(f"Limitation: {evidence.limitations[0]}")
    if not evidence_lines:
        evidence_lines.append("No relevant aggregate evidence was returned for the selected scope.")

    actions = _recommended_actions(evidence, priority_entity)
    evidence_markdown = "\n".join(f"- {line}" for line in evidence_lines[:4])
    action_markdown = "\n".join(
        f"{index}. {action}" for index, action in enumerate(actions[:2], start=1)
    )
    return (
        f"**Executive answer**\n\n{executive_answer}\n\n"
        f"**Evidence**\n{evidence_markdown}\n\n"
        f"**Recommended actions**\n{action_markdown}"
    )


def record_evaluation(
    *,
    evaluation_dir: str | Path,
    evidence: QuestionEvidence,
    answer: str | None,
    response_engine: str,
    response_status: str,
    configured_model: str | None,
    generation_profile: str | None,
    model_check_status: str | None,
    filter_spec_json: str,
    planning_ms: float | None,
    response_ms: float | None,
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
        "schema_version": 4,
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
        "response_engine": response_engine,
        "response_status": response_status,
        "configured_model": configured_model or "",
        "generation_profile": generation_profile or "",
        "model_check_status": model_check_status or "not_checked",
        "error_message": error_message or "",
        "planning_ms": planning_ms,
        "response_ms": response_ms,
        "dashboard_query_ms": dashboard_query_ms,
    }
    pd.DataFrame([record]).to_parquet(output_path, index=False, compression="zstd")
    return output_path
