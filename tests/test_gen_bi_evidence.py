from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from policy_dashboard.gen_bi import build_question_evidence, record_evaluation


def _snapshot() -> dict[str, object]:
    months = pd.to_datetime(["2026-01-31", "2026-02-28"])
    return {
        "summary": pd.DataFrame(
            [
                {
                    "policy_records": 120,
                    "unique_beneficiaries": 100,
                    "master_contracts": 8,
                    "registered_users": 68,
                    "registered_beneficiaries": 66,
                    "gross_premium_usd": 120000.0,
                    "net_premium_usd": 106000.0,
                    "tpa_fee_usd": 6000.0,
                    "latest_active_population": 96,
                    "latest_active_month": months[-1],
                }
            ]
        ),
        "active_population": pd.DataFrame(
            {"month_end": months, "active_population": [92, 96]}
        ),
        "monthly_kpis": pd.DataFrame(
            {
                "month_end": months,
                "active_population": [92, 96],
                "active_registered_users": [60, 68],
                "gross_premium_usd": [58000.0, 62000.0],
                "net_premium_usd": [51000.0, 55000.0],
                "tpa_fee_usd": [2800.0, 3200.0],
            }
        ),
        "monthly_country_kpis": pd.DataFrame(
            {
                "month_end": [months[0], months[0], months[1], months[1]],
                "payer_country": ["Egypt", "United Arab Emirates"] * 2,
                "active_population": [48, 44, 50, 46],
                "active_registered_users": [30, 30, 36, 32],
            }
        ),
        "premium_by_year": pd.DataFrame(
            {
                "uw_year": [2025, 2026],
                "beneficiaries": [98, 100],
                "gross_premium_usd": [110000.0, 120000.0],
                "net_premium_usd": [97000.0, 106000.0],
                "tpa_fee_usd": [5500.0, 6000.0],
            }
        ),
        "payer_review": pd.DataFrame(
            {
                "payer_name": ["Allianz", "Bupa", "Cigna"],
                "beneficiaries": [40, 35, 25],
                "registered_users": [32, 18, 18],
                "registered_beneficiaries": [32, 18, 16],
                "gross_premium_usd": [50000.0, 43000.0, 27000.0],
                "net_premium_usd": [44000.0, 38000.0, 24000.0],
                "tpa_fee_usd": [2500.0, 2200.0, 1300.0],
                "app_penetration_rate": [0.8, 0.5143, 0.72],
                "net_to_gross_ratio": [0.88, 0.8837, 0.8889],
                "tpa_to_gross_ratio": [0.05, 0.0512, 0.0481],
            }
        ),
        "policy_type_review": pd.DataFrame(
            {
                "policy_type": ["Group", "Individual"],
                "beneficiaries": [80, 20],
                "gross_premium_usd": [95000.0, 25000.0],
                "net_premium_usd": [84000.0, 22000.0],
                "tpa_fee_usd": [4800.0, 1200.0],
            }
        ),
        "mobile_by_payer": pd.DataFrame(
            {
                "payer_name": ["Allianz", "Bupa", "Cigna"],
                "unique_beneficiaries": [40, 35, 25],
                "unique_registered_users": [32, 18, 18],
                "linked_beneficiaries": [32, 18, 16],
                "registered_user_penetration": [0.8, 0.5143, 0.72],
                "linked_beneficiary_coverage": [0.8, 0.5143, 0.64],
            }
        ),
        "metadata": {"uw_year_min": "2025", "uw_year_max": "2026"},
        "query_ms": 12.4,
    }


def test_question_evidence_selects_only_named_payer_rows() -> None:
    evidence = build_question_evidence(
        "Compare Allianz and Bupa app penetration and recommend an action.",
        _snapshot(),
        "All portfolio",
        entity_catalog={
            "payer_name": ["Allianz", "Bupa", "Cigna"],
            "payer_country": ["Egypt", "United Arab Emirates"],
            "policy_type": ["Group", "Individual"],
        },
    )

    assert evidence.focus == "Payer digital-adoption comparison"
    assert "adoption" in evidence.intents
    assert evidence.matched_entities["payers"] == ("Allianz", "Bupa")
    assert set(evidence.tables["payer_review"].columns) == {
        "payer_name",
        "beneficiaries",
        "registered_users",
        "app_penetration_rate",
    }
    assert set(evidence.tables["payer_review"]["payer_name"]) == {"Allianz", "Bupa"}
    assert set(evidence.tables["mobile_by_payer"]["payer_name"]) == {"Allianz", "Bupa"}

    context = json.loads(evidence.context_json)
    assert set(context["evidence"]["payer_review"][0]) >= {
        "payer_name",
        "app_penetration_rate",
    }
    assert "Cigna" not in json.dumps(context["evidence"])


def test_record_evaluation_writes_aggregate_parquet_row(tmp_path: Path) -> None:
    snapshot = _snapshot()
    evidence = build_question_evidence(
        "Which policy type has the highest TPA ratio?",
        snapshot,
        "payer country: Egypt",
        entity_catalog={"policy_type": ["Group", "Individual"]},
    )

    output = record_evaluation(
        evaluation_dir=tmp_path / "gen_bi_evaluations",
        evidence=evidence,
        answer="**Executive answer** Group requires attention.",
        model="llama3.2:1b",
        response_status="success",
        filter_spec_json='{"payer_countries":["Egypt"]}',
        planning_ms=1.5,
        model_ms=340.2,
        dashboard_query_ms=12.4,
    )

    assert output.suffix == ".parquet"
    assert output.parent.name.startswith("date=")
    saved = pd.read_parquet(output)
    assert saved.loc[0, "question"] == evidence.question
    assert saved.loc[0, "answer"].startswith("**Executive answer**")
    assert saved.loc[0, "response_status"] == "success"
    assert pd.notna(saved.loc[0, "timestamp_utc"])

    all_metrics = json.loads(saved.loc[0, "all_aggregate_metrics_json"])
    assert "metric_catalog" in all_metrics
    assert "monthly_kpis" in all_metrics["values"]
    assert "summary" in all_metrics["values"]
