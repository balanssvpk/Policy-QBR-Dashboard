from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from generate_demo_data import HEADERS, generate_rows
from policy_dashboard.data import FilterSpec, build_mart, get_filter_options, query_dashboard


def test_demo_data_covers_every_dashboard_dimension(tmp_path: Path) -> None:
    source = tmp_path / "policies_demo.csv"
    mart = tmp_path / "policy_mart.duckdb"
    rows = generate_rows(rows_per_payer=16)
    assert len(rows) == 288

    import csv

    with source.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(rows)

    build_mart(source, mart)
    options = get_filter_options(mart)
    expected_minimum_values = {
        "payer_country": 6,
        "payer_name": 6,
        "service_provider": 6,
        "master_contract": 12,
        "policy_type": 3,
        "policy_type_detail": 3,
        "licensing_authority": 5,
        "network_type": 3,
        "network_group": 3,
        "payer_type": 3,
        "server_name": 6,
        "app_name": 3,
        "dependency": 4,
        "nationality": 10,
        "marital_status": 3,
        "gender": 2,
        "dependent_status": 2,
        "age_profile": 3,
        "age_bucket": 6,
    }
    for dimension, minimum in expected_minimum_values.items():
        assert len(options[dimension]) >= minimum, dimension

    scoped = query_dashboard(
        mart,
        FilterSpec(
            year_start=2025,
            year_end=2026,
            payers=("Bupa",),
            age_buckets=("30-39",),
            network_groups=("Premium",),
        ),
    )
    assert scoped["summary"].iloc[0]["policy_records"] > 0
    assert not scoped["active_population"].empty
    assert not scoped["monthly_country_kpis"].empty
