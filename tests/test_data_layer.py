from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from policy_dashboard.data import FilterSpec, build_mart, get_filter_options, query_dashboard


SAMPLE = """index,servername,serviceprovidername,payercountry,payer_name,payer_key,mastercontract,policy_type,licensing_authority,op_member_network_type,dependency,nationality,maritalstatus,gross_premium_usd,net_premium_usd,tpa_usd,gender,isdependent,age_profile,uwyear,payer_type,policytype,networkgroup,rn1,member_start_date,member_stop_date,beneficiarykey,beneficiarypayerkey,mastercontractkey,registereduserkey,registrationdate,registrationyear,appname,age_bucket,reload_ts
11303252,EGP,NEXtCARE Egypt,Egypt,Allianz,4 Jan,CIB Families,Group,Others,GN,Principal,Egypt,Married,464.7,433.1,24.6,Female,Principal,Adult,2026,Insured,Group,NonPCP,1,1 01 2026,1 01 2027,4-999999,1-999999,4-78232,,,None,30-39,20260308115210
11303253,EGP,NEXtCARE Egypt,Egypt,Allianz,4 Jan,CIB Families,Group,Others,GN,Principal,Egypt,Married,372.4,345,19.7,Female,Principal,Adult,2025,Insured,Group,NonPCP,2,1 01 2025,1 01 2026,4-999998,1-999998,4-68162,3467382,23 03 2024,2024,Lumi,30-39,20260308115210
11303254,EGP,NEXtCARE Egypt,Egypt,Allianz,4 Jan,CIB Families,Group,MOH,GN,Principal,Egypt,Married,287.8,263.9,15.3,Female,Principal,Adult,2024,Insured,Group,NonPCP,3,1 01 2024,1 01 2025,4-999997,1-999997,4-56692,3467382,23 03 2024,2024,Lumi,30-39,20260308115210
"""


def test_mart_build_and_dashboard_queries(tmp_path: Path) -> None:
    source = tmp_path / "policies.csv"
    mart = tmp_path / "policy_mart.duckdb"
    source.write_text(SAMPLE, encoding="utf-8")

    result = build_mart(source, mart)
    assert result.policy_records == 3
    assert result.active_member_month_rows == 36

    options = get_filter_options(mart)
    assert options["payer_name"] == ["Allianz"]
    snapshot = query_dashboard(mart, FilterSpec(year_start=2024, year_end=2026))
    summary = snapshot["summary"].iloc[0]
    assert summary["unique_beneficiaries"] == 3
    assert summary["registered_users"] == 1
    assert summary["gross_premium_usd"] == pytest.approx(1124.9)
    assert len(snapshot["active_population"]) == 36
    monthly_kpis = snapshot["monthly_kpis"]
    assert len(monthly_kpis) == 36
    assert set(monthly_kpis.columns) == {
        "month_end",
        "active_population",
        "active_registered_users",
        "gross_premium_usd",
        "net_premium_usd",
        "tpa_fee_usd",
    }
    assert monthly_kpis["gross_premium_usd"].sum() == pytest.approx(
        summary["gross_premium_usd"]
    )
    monthly_country_kpis = snapshot["monthly_country_kpis"]
    assert len(monthly_country_kpis) == 36
    assert set(monthly_country_kpis.columns) == {
        "month_end",
        "payer_country",
        "active_population",
        "active_registered_users",
    }
    assert set(monthly_country_kpis["payer_country"]) == {"Egypt"}
    assert monthly_country_kpis["active_population"].tolist() == monthly_kpis[
        "active_population"
    ].tolist()


def test_loader_sniffs_tab_delimited_csv_content(tmp_path: Path) -> None:
    source = tmp_path / "policies.csv"
    mart = tmp_path / "policy_mart.duckdb"
    source.write_text(SAMPLE.replace(",", "\t"), encoding="utf-8")

    result = build_mart(source, mart)
    assert result.policy_records == 3
    assert result.active_member_month_rows == 36
