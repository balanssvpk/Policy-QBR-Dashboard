"""Generate a deterministic, all-dimension insurance policy demo extract.

The resulting CSV is safe synthetic data. It is deliberately varied so every
dashboard filter has multiple values and the month-end population, adoption,
and premium views show meaningful movement.
"""

from __future__ import annotations

import argparse
import csv
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parent
HEADERS = [
    "index",
    "servername",
    "serviceprovidername",
    "payercountry",
    "payer_name",
    "payer_key",
    "mastercontract",
    "policy_type",
    "licensing_authority",
    "op_member_network_type",
    "dependency",
    "nationality",
    "maritalstatus",
    "gross_premium_usd",
    "net_premium_usd",
    "tpa_usd",
    "gender",
    "isdependent",
    "age_profile",
    "uwyear",
    "payer_type",
    "policytype",
    "networkgroup",
    "rn1",
    "member_start_date",
    "member_stop_date",
    "beneficiarykey",
    "beneficiarypayerkey",
    "mastercontractkey",
    "registereduserkey",
    "registrationdate",
    "registrationyear",
    "appname",
    "age_bucket",
    "reload_ts",
]

PAYERS = [
    {
        "country": "Egypt",
        "payer": "Allianz",
        "payer_key": "EG-ALL",
        "server": "EGP",
        "provider": "NEXtCARE Egypt",
        "authority": "MOH",
        "network_type": "GN",
        "contracts": ("CIB Families", "Nile Manufacturing"),
    },
    {
        "country": "United Arab Emirates",
        "payer": "Bupa",
        "payer_key": "AE-BUP",
        "server": "UAE",
        "provider": "NEXtCARE UAE",
        "authority": "DHA",
        "network_type": "GN Plus",
        "contracts": ("Atlas Aviation", "Dubai Retail Group"),
    },
    {
        "country": "Saudi Arabia",
        "payer": "AXA",
        "payer_key": "SA-AXA",
        "server": "KSA",
        "provider": "MedNet Saudi",
        "authority": "CCHI",
        "network_type": "Regional",
        "contracts": ("Riyadh Technology", "Jeddah Logistics"),
    },
    {
        "country": "Qatar",
        "payer": "MetLife",
        "payer_key": "QA-MET",
        "server": "QAT",
        "provider": "QLM Health Services",
        "authority": "QCHP",
        "network_type": "GN",
        "contracts": ("Doha Energy", "Qatar Education Trust"),
    },
    {
        "country": "Oman",
        "payer": "GIG",
        "payer_key": "OM-GIG",
        "server": "OMN",
        "provider": "NEXtCARE Oman",
        "authority": "MOH",
        "network_type": "Regional",
        "contracts": ("Muscat Hospitality", "Sohar Industrial"),
    },
    {
        "country": "Bahrain",
        "payer": "Cigna",
        "payer_key": "BH-CIG",
        "server": "BHR",
        "provider": "Tawuniya Health",
        "authority": "NHRA",
        "network_type": "GN Plus",
        "contracts": ("Manama Finance", "Bahrain Commerce"),
    },
]

MEMBER_PROFILES = [
    ("Principal", "Principal", "Female", "Adult", "30-39", "Egypt", "Married"),
    ("Principal", "Principal", "Male", "Adult", "40-49", "India", "Married"),
    ("Spouse", "Dependent", "Female", "Adult", "30-39", "Philippines", "Married"),
    ("Spouse", "Dependent", "Male", "Adult", "40-49", "Jordan", "Married"),
    ("Child", "Dependent", "Female", "Child", "0-17", "Egypt", "Single"),
    ("Child", "Dependent", "Male", "Child", "0-17", "Pakistan", "Single"),
    ("Principal", "Principal", "Female", "Adult", "50-59", "Lebanon", "Single"),
    ("Principal", "Principal", "Male", "Senior", "60+", "United Kingdom", "Married"),
    ("Parent", "Dependent", "Female", "Senior", "60+", "Sudan", "Widowed"),
    ("Parent", "Dependent", "Male", "Senior", "60+", "Syria", "Married"),
    ("Principal", "Principal", "Female", "Adult", "18-29", "United Arab Emirates", "Single"),
    ("Principal", "Principal", "Male", "Adult", "18-29", "Saudi Arabia", "Single"),
    ("Spouse", "Dependent", "Female", "Adult", "50-59", "Kenya", "Married"),
    ("Child", "Dependent", "Male", "Child", "0-17", "Nigeria", "Single"),
    ("Principal", "Principal", "Female", "Adult", "40-49", "France", "Married"),
    ("Principal", "Principal", "Male", "Adult", "30-39", "Germany", "Married"),
]

POLICY_PRODUCTS = [
    ("Group", "Corporate"),
    ("Group", "Family"),
    ("Individual", "Retail"),
    ("SME", "Corporate"),
]
NETWORK_GROUPS = ("NonPCP", "PCP", "Premium")
PAYER_TYPES = ("Insured", "Sponsor", "Self-funded")
APP_NAMES = ("Lumi", "MyHealth", "CareHub")


def _format_date(value: date | None) -> str:
    return value.strftime("%d %m %Y") if value else ""


def generate_rows(rows_per_payer: int) -> list[dict[str, str | int | float]]:
    if rows_per_payer < 1 or rows_per_payer > len(MEMBER_PROFILES):
        raise ValueError(f"rows-per-payer must be between 1 and {len(MEMBER_PROFILES)}")

    rows: list[dict[str, str | int | float]] = []
    record_index = 11310000
    for year in (2024, 2025, 2026):
        for payer_index, payer in enumerate(PAYERS):
            for member_index, profile in enumerate(MEMBER_PROFILES[:rows_per_payer]):
                dependency, dependent_status, gender, age_profile, age_bucket, nationality, marital = profile
                contract_index = (member_index + year) % len(payer["contracts"])
                master_contract = payer["contracts"][contract_index]
                policy_type, policy_type_detail = POLICY_PRODUCTS[(member_index + payer_index) % len(POLICY_PRODUCTS)]
                network_group = NETWORK_GROUPS[(member_index + year + payer_index) % len(NETWORK_GROUPS)]
                start_month = (member_index * 2 + payer_index + year) % 10 + 1
                member_start = date(year, start_month, 1)
                if member_index % 11 == 0:
                    member_stop = date(year, 6, 30)
                elif member_index % 7 == 0:
                    member_stop = date(year, 9, 30)
                else:
                    member_stop = date(year, 12, 31)

                gross = round((285 + payer_index * 68 + member_index * 19) * (1 + (year - 2024) * 0.06), 2)
                net_ratio = 0.87 + ((member_index + payer_index) % 4) * 0.015
                net = round(gross * net_ratio, 2)
                tpa = round(gross * (0.035 + ((member_index + payer_index) % 3) * 0.006), 2)
                is_registered = (member_index + payer_index + year) % 5 != 0
                registered_user = (
                    f"REG-{payer_index + 1:02d}-{member_index // 2 + 1:03d}"
                    if is_registered
                    else ""
                )
                registration_date = date(year, max(1, start_month - 1), 15) if is_registered else None
                app_name = APP_NAMES[(member_index + payer_index) % len(APP_NAMES)] if is_registered else "None"
                beneficiary_key = f"BEN-{payer_index + 1:02d}-{member_index + 1:03d}"
                row = {
                    "index": record_index,
                    "servername": payer["server"],
                    "serviceprovidername": payer["provider"],
                    "payercountry": payer["country"],
                    "payer_name": payer["payer"],
                    "payer_key": payer["payer_key"],
                    "mastercontract": master_contract,
                    "policy_type": policy_type,
                    "licensing_authority": payer["authority"],
                    "op_member_network_type": payer["network_type"],
                    "dependency": dependency,
                    "nationality": nationality,
                    "maritalstatus": marital,
                    "gross_premium_usd": gross,
                    "net_premium_usd": net,
                    "tpa_usd": tpa,
                    "gender": gender,
                    "isdependent": dependent_status,
                    "age_profile": age_profile,
                    "uwyear": year,
                    "payer_type": PAYER_TYPES[(payer_index + member_index) % len(PAYER_TYPES)],
                    "policytype": policy_type_detail,
                    "networkgroup": network_group,
                    "rn1": member_index + 1,
                    "member_start_date": _format_date(member_start),
                    "member_stop_date": _format_date(member_stop),
                    "beneficiarykey": beneficiary_key,
                    "beneficiarypayerkey": f"BP-{payer_index + 1:02d}-{member_index + 1:03d}",
                    "mastercontractkey": f"MC-{payer_index + 1:02d}-{contract_index + 1:02d}",
                    "registereduserkey": registered_user,
                    "registrationdate": _format_date(registration_date),
                    "registrationyear": registration_date.year if registration_date else "",
                    "appname": app_name,
                    "age_bucket": age_bucket,
                    "reload_ts": f"{year + 1}0308115210",
                }
                rows.append(row)
                record_index += 1
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a rich synthetic policy demo extract.")
    parser.add_argument(
        "--output",
        default=str(ROOT / "data" / "policies_demo.csv"),
        help="CSV output path (default: data/policies_demo.csv).",
    )
    parser.add_argument(
        "--rows-per-payer",
        type=int,
        default=16,
        help="Distinct member profiles per payer/year (1–16; default: 16).",
    )
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = generate_rows(args.rows_per_payer)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows):,} synthetic policy-year records to {output}")


if __name__ == "__main__":
    main()
