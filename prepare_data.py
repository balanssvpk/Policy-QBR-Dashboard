"""Build the local DuckDB mart used by the Policy QBR Streamlit dashboard."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from policy_dashboard.data import build_mart  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Normalize a policy extract and build a high-performance DuckDB mart."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to the source policy CSV/TSV/Parquet file.",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "data" / "policy_mart.duckdb"),
        help="Output DuckDB mart path (default: data/policy_mart.duckdb).",
    )
    args = parser.parse_args()

    try:
        result = build_mart(args.source, args.output)
    except Exception as exc:
        print(f"Mart build failed: {exc}", file=sys.stderr)
        return 1

    print(f"Built {result.mart_path}")
    print(f"Policy records: {result.policy_records:,}")
    print(f"Active member-month rows: {result.active_member_month_rows:,}")
    print(f"Elapsed: {result.elapsed_seconds:,.1f} seconds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
