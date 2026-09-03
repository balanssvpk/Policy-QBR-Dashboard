"""DuckDB mart construction and read-only analytics for the Policy QBR app.

The dashboard never loads the full policy extract into Pandas. The one-time
builder writes an analytics mart; interactive queries then execute inside
DuckDB and return only small aggregated result sets to Streamlit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import re
import time
from typing import Any, Iterable

import duckdb
import pandas as pd


MISSING_TEXT = ("", "<na>", "na", "nan", "none", "null", "nat")

# Alias lists are normalized with _normalize_name before lookup. The first
# alias in each list documents the expected source field from the supplied
# sample extract.
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "policy_record_id": ("index", "policy_record_id", "policy_id", "record_id"),
    "server_name": ("servername", "server_name"),
    "service_provider": (
        "serviceprovidername",
        "service_provider_name",
        "service_provider",
    ),
    "payer_country": ("payercountry", "payer_country"),
    "payer_name": ("payer_name", "payername", "payer"),
    "payer_key": ("payer_key", "payerkey"),
    "master_contract": ("mastercontract", "master_contract"),
    "master_contract_key": ("mastercontractkey", "master_contract_key"),
    "policy_type": ("policy_type", "policytype_main"),
    "policy_type_detail": ("policytype", "policy_type_detail"),
    "licensing_authority": ("licensing_authority", "licensingauthority"),
    "network_type": ("op_member_network_type", "network_type"),
    "network_group": ("networkgroup", "network_group"),
    "dependency": ("dependency",),
    "nationality": ("nationality",),
    "marital_status": ("maritalstatus", "marital_status"),
    "gender": ("gender",),
    "dependent_status": ("isdependent", "is_dependent", "dependent_status"),
    "age_profile": ("age_profile", "ageprofile"),
    "age_bucket": ("age_bucket", "agebucket"),
    "payer_type": ("payer_type", "payertype"),
    "beneficiary_key": ("beneficiarykey", "beneficiary_key"),
    "beneficiary_payer_key": ("beneficiarypayerkey", "beneficiary_payer_key"),
    "registered_user_key": ("registered_user_key", "registereduserkey"),
    "app_name": ("appname", "app_name"),
    "gross_premium_usd": ("gross_premium_usd", "grosspremiumusd"),
    "net_premium_usd": ("net_premium_usd", "netpremiumusd"),
    "tpa_fee_usd": ("tpa_usd", "tpa_fee_usd", "tpafeeusd"),
    "uw_year": ("uwyear", "uw_year", "underwriting_year"),
    "member_start_date": ("member_start_date", "memberstartdate"),
    "member_stop_date": ("member_stop_date", "memberstopdate"),
    "registration_date": ("registrationdate", "registration_date"),
    "registration_year": ("registrationyear", "registration_year"),
    "reload_timestamp": ("reload_ts", "reloadtimestamp", "reload_timestamp"),
    "rn1": ("rn1",),
}

TEXT_COLUMNS = (
    "policy_record_id",
    "server_name",
    "service_provider",
    "payer_country",
    "payer_name",
    "payer_key",
    "master_contract",
    "master_contract_key",
    "policy_type",
    "policy_type_detail",
    "licensing_authority",
    "network_type",
    "network_group",
    "dependency",
    "nationality",
    "marital_status",
    "gender",
    "dependent_status",
    "age_profile",
    "age_bucket",
    "payer_type",
    "beneficiary_key",
    "beneficiary_payer_key",
    "registered_user_key",
    "app_name",
)
NUMBER_COLUMNS = ("gross_premium_usd", "net_premium_usd", "tpa_fee_usd")
INTEGER_COLUMNS = ("uw_year", "registration_year", "rn1")
DATE_COLUMNS = ("member_start_date", "member_stop_date", "registration_date")

# These fields are deliberately limited to the fields copied into the active
# membership table. This makes the filter SQL static and safe.
FILTER_COLUMNS: dict[str, str] = {
    "payer_countries": "payer_country",
    "payers": "payer_name",
    "providers": "service_provider",
    "contracts": "master_contract",
    "policy_types": "policy_type",
    "policy_type_details": "policy_type_detail",
    "licensing_authorities": "licensing_authority",
    "network_types": "network_type",
    "network_groups": "network_group",
    "payer_types": "payer_type",
    "server_names": "server_name",
    "app_names": "app_name",
    "dependencies": "dependency",
    "nationalities": "nationality",
    "marital_statuses": "marital_status",
    "gender": "gender",
    "dependent_statuses": "dependent_status",
    "age_profiles": "age_profile",
    "age_buckets": "age_bucket",
}
FILTER_OPTION_COLUMNS = ("uw_year", *FILTER_COLUMNS.values())


@dataclass(frozen=True)
class FilterSpec:
    """An immutable, cache-friendly dashboard selection."""

    year_start: int
    year_end: int
    payer_countries: tuple[str, ...] = field(default_factory=tuple)
    payers: tuple[str, ...] = field(default_factory=tuple)
    providers: tuple[str, ...] = field(default_factory=tuple)
    contracts: tuple[str, ...] = field(default_factory=tuple)
    policy_types: tuple[str, ...] = field(default_factory=tuple)
    policy_type_details: tuple[str, ...] = field(default_factory=tuple)
    licensing_authorities: tuple[str, ...] = field(default_factory=tuple)
    network_types: tuple[str, ...] = field(default_factory=tuple)
    network_groups: tuple[str, ...] = field(default_factory=tuple)
    payer_types: tuple[str, ...] = field(default_factory=tuple)
    server_names: tuple[str, ...] = field(default_factory=tuple)
    app_names: tuple[str, ...] = field(default_factory=tuple)
    dependencies: tuple[str, ...] = field(default_factory=tuple)
    nationalities: tuple[str, ...] = field(default_factory=tuple)
    marital_statuses: tuple[str, ...] = field(default_factory=tuple)
    gender: tuple[str, ...] = field(default_factory=tuple)
    dependent_statuses: tuple[str, ...] = field(default_factory=tuple)
    age_profiles: tuple[str, ...] = field(default_factory=tuple)
    age_buckets: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "year_start": self.year_start,
            "year_end": self.year_end,
            **{attribute: list(getattr(self, attribute)) for attribute in FILTER_COLUMNS},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FilterSpec":
        values: dict[str, Any] = {
            "year_start": int(payload["year_start"]),
            "year_end": int(payload["year_end"]),
        }
        for attribute in FILTER_COLUMNS:
            values[attribute] = tuple(
                str(value)
                for value in payload.get(attribute, [])
                if value is not None and str(value).strip()
            )
        return cls(**values)

    @property
    def has_dimension_filters(self) -> bool:
        return any(getattr(self, attribute) for attribute in FILTER_COLUMNS)


@dataclass(frozen=True)
class BuildResult:
    source_path: Path
    mart_path: Path
    policy_records: int
    active_member_month_rows: int
    elapsed_seconds: float


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _source_expression(source_columns: dict[str, str], target_column: str) -> str:
    for alias in COLUMN_ALIASES[target_column]:
        actual_name = source_columns.get(_normalize_name(alias))
        if actual_name:
            return _quote_identifier(actual_name)
    return "NULL"


def _clean_text(expression: str) -> str:
    raw = f"TRIM(CAST({expression} AS VARCHAR))"
    missing = ", ".join(f"'{token}'" for token in MISSING_TEXT)
    return (
        f"CASE WHEN {expression} IS NULL OR LOWER({raw}) IN ({missing}) "
        f"THEN NULL ELSE {raw} END"
    )


def _number_expression(expression: str) -> str:
    text = _clean_text(expression)
    sanitized = (
        f"REPLACE(REPLACE(REPLACE(REPLACE({text}, ',', ''), '$', ''), "
        f"'USD', ''), ' ', '')"
    )
    return f"TRY_CAST(NULLIF({sanitized}, '') AS DOUBLE)"


def _integer_expression(expression: str) -> str:
    return f"TRY_CAST({_number_expression(expression)} AS INTEGER)"


def _date_expression(expression: str) -> str:
    text = _clean_text(expression)
    return f"""COALESCE(
        TRY_CAST({text} AS DATE),
        CAST(TRY_STRPTIME({text}, '%d %m %Y') AS DATE),
        CAST(TRY_STRPTIME({text}, '%Y-%m-%d') AS DATE),
        CAST(TRY_STRPTIME({text}, '%d/%m/%Y') AS DATE)
    )"""


def _timestamp_expression(expression: str) -> str:
    text = _clean_text(expression)
    return f"""COALESCE(
        TRY_CAST({text} AS TIMESTAMP),
        TRY_STRPTIME({text}, '%Y%m%d%H%M%S'),
        TRY_STRPTIME({text}, '%Y-%m-%d %H:%M:%S')
    )"""


def _reader_sql(source_path: Path) -> str:
    suffix = source_path.suffix.lower()
    if suffix in {".csv", ".tsv", ".txt"}:
        # Be explicit after sniffing. DuckDB's auto-detect can collapse a wide
        # extract with irregular trailing fields into a single column.
        with source_path.open("r", encoding="utf-8-sig", errors="replace") as file:
            header = next((line for line in file if line.strip()), "")
        fallback = "\t" if suffix == ".tsv" else ","
        candidates = {delimiter: header.count(delimiter) for delimiter in (",", "\t", ";", "|")}
        delimiter = max(candidates, key=candidates.get) if header else fallback
        if delimiter not in {",", "\t", ";", "|"}:
            delimiter = fallback
        return (
            f"read_csv(?, delim='{delimiter}', header=true, all_varchar=true, "
            "null_padding=true, strict_mode=false)"
        )
    if suffix in {".parquet", ".pq"}:
        return "read_parquet(?)"
    raise ValueError("Unsupported source type. Use a CSV, TSV, TXT, Parquet, or PQ extract.")


def _validate_source_columns(source_columns: dict[str, str]) -> None:
    essential = (
        "beneficiary_key",
        "gross_premium_usd",
        "net_premium_usd",
        "tpa_fee_usd",
        "uw_year",
        "member_start_date",
        "member_stop_date",
    )
    missing = [
        column
        for column in essential
        if not any(_normalize_name(alias) in source_columns for alias in COLUMN_ALIASES[column])
    ]
    if missing:
        raise ValueError(
            "The source is missing required policy fields: "
            f"{', '.join(missing)}. Check the header or update COLUMN_ALIASES."
        )


def _typed_select_list(source_columns: dict[str, str]) -> list[str]:
    select_list: list[str] = []
    for column in TEXT_COLUMNS:
        select_list.append(
            f"{_clean_text(_source_expression(source_columns, column))} AS {column}"
        )
    for column in NUMBER_COLUMNS:
        select_list.append(
            f"{_number_expression(_source_expression(source_columns, column))} AS {column}"
        )
    for column in INTEGER_COLUMNS:
        select_list.append(
            f"{_integer_expression(_source_expression(source_columns, column))} AS {column}"
        )
    for column in DATE_COLUMNS:
        select_list.append(
            f"{_date_expression(_source_expression(source_columns, column))} AS {column}"
        )
    select_list.append(
        f"{_timestamp_expression(_source_expression(source_columns, 'reload_timestamp'))} "
        "AS reload_timestamp"
    )
    return select_list


def build_mart(source_path: str | Path, mart_path: str | Path) -> BuildResult:
    """Create or replace a compact, query-ready DuckDB policy mart.

    This is intentionally a one-time batch process. It normalizes source data,
    derives policy coverage dates, expands policies to month-end memberships,
    and creates an unfiltered fast path for the main population chart.
    """

    source = Path(source_path).expanduser().resolve()
    mart = Path(mart_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Policy source not found: {source}")
    mart.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    connection = duckdb.connect(str(mart))
    try:
        connection.execute("DROP TABLE IF EXISTS raw_policy_source")
        connection.execute("DROP TABLE IF EXISTS monthly_kpis_default")
        connection.execute("DROP TABLE IF EXISTS monthly_country_kpis_default")
        connection.execute("DROP TABLE IF EXISTS monthly_dimension_population_default")
        connection.execute("DROP TABLE IF EXISTS active_population_default")
        connection.execute("DROP TABLE IF EXISTS active_member_months")
        connection.execute("DROP TABLE IF EXISTS filter_options")
        connection.execute("DROP TABLE IF EXISTS policy_records")
        connection.execute("DROP TABLE IF EXISTS mart_metadata")

        connection.execute(
            f"CREATE TABLE raw_policy_source AS SELECT * FROM {_reader_sql(source)}",
            [str(source)],
        )
        source_columns = {
            _normalize_name(name): name
            for _, name, *_ in connection.execute(
                "PRAGMA table_info('raw_policy_source')"
            ).fetchall()
        }
        _validate_source_columns(source_columns)
        typed_select = ",\n                    ".join(_typed_select_list(source_columns))
        connection.execute(
            f"""
            CREATE TABLE policy_records AS
            WITH typed AS (
                SELECT
                    {typed_select}
                FROM raw_policy_source
            ), dated AS (
                SELECT
                    *,
                    COALESCE(
                        member_start_date,
                        CASE WHEN uw_year BETWEEN 1900 AND 2100
                             THEN MAKE_DATE(uw_year, 1, 1) END
                    ) AS effective_start_date,
                    COALESCE(
                        member_stop_date,
                        CASE WHEN uw_year BETWEEN 1900 AND 2100
                             THEN MAKE_DATE(uw_year, 12, 31) END
                    ) AS provisional_stop_date
                FROM typed
            )
            SELECT
                *,
                CASE
                    WHEN provisional_stop_date < effective_start_date
                    THEN effective_start_date
                    ELSE provisional_stop_date
                END AS effective_stop_date
            FROM dated
            """
        )
        active_dimensions = ",\n                    ".join(FILTER_COLUMNS.values())
        connection.execute(
            f"""
            CREATE TABLE active_member_months AS
            WITH active_base AS (
                SELECT
                    ROW_NUMBER() OVER () AS policy_row_id,
                    uw_year,
                    {active_dimensions},
                    beneficiary_key,
                    registered_user_key,
                    registration_date,
                    gross_premium_usd,
                    net_premium_usd,
                    tpa_fee_usd,
                    effective_stop_date,
                    DATE_TRUNC('month', effective_start_date)::DATE AS start_month,
                    DATE_TRUNC('month', effective_stop_date)::DATE AS stop_month
                FROM policy_records
                WHERE effective_start_date IS NOT NULL
                  AND effective_stop_date IS NOT NULL
            ), expanded AS (
                SELECT
                    policy_row_id,
                    LAST_DAY(month_series.month_start)::DATE AS month_end,
                    uw_year,
                    {active_dimensions},
                    beneficiary_key,
                    registered_user_key,
                    registration_date,
                    gross_premium_usd,
                    net_premium_usd,
                    tpa_fee_usd
                FROM active_base
                CROSS JOIN GENERATE_SERIES(
                    start_month,
                    stop_month,
                    INTERVAL '1 month'
                ) AS month_series(month_start)
                WHERE LAST_DAY(month_series.month_start)::DATE <= effective_stop_date
            ), apportioned AS (
                SELECT
                    *,
                    COUNT(*) OVER (PARTITION BY policy_row_id) AS active_month_count
                FROM expanded
            )
            SELECT
                month_end,
                uw_year,
                {active_dimensions},
                beneficiary_key,
                registered_user_key,
                registration_date,
                gross_premium_usd / NULLIF(active_month_count, 0) AS gross_premium_usd,
                net_premium_usd / NULLIF(active_month_count, 0) AS net_premium_usd,
                tpa_fee_usd / NULLIF(active_month_count, 0) AS tpa_fee_usd
            FROM apportioned
            """
        )
        connection.execute(
            """
            CREATE TABLE monthly_kpis_default AS
            SELECT
                month_end,
                COUNT(DISTINCT beneficiary_key) AS active_population,
                COUNT(DISTINCT CASE
                    WHEN beneficiary_key IS NOT NULL
                     AND registered_user_key IS NOT NULL
                     AND (registration_date IS NULL OR registration_date <= month_end)
                    THEN registered_user_key
                END) AS active_registered_users,
                COALESCE(SUM(gross_premium_usd), 0) AS gross_premium_usd,
                COALESCE(SUM(net_premium_usd), 0) AS net_premium_usd,
                COALESCE(SUM(tpa_fee_usd), 0) AS tpa_fee_usd
            FROM active_member_months
            GROUP BY month_end
            ORDER BY month_end
            """
        )
        connection.execute(
            """
            CREATE TABLE active_population_default AS
            SELECT month_end, active_population
            FROM monthly_kpis_default
            ORDER BY month_end
            """
        )
        connection.execute(
            """
            CREATE TABLE monthly_country_kpis_default AS
            SELECT
                month_end,
                COALESCE(payer_country, 'Unassigned') AS payer_country,
                COUNT(DISTINCT beneficiary_key) AS active_population,
                COUNT(DISTINCT CASE
                    WHEN beneficiary_key IS NOT NULL
                     AND registered_user_key IS NOT NULL
                     AND (registration_date IS NULL OR registration_date <= month_end)
                    THEN registered_user_key
                END) AS active_registered_users,
                COALESCE(SUM(gross_premium_usd), 0) AS gross_premium_usd,
                COALESCE(SUM(net_premium_usd), 0) AS net_premium_usd,
                COALESCE(SUM(tpa_fee_usd), 0) AS tpa_fee_usd
            FROM active_member_months
            GROUP BY month_end, COALESCE(payer_country, 'Unassigned')
            ORDER BY month_end, payer_country
            """
        )
        connection.execute(
            """
            CREATE TABLE monthly_dimension_population_default AS
            WITH grouped AS (
                SELECT
                    month_end,
                    network_type,
                    network_group,
                    policy_type,
                    GROUPING(network_type) AS network_type_grouped,
                    GROUPING(network_group) AS network_group_grouped,
                    GROUPING(policy_type) AS policy_type_grouped,
                    COUNT(DISTINCT beneficiary_key) AS active_population,
                    COUNT(DISTINCT CASE
                        WHEN beneficiary_key IS NOT NULL
                         AND registered_user_key IS NOT NULL
                         AND (registration_date IS NULL OR registration_date <= month_end)
                        THEN registered_user_key
                    END) AS active_registered_users,
                    COALESCE(SUM(gross_premium_usd), 0) AS gross_premium_usd,
                    COALESCE(SUM(net_premium_usd), 0) AS net_premium_usd,
                    COALESCE(SUM(tpa_fee_usd), 0) AS tpa_fee_usd
                FROM active_member_months
                GROUP BY GROUPING SETS (
                    (month_end, network_type),
                    (month_end, network_group),
                    (month_end, policy_type)
                )
            )
            SELECT
                month_end,
                CASE
                    WHEN network_type_grouped = 0 THEN 'network_type'
                    WHEN network_group_grouped = 0 THEN 'network_group'
                    ELSE 'policy_type'
                END AS dimension,
                CASE
                    WHEN network_type_grouped = 0 THEN COALESCE(network_type, 'Unassigned')
                    WHEN network_group_grouped = 0 THEN COALESCE(network_group, 'Unassigned')
                    ELSE COALESCE(policy_type, 'Unassigned')
                END AS dimension_value,
                active_population,
                active_registered_users,
                gross_premium_usd,
                net_premium_usd,
                tpa_fee_usd
            FROM grouped
            ORDER BY month_end, dimension, dimension_value
            """
        )
        filter_option_unions = "\nUNION ALL\n".join(
            f"""
            SELECT '{column}' AS dimension, CAST({column} AS VARCHAR) AS value
            FROM policy_records
            WHERE {column} IS NOT NULL
            """
            for column in FILTER_OPTION_COLUMNS
        )
        connection.execute(
            f"""
            CREATE TABLE filter_options AS
            SELECT dimension, value
            FROM ({filter_option_unions})
            GROUP BY dimension, value
            """
        )
        connection.execute(
            """
            CREATE TABLE mart_metadata (
                key VARCHAR PRIMARY KEY,
                value VARCHAR
            )
            """
        )
        policy_count = int(connection.execute("SELECT COUNT(*) FROM policy_records").fetchone()[0])
        active_count = int(connection.execute("SELECT COUNT(*) FROM active_member_months").fetchone()[0])
        min_year, max_year = connection.execute(
            "SELECT MIN(uw_year), MAX(uw_year) FROM policy_records"
        ).fetchone()
        metadata_rows = [
            ("source_path", str(source)),
            ("built_at_utc", datetime.now(timezone.utc).isoformat(timespec="seconds")),
            ("policy_records", str(policy_count)),
            ("active_member_month_rows", str(active_count)),
            ("uw_year_min", "" if min_year is None else str(min_year)),
            ("uw_year_max", "" if max_year is None else str(max_year)),
        ]
        connection.executemany(
            "INSERT INTO mart_metadata (key, value) VALUES (?, ?)", metadata_rows
        )
        # The raw copy would duplicate the source extract without helping the dashboard.
        connection.execute("DROP TABLE raw_policy_source")
        connection.execute("CHECKPOINT")
    finally:
        connection.close()

    return BuildResult(
        source_path=source,
        mart_path=mart,
        policy_records=policy_count,
        active_member_month_rows=active_count,
        elapsed_seconds=time.perf_counter() - started,
    )


def _connect_readonly(mart_path: str | Path) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(Path(mart_path).resolve()), read_only=True)


def get_mart_metadata(mart_path: str | Path) -> dict[str, str]:
    with _connect_readonly(mart_path) as connection:
        rows = connection.execute("SELECT key, value FROM mart_metadata").fetchall()
    return {str(key): str(value) for key, value in rows}


def get_filter_options(mart_path: str | Path) -> dict[str, list[Any]]:
    """Return small dimension tables for the sidebar. Cache this at the UI layer."""

    options: dict[str, list[Any]] = {column: [] for column in FILTER_OPTION_COLUMNS}
    with _connect_readonly(mart_path) as connection:
        rows = connection.execute(
            "SELECT dimension, value FROM filter_options ORDER BY dimension, value"
        ).fetchall()
    for dimension, value in rows:
        if dimension == "uw_year":
            options[dimension].append(int(value))
        else:
            options[dimension].append(value)
    return options


def _where_clause(filters: FilterSpec, table_alias: str = "") -> tuple[str, list[Any]]:
    prefix = f"{table_alias}." if table_alias else ""
    clauses = [f"{prefix}uw_year BETWEEN ? AND ?"]
    parameters: list[Any] = [filters.year_start, filters.year_end]
    for attribute, column in FILTER_COLUMNS.items():
        selected = getattr(filters, attribute)
        if selected:
            placeholders = ", ".join("?" for _ in selected)
            clauses.append(f"{prefix}{column} IN ({placeholders})")
            parameters.extend(selected)
    return " AND ".join(clauses), parameters


def _to_frame(
    connection: duckdb.DuckDBPyConnection, sql: str, parameters: Iterable[Any] = ()
) -> pd.DataFrame:
    return connection.execute(sql, list(parameters)).df()


def _table_exists(connection: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    row = connection.execute(
        """
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = 'main' AND table_name = ?
        LIMIT 1
        """,
        [table_name],
    ).fetchone()
    return row is not None


def _table_has_columns(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
    required_columns: set[str],
) -> bool:
    """Check whether a materialized table supports the required dashboard metrics."""

    available_columns = {
        str(row[0])
        for row in connection.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'main' AND table_name = ?
            """,
            [table_name],
        ).fetchall()
    }
    return required_columns.issubset(available_columns)


def _dimension_population_frame(
    frame: pd.DataFrame, dimension: str, dimension_column: str
) -> pd.DataFrame:
    return (
        frame.loc[
            frame["dimension"].eq(dimension),
            [
                "month_end",
                "dimension_value",
                "active_population",
                "active_registered_users",
                "gross_premium_usd",
                "net_premium_usd",
                "tpa_fee_usd",
            ],
        ]
        .rename(columns={"dimension_value": dimension_column})
        .sort_values(["month_end", dimension_column])
        .reset_index(drop=True)
    )


def query_dashboard(mart_path: str | Path, filters: FilterSpec) -> dict[str, Any]:
    """Return all small aggregated tables needed by the dashboard.

    A slim scoped temporary table ensures the policy fact table is scanned once
    for the premium, payer, and app views. The larger membership expansion is
    read only once for the active-population charts.
    """

    started = time.perf_counter()
    with _connect_readonly(mart_path) as connection:
        where_sql, parameters = _where_clause(filters)
        connection.execute(
            f"""
            CREATE TEMP TABLE scoped_policies AS
            SELECT
                uw_year,
                payer_country,
                payer_name,
                service_provider,
                master_contract,
                policy_type,
                network_type,
                network_group,
                age_bucket,
                beneficiary_key,
                registered_user_key,
                master_contract_key,
                gross_premium_usd,
                net_premium_usd,
                tpa_fee_usd
            FROM policy_records
            WHERE {where_sql}
            """,
            parameters,
        )
        summary = _to_frame(
            connection,
            """
            SELECT
                COUNT(*) AS policy_records,
                COUNT(DISTINCT beneficiary_key) AS unique_beneficiaries,
                COUNT(DISTINCT master_contract_key) AS master_contracts,
                COUNT(DISTINCT registered_user_key) AS registered_users,
                COUNT(DISTINCT CASE WHEN registered_user_key IS NOT NULL
                                    THEN beneficiary_key END) AS registered_beneficiaries,
                COALESCE(SUM(gross_premium_usd), 0) AS gross_premium_usd,
                COALESCE(SUM(net_premium_usd), 0) AS net_premium_usd,
                COALESCE(SUM(tpa_fee_usd), 0) AS tpa_fee_usd
            FROM scoped_policies
            """,
        )
        premium_by_year = _to_frame(
            connection,
            """
            SELECT
                uw_year,
                COUNT(DISTINCT beneficiary_key) AS beneficiaries,
                COUNT(DISTINCT registered_user_key) AS registered_users,
                CASE
                    WHEN COUNT(DISTINCT beneficiary_key) = 0 THEN NULL
                    ELSE COUNT(DISTINCT registered_user_key) * 1.0
                         / COUNT(DISTINCT beneficiary_key)
                END AS app_penetration,
                SUM(gross_premium_usd) AS gross_premium_usd,
                SUM(net_premium_usd) AS net_premium_usd,
                SUM(tpa_fee_usd) AS tpa_fee_usd
            FROM scoped_policies
            GROUP BY uw_year
            ORDER BY uw_year
            """,
        )
        payer_review_all = _to_frame(
            connection,
            """
            WITH payer_metrics AS (
                SELECT
                    COALESCE(payer_name, 'Unassigned') AS payer_name,
                    COUNT(DISTINCT beneficiary_key) AS beneficiaries,
                    COUNT(DISTINCT registered_user_key) AS registered_users,
                    COUNT(DISTINCT CASE WHEN registered_user_key IS NOT NULL
                                        THEN beneficiary_key END) AS registered_beneficiaries,
                    SUM(gross_premium_usd) AS gross_premium_usd,
                    SUM(net_premium_usd) AS net_premium_usd,
                    SUM(tpa_fee_usd) AS tpa_fee_usd
                FROM scoped_policies
                GROUP BY 1
            )
            SELECT
                *,
                CASE WHEN beneficiaries = 0 THEN NULL
                     ELSE registered_users * 1.0 / beneficiaries END AS app_penetration_rate,
                CASE WHEN gross_premium_usd = 0 THEN NULL
                     ELSE net_premium_usd / gross_premium_usd END AS net_to_gross_ratio,
                CASE WHEN gross_premium_usd = 0 THEN NULL
                     ELSE tpa_fee_usd / gross_premium_usd END AS tpa_to_gross_ratio
            FROM payer_metrics
            ORDER BY gross_premium_usd DESC NULLS LAST
            """,
        )
        payer_review = payer_review_all.head(25).copy()
        payer_network_premium = _to_frame(
            connection,
            """
            SELECT
                COALESCE(payer_name, 'Unassigned') AS payer_name,
                COALESCE(network_type, 'Unassigned') AS network_type,
                COALESCE(SUM(gross_premium_usd), 0) AS gross_premium_usd
            FROM scoped_policies
            GROUP BY 1, 2
            ORDER BY gross_premium_usd DESC NULLS LAST
            """,
        )
        master_contract_network_premium = _to_frame(
            connection,
            """
            SELECT
                COALESCE(master_contract, 'Unassigned') AS master_contract,
                COALESCE(network_type, 'Unassigned') AS network_type,
                COALESCE(SUM(gross_premium_usd), 0) AS gross_premium_usd
            FROM scoped_policies
            GROUP BY 1, 2
            ORDER BY gross_premium_usd DESC NULLS LAST
            """,
        )
        age_bucket_review = _to_frame(
            connection,
            """
            WITH age_bucket_metrics AS (
                SELECT
                    COALESCE(age_bucket, 'Unassigned') AS age_bucket,
                    COUNT(DISTINCT beneficiary_key) AS beneficiaries,
                    COUNT(DISTINCT registered_user_key) AS registered_users,
                    COUNT(DISTINCT CASE WHEN registered_user_key IS NOT NULL
                                        THEN beneficiary_key END) AS registered_beneficiaries,
                    SUM(gross_premium_usd) AS gross_premium_usd,
                    SUM(net_premium_usd) AS net_premium_usd,
                    SUM(tpa_fee_usd) AS tpa_fee_usd
                FROM scoped_policies
                GROUP BY 1
            )
            SELECT
                *,
                CASE WHEN beneficiaries = 0 THEN NULL
                     ELSE registered_users * 1.0 / beneficiaries END AS app_penetration_rate,
                CASE WHEN gross_premium_usd = 0 THEN NULL
                     ELSE net_premium_usd / gross_premium_usd END AS net_to_gross_ratio,
                CASE WHEN gross_premium_usd = 0 THEN NULL
                     ELSE tpa_fee_usd / gross_premium_usd END AS tpa_to_gross_ratio
            FROM age_bucket_metrics
            ORDER BY beneficiaries DESC NULLS LAST, age_bucket
            """,
        )
        policy_type_review = _to_frame(
            connection,
            """
            SELECT
                COALESCE(policy_type, 'Unassigned') AS policy_type,
                COUNT(DISTINCT beneficiary_key) AS beneficiaries,
                SUM(gross_premium_usd) AS gross_premium_usd,
                SUM(net_premium_usd) AS net_premium_usd,
                SUM(tpa_fee_usd) AS tpa_fee_usd
            FROM scoped_policies
            GROUP BY 1
            ORDER BY gross_premium_usd DESC NULLS LAST
            LIMIT 20
            """,
        )
        mobile_by_payer_all = _to_frame(
            connection,
            """
            WITH mobile_metrics AS (
                SELECT
                    COALESCE(payer_name, 'Unassigned') AS payer_name,
                    COUNT(DISTINCT beneficiary_key) AS unique_beneficiaries,
                    COUNT(DISTINCT registered_user_key) AS unique_registered_users,
                    COUNT(DISTINCT CASE WHEN registered_user_key IS NOT NULL
                                        THEN beneficiary_key END) AS linked_beneficiaries
                FROM scoped_policies
                GROUP BY 1
            )
            SELECT
                *,
                CASE WHEN unique_beneficiaries = 0 THEN NULL
                     ELSE unique_registered_users * 1.0 / unique_beneficiaries END
                    AS registered_user_penetration,
                CASE WHEN unique_beneficiaries = 0 THEN NULL
                     ELSE linked_beneficiaries * 1.0 / unique_beneficiaries END
                    AS linked_beneficiary_coverage
            FROM mobile_metrics
            ORDER BY unique_beneficiaries DESC
            """,
        )
        mobile_by_payer = mobile_by_payer_all.head(25).copy()

        metadata = {
            key: value
            for key, value in connection.execute(
                "SELECT key, value FROM mart_metadata"
            ).fetchall()
        }
        min_year = int(metadata["uw_year_min"])
        max_year = int(metadata["uw_year_max"])
        can_use_unfiltered_monthly_kpi_fast_path = (
            not filters.has_dimension_filters
            and filters.year_start == min_year
            and filters.year_end == max_year
            and _table_exists(connection, "monthly_dimension_population_default")
            and _table_has_columns(
                connection,
                "monthly_dimension_population_default",
                {
                    "active_population",
                    "active_registered_users",
                    "gross_premium_usd",
                    "net_premium_usd",
                    "tpa_fee_usd",
                },
            )
        )
        if can_use_unfiltered_monthly_kpi_fast_path:
            monthly_kpis = _to_frame(
                connection,
                """
                SELECT
                    month_end,
                    active_population,
                    active_registered_users,
                    gross_premium_usd,
                    net_premium_usd,
                    tpa_fee_usd
                FROM monthly_kpis_default
                ORDER BY month_end
                """,
            )
            monthly_country_kpis = _to_frame(
                connection,
                """
                SELECT
                    month_end,
                    payer_country,
                    active_population,
                    active_registered_users,
                    gross_premium_usd,
                    net_premium_usd,
                    tpa_fee_usd
                FROM monthly_country_kpis_default
                ORDER BY month_end, payer_country
                """,
            )
            monthly_dimension_population = _to_frame(
                connection,
                """
                SELECT
                    month_end,
                    dimension,
                    dimension_value,
                    active_population,
                    active_registered_users,
                    gross_premium_usd,
                    net_premium_usd,
                    tpa_fee_usd
                FROM monthly_dimension_population_default
                ORDER BY month_end, dimension, dimension_value
                """,
            )
            monthly_network_type_kpis = _dimension_population_frame(
                monthly_dimension_population, "network_type", "network_type"
            )
            monthly_network_group_kpis = _dimension_population_frame(
                monthly_dimension_population, "network_group", "network_group"
            )
            monthly_policy_type_kpis = _dimension_population_frame(
                monthly_dimension_population, "policy_type", "policy_type"
            )
        else:
            active_where_sql, active_parameters = _where_clause(filters)
            scoped_monthly_metrics = _to_frame(
                connection,
                f"""
                WITH scoped_active_months AS (
                    SELECT
                        month_end,
                        COALESCE(payer_country, 'Unassigned') AS payer_country,
                        COALESCE(network_type, 'Unassigned') AS network_type,
                        COALESCE(network_group, 'Unassigned') AS network_group,
                        COALESCE(policy_type, 'Unassigned') AS policy_type,
                        beneficiary_key,
                        registered_user_key,
                        registration_date,
                        gross_premium_usd,
                        net_premium_usd,
                        tpa_fee_usd
                    FROM active_member_months
                    WHERE {active_where_sql}
                )
                SELECT
                    month_end,
                    payer_country,
                    network_type,
                    network_group,
                    policy_type,
                    GROUPING(payer_country) AS payer_country_grouped,
                    GROUPING(network_type) AS network_type_grouped,
                    GROUPING(network_group) AS network_group_grouped,
                    GROUPING(policy_type) AS policy_type_grouped,
                    COUNT(DISTINCT beneficiary_key) AS active_population,
                    COUNT(DISTINCT CASE
                        WHEN beneficiary_key IS NOT NULL
                         AND registered_user_key IS NOT NULL
                         AND (registration_date IS NULL OR registration_date <= month_end)
                        THEN registered_user_key
                    END) AS active_registered_users,
                    COALESCE(SUM(gross_premium_usd), 0) AS gross_premium_usd,
                    COALESCE(SUM(net_premium_usd), 0) AS net_premium_usd,
                    COALESCE(SUM(tpa_fee_usd), 0) AS tpa_fee_usd
                FROM scoped_active_months
                GROUP BY GROUPING SETS (
                    (month_end),
                    (month_end, payer_country),
                    (month_end, network_type),
                    (month_end, network_group),
                    (month_end, policy_type)
                )
                ORDER BY month_end, payer_country, network_type, network_group, policy_type
                """,
                active_parameters,
            )
            monthly_kpis = scoped_monthly_metrics.loc[
                scoped_monthly_metrics["payer_country_grouped"].eq(1)
                & scoped_monthly_metrics["network_type_grouped"].eq(1)
                & scoped_monthly_metrics["network_group_grouped"].eq(1)
                & scoped_monthly_metrics["policy_type_grouped"].eq(1),
                [
                    "month_end",
                    "active_population",
                    "active_registered_users",
                    "gross_premium_usd",
                    "net_premium_usd",
                    "tpa_fee_usd",
                ],
            ].reset_index(drop=True)
            monthly_country_kpis = scoped_monthly_metrics.loc[
                scoped_monthly_metrics["payer_country_grouped"].eq(0),
                [
                    "month_end",
                    "payer_country",
                    "active_population",
                    "active_registered_users",
                    "gross_premium_usd",
                    "net_premium_usd",
                    "tpa_fee_usd",
                ],
            ].reset_index(drop=True)
            monthly_network_type_kpis = scoped_monthly_metrics.loc[
                scoped_monthly_metrics["network_type_grouped"].eq(0),
                [
                    "month_end",
                    "network_type",
                    "active_population",
                    "active_registered_users",
                    "gross_premium_usd",
                    "net_premium_usd",
                    "tpa_fee_usd",
                ],
            ].reset_index(drop=True)
            monthly_network_group_kpis = scoped_monthly_metrics.loc[
                scoped_monthly_metrics["network_group_grouped"].eq(0),
                [
                    "month_end",
                    "network_group",
                    "active_population",
                    "active_registered_users",
                    "gross_premium_usd",
                    "net_premium_usd",
                    "tpa_fee_usd",
                ],
            ].reset_index(drop=True)
            monthly_policy_type_kpis = scoped_monthly_metrics.loc[
                scoped_monthly_metrics["policy_type_grouped"].eq(0),
                [
                    "month_end",
                    "policy_type",
                    "active_population",
                    "active_registered_users",
                    "gross_premium_usd",
                    "net_premium_usd",
                    "tpa_fee_usd",
                ],
            ].reset_index(drop=True)

    active_population = monthly_kpis.loc[:, ["month_end", "active_population"]].copy()
    latest_active_population = (
        int(active_population.iloc[-1]["active_population"])
        if not active_population.empty
        else 0
    )
    latest_active_month = (
        active_population.iloc[-1]["month_end"] if not active_population.empty else None
    )
    summary["latest_active_population"] = latest_active_population
    summary["latest_active_month"] = latest_active_month
    return {
        "summary": summary,
        "active_population": active_population,
        "monthly_kpis": monthly_kpis,
        "monthly_country_kpis": monthly_country_kpis,
        "monthly_network_type_kpis": monthly_network_type_kpis,
        "monthly_network_group_kpis": monthly_network_group_kpis,
        "monthly_policy_type_kpis": monthly_policy_type_kpis,
        "premium_by_year": premium_by_year,
        "payer_review": payer_review,
        "payer_review_all": payer_review_all,
        "payer_network_premium": payer_network_premium,
        "master_contract_network_premium": master_contract_network_premium,
        "age_bucket_review": age_bucket_review,
        "policy_type_review": policy_type_review,
        "mobile_by_payer": mobile_by_payer,
        "mobile_by_payer_all": mobile_by_payer_all,
        "metadata": metadata,
        "query_ms": round((time.perf_counter() - started) * 1000, 1),
    }
