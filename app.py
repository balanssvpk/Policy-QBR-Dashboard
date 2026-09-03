"""C-level Policy QBR dashboard."""

from __future__ import annotations

from html import escape
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
ALLIANZ_LOGO_PATH = ROOT / "AZ_Partners_Attached_Descriptor_Positive_RGB_.png"
GEN_BI_EVALUATION_DIR = Path(
    os.getenv("GEN_BI_EVALUATION_DIR", str(ROOT / "data" / "gen_bi_evaluations"))
).expanduser()
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from policy_dashboard.data import (
    FILTER_COLUMNS,
    FilterSpec,
    get_filter_options,
    get_mart_metadata,
    query_dashboard,
)

from policy_dashboard.ollama_runtime import (
    OllamaRuntimeStatus,
    ensure_ollama,
    get_ollama_serve_status,
)

from policy_dashboard.gen_bi import (
    DETERMINISTIC_ENGINE,
    OLLAMA_ENGINE,
    OLLAMA_REPRODUCIBILITY_PROFILE,
    OllamaModelStatus,
    QuestionEvidence,
    ask_ollama,
    build_question_evidence,
    check_ollama_model_response,
    evidence_table_label,
    generate_executive_answer,
    get_ollama_config,
    record_evaluation,
)


st.set_page_config(
    page_title="Policy Portfolio QBR",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


DEFAULT_CHART_CATEGORICAL_COLORS = (
    "#003781",
    "#005FA8",
    "#007AB8",
    "#168CC7",
    "#4CB7E8",
    "#7CCBE9",
    "#004C8C",
)


def _theme_color(option_name: str, fallback: str) -> str:
    configured_color = st.get_option(option_name)
    return str(configured_color) if configured_color else fallback


CHART_CATEGORICAL_COLORS = tuple(
    str(color)
    for color in (
        st.get_option("theme.chartCategoricalColors")
        or DEFAULT_CHART_CATEGORICAL_COLORS
    )
)


def _chart_color(index: int) -> str:
    return CHART_CATEGORICAL_COLORS[index % len(CHART_CATEGORICAL_COLORS)]


# Legacy semantic key names are retained at the call sites, but every value is
# resolved from the active Streamlit theme in .streamlit/config.toml.
PALETTE = {
    "navy": _theme_color("theme.primaryColor", _chart_color(0)),
    "teal": _chart_color(1),
    "coral": _chart_color(2),
    "amber": _chart_color(3),
    "mint": _chart_color(4),
    "sky": _chart_color(5),
    "slate": _theme_color("theme.textColor", "#002B5C"),
    "mist": _theme_color("theme.secondaryBackgroundColor", "#EEF7FB"),
}
# Original executive-dashboard stack palette. This remains local to the
# dimensional population views; the rest of the app follows the active theme.
STACKED_TREND_COLORS = (
    "#00A6A6",
    "#11263E",
    "#E9795D",
    "#F2B84B",
    "#8ED6C8",
    "#607084",
)
MONTHLY_TREND_COLORS = {
    "active_population": "#00A6A6",
    "registered_users": "#11263E",
    "app_penetration": "#E9795D",
}
TAB_CHART_HEIGHT = 425


def _label_text_color(background_color: str) -> str:
    """Choose a readable label color for a hexadecimal chart fill color."""

    color = background_color.lstrip("#")
    if len(color) == 3:
        color = "".join(channel * 2 for channel in color)
    if len(color) != 6:
        return "#FFFFFF"
    try:
        red, green, blue = (int(color[offset : offset + 2], 16) for offset in (0, 2, 4))
    except ValueError:
        return "#FFFFFF"
    luminance = (0.2126 * red + 0.7152 * green + 0.0722 * blue) / 255
    return "#FFFFFF" if luminance < 0.56 else PALETTE["slate"]


FILTER_WIDGET_KEYS = {
    **{
        attribute: f"portfolio_filter_{attribute}"
        for attribute in FILTER_COLUMNS
    },
}
FILTER_RESET_EPOCH_KEY = "portfolio_filter_reset_epoch"
SNAPSHOT_CACHE_VERSION = "gen-bi-dimension-evidence-v1"
REQUIRED_SNAPSHOT_TABLES = (
    "master_contract_network_premium",
    "age_bucket_review",
)
GEN_BI_QUESTION_KEY = "gen_bi_question_text"
GEN_BI_SUGGESTION_KEY = "gen_bi_suggested_question"
GEN_BI_SUGGESTIONS = (
    "Top active‑population drivers and the action to prioritise.",
    "Largest gap between active population and registered users and the fix.",
    "Weakest app‑penetration segment and the digital push needed.",
    "Highest‑premium contracts, exposure concentration, and renewal checks.",
    "Highest TPA‑fee groups and levers to improve profitability.",
    "Segments with rising utilization and the action to control cost.",
    "Employer groups with poor digital adoption and the engagement to deploy.",
    "Networks causing the most leakage and the corrective step required.",
    "Beneficiary cohorts with high call‑center load and the digital shift needed.",
)


FILTER_LABELS = {
    "payer_countries": "Payer country",
    "payers": "Payer",
    "providers": "Service provider",
    "server_names": "Server name",
    "contracts": "Master contract",
    "payer_types": "Payer type",
    "policy_types": "Policy type",
    "policy_type_details": "Policy type detail",
    "licensing_authorities": "Licensing authority",
    "network_types": "Network type",
    "network_groups": "Network group",
    "dependencies": "Dependency",
    "nationalities": "Nationality",
    "marital_statuses": "Marital status",
    "gender": "Gender",
    "dependent_statuses": "Dependent status",
    "age_profiles": "Age profile",
    "age_buckets": "Age bucket",
    "app_names": "App name",
}

FILTER_GROUPS = (
    (
        "Portfolio & contract",
        (
            "payer_countries",
            "payers",
            "providers",
            "server_names",
            "contracts",
            "payer_types",
        ),
    ),
    (
        "Policy & network",
        (
            "policy_types",
            "policy_type_details",
            "licensing_authorities",
            "network_types",
            "network_groups",
        ),
    ),
    (
        "Member profile",
        (
            "dependencies",
            "nationalities",
            "marital_statuses",
            "gender",
            "dependent_statuses",
            "age_profiles",
            "age_buckets",
        ),
    ),
    ("Digital adoption", ("app_names",)),
)


def _inject_css() -> None:
    st.markdown(
        """
        <style>
          .block-container { max-width: 1540px; padding-top: 1.9rem; padding-bottom: 2rem; }
          .qbr-kicker { color: #00A6A6; font-size: 0.73rem; font-weight: 750; letter-spacing: 0.13em; }
          .qbr-title { margin: 0.25rem 0 0.35rem; color: #11263E; font-size: 2.4rem; font-weight: 720; letter-spacing: -0.045em; }
          .qbr-subtitle { color: #607084; margin-bottom: 1.35rem; font-size: 1rem; }
          .insight-card { background: #FFFFFF; border: 1px solid #E1E6EA; border-left: 4px solid #00A6A6; border-radius: 8px; padding: 1rem 1.15rem; min-height: 109px; }
          .insight-label { color: #607084; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; }
          .insight-value { color: #11263E; font-size: 1.65rem; line-height: 1.3; font-weight: 720; margin-top: 0.22rem; }
          .insight-trends { display: flex; gap: 0.75rem; margin-top: 0.55rem; }
          .insight-trend { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.01em; }
          .insight-trend-positive { color: #008A8A; }
          .insight-trend-negative { color: #C85F48; }
          .insight-trend-neutral { color: #607084; }
          .section-label { color: #11263E; font-size: 1.12rem; font-weight: 720; margin: 1.25rem 0 0.45rem; }
          [data-testid="stMetric"] { background: #FFFFFF; border: 1px solid #E1E6EA; border-radius: 8px; padding: 0.7rem 0.8rem; }
          [data-testid="stMetricLabel"] { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; }

          .stTabs [data-baseweb="tab-list"] { gap: 1.25rem; border-bottom: 1px solid #D9E0E5; }
          .stTabs [data-baseweb="tab"] { height: 42px; padding: 0; color: #607084; font-weight: 800; font-size: 20px; }
          .stTabs [aria-selected="true"] { color: #11263E; border-bottom-color: #00A6A6; font-weight: 900; }

          .st-key-page-portfolio-tabs [data-baseweb="tab"] { font-size: 14px; font-weight: 700; }
          .st-key-page-portfolio-tabs [data-baseweb="tab"][aria-selected="true"] { font-weight: 700; }
          .st-key-page-portfolio-tabs [data-baseweb="tab"] p { font-size: inherit; font-weight: inherit; }

          .st-key-overview-portfolio-tabs [data-baseweb="tab"] { font-size: 14px; font-weight: 700; }
          .st-key-overview-portfolio-tabs [data-baseweb="tab"][aria-selected="true"] { font-weight: 700; }
          .st-key-overview-portfolio-tabs [data-baseweb="tab"] p { font-size: inherit; font-weight: inherit; }

          .st-key-active-population-portfolio-tabs [data-baseweb="tab"] { font-size: 14px; font-weight: 700; }
          .st-key-active-population-portfolio-tabs [data-baseweb="tab"][aria-selected="true"] { font-weight: 700; }
          .st-key-active-population-portfolio-tabs [data-baseweb="tab"] p { font-size: inherit; font-weight: inherit; }

          .st-key-active-registered-portfolio-tabs [data-baseweb="tab"] { font-size: 14px; font-weight: 700; }
          .st-key-active-registered-portfolio-tabs [data-baseweb="tab"][aria-selected="true"] { font-weight: 700; }
          .st-key-active-registered-portfolio-tabs [data-baseweb="tab"] p { font-size: inherit; font-weight: inherit; }

          .st-key-premium-portfolio-tabs [data-baseweb="tab"] { font-size: 14px; font-weight: 700; }
          .st-key-premium-portfolio-tabs [data-baseweb="tab"][aria-selected="true"] { font-weight: 700; }
          .st-key-premium-portfolio-tabs [data-baseweb="tab"] p { font-size: inherit; font-weight: inherit; }

          .st-key-active-population-gross-premium-mix-tabs [data-baseweb="tab"],
          .st-key-active-registered-gross-premium-mix-tabs [data-baseweb="tab"],
          .st-key-premium-gross-premium-mix-tabs [data-baseweb="tab"] { font-size: 14px; font-weight: 700; }
          .st-key-active-population-gross-premium-mix-tabs [data-baseweb="tab"][aria-selected="true"],
          .st-key-active-registered-gross-premium-mix-tabs [data-baseweb="tab"][aria-selected="true"],
          .st-key-premium-gross-premium-mix-tabs [data-baseweb="tab"][aria-selected="true"] { font-weight: 700; }
          .st-key-active-population-gross-premium-mix-tabs [data-baseweb="tab"] p,
          .st-key-active-registered-gross-premium-mix-tabs [data-baseweb="tab"] p,
          .st-key-premium-gross-premium-mix-tabs [data-baseweb="tab"] p { font-size: inherit; font-weight: inherit; }

          .stButton > button { border-radius: 6px; font-weight: 650; }
          [data-testid="stDataFrame"] { border: 1px solid #E1E6EA; border-radius: 8px; overflow: hidden; }

          /* Sidebar expander icon styling — arrows visible */
          [data-testid="stSidebar"] [data-testid="stExpander"] > details > summary [data-testid="stIconMaterial"] {
            color: #003781 !important;     /* Allianz dark blue */
            opacity: 1 !important;
            background: none !important;   /* remove pill */
            box-shadow: none !important;   /* remove outline */
            border-radius: 0 !important;   /* restore default arrow shape */
          }

          /* Closed state — keep color but do NOT hide arrow */
          [data-testid="stSidebar"] [data-testid="stExpander"] > details:not([open]) > summary [data-testid="stIconMaterial"] {
            color: #003781 !important;
            opacity: 1 !important;
            background: none !important;
            box-shadow: none !important;
            border-radius: 0 !important;
          }

        /* Keep the native header toolbar available: it owns Streamlit's
           sidebar collapse/expand control. */
        #MainMenu { visibility: hidden !important; }
        .stDeployButton,
        .stAppDeployButton,
        [data-testid="stAppDeployButton"] { display: none !important; }


        </style>
        """,
        unsafe_allow_html=True,
    )



@st.cache_data(show_spinner=False)
def _cached_filter_options(db_path: str, modified_ns: int) -> dict[str, list[Any]]:
    del modified_ns
    return get_filter_options(db_path)


@st.cache_data(show_spinner=False)
def _cached_snapshot(
    db_path: str,
    modified_ns: int,
    filters_json: str,
    snapshot_cache_version: str,
) -> dict[str, Any]:
    # The version is intentionally part of the cache key.  query_dashboard can
    # gain new result frames while this wrapper's query arguments stay the same.
    del modified_ns, snapshot_cache_version
    return query_dashboard(db_path, FilterSpec.from_dict(json.loads(filters_json)))


@st.cache_data(show_spinner=False, ttl=900, max_entries=128)
def _cached_ollama_answer(
    host: str,
    model: str,
    question: str,
    context_json: str,
    generation_profile: str,
) -> str:
    """Cache repeated aggregate-only Generative BI requests for a short, bounded window."""

    # The profile is intentionally a cache-key component. Altering generation
    # controls must never return a response created under a previous profile.
    del generation_profile
    return ask_ollama(
        host=host,
        model=model,
        question=question,
        context_json=context_json,
    )


@st.cache_data(show_spinner=False, ttl=15, max_entries=16)
def _cached_ollama_serve_status(host: str) -> OllamaRuntimeStatus:
    """Keep the passive service health probe out of ordinary filter reruns."""

    return get_ollama_serve_status(host)


def _compact_money(value: Any) -> str:
    value = 0.0 if value is None or pd.isna(value) else float(value)
    for threshold, suffix in ((1_000_000_000, "bn"), (1_000_000, "m"), (1_000, "k")):
        if abs(value) >= threshold:
            return f"${value / threshold:,.1f}{suffix}"
    return f"${value:,.0f}"


def _compact_number(value: Any) -> str:
    value = 0.0 if value is None or pd.isna(value) else float(value)
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:,.1f}m"
    if abs(value) >= 1_000:
        return f"{value / 1_000:,.1f}k"
    return f"{value:,.0f}"


def _percent(numerator: Any, denominator: Any) -> float | None:
    if numerator is None or denominator is None or pd.isna(numerator) or pd.isna(denominator):
        return None
    denominator = float(denominator)
    return float(numerator) / denominator if denominator else None


def _format_frame(frame: pd.DataFrame, percent_columns: set[str] | None = None) -> Any:
    percent_columns = percent_columns or set()
    currency_columns = {
        column
        for column in frame.columns
        if column in {"gross_premium_usd", "net_premium_usd", "tpa_fee_usd"}
    }
    number_columns = {
        column
        for column in frame.columns
        if column not in currency_columns | percent_columns
        and pd.api.types.is_numeric_dtype(frame[column])
    }
    formats: dict[str, str] = {column: "${:,.0f}" for column in currency_columns}
    formats.update({column: "{:.1%}" for column in percent_columns if column in frame})
    formats.update({column: "{:,.0f}" for column in number_columns})
    return frame.style.format(formats, na_rep="—")


def _country_month_matrix_html(
    country_monthly: pd.DataFrame, display_months: pd.DatetimeIndex
) -> str:
    """Render a compact, grouped country-by-month population matrix."""

    payer_countries = sorted(country_monthly["payer_country"].dropna().unique())
    active_population_by_month = country_monthly.pivot(
        index="payer_country",
        columns="month_end",
        values="active_population",
    ).reindex(index=payer_countries, columns=display_months, fill_value=0).fillna(0)
    active_registered_by_month = country_monthly.pivot(
        index="payer_country",
        columns="month_end",
        values="active_registered_users",
    ).reindex(index=payer_countries, columns=display_months, fill_value=0).fillna(0)

    month_headers = "".join(
        "<th class=\"policy-country-matrix__month\" colspan=\"3\" "
        f"scope=\"colgroup\">{escape(month.strftime('%b %Y'))}</th>"
        for month in display_months
    )
    metric_headers = "".join(
        "<th scope=\"col\"><abbr title=\"Active population\">AP</abbr></th>"
        "<th scope=\"col\"><abbr title=\"Active Registered\">AR</abbr></th>"
        "<th class=\"policy-country-matrix__month-end\" scope=\"col\">"
        "<abbr title=\"App Penetration\">Pen</abbr></th>"
        for _ in display_months
    )
    body_rows: list[str] = []
    for payer_country in payer_countries:
        month_values: list[str] = []
        for month in display_months:
            population = int(active_population_by_month.loc[payer_country, month])
            registered = int(active_registered_by_month.loc[payer_country, month])
            penetration = "&mdash;" if population == 0 else f"{registered / population:.1%}"
            month_values.extend(
                [
                    f"<td>{population:,}</td>",
                    f"<td>{registered:,}</td>",
                    f"<td class=\"policy-country-matrix__month-end\">{penetration}</td>",
                ]
            )
        body_rows.append(
            "<tr>"
            f"<th class=\"policy-country-matrix__country\" scope=\"row\">{escape(str(payer_country))}</th>"
            f"{''.join(month_values)}"
            "</tr>"
        )

    return f"""
    <style>
      .policy-country-matrix__scroller {{
        max-height: 450px;
        overflow: auto;
        border: 1px solid #C7DDEA;
        border-radius: 8px;
        background: #FFFFFF;
      }}
      .policy-country-matrix {{
        width: max-content;
        min-width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        color: #002B5C;
        font-family: inherit;
        font-size: 0.72rem;
      }}
      .policy-country-matrix th,
      .policy-country-matrix td {{
        padding: 0.52rem 0.58rem;
        white-space: nowrap;
        vertical-align: middle;
        border-bottom: 1px solid #E4F3FA;
      }}
      .policy-country-matrix td {{
        text-align: right;
        font-variant-numeric: tabular-nums;
      }}
      .policy-country-matrix thead th {{
        position: sticky;
        z-index: 2;
        font-weight: 750;
        text-align: center;
        vertical-align: middle;
      }}
      .policy-country-matrix thead tr:first-child th {{
        top: 0;
        color: #FFFFFF;
        background: #00A6A6;
        letter-spacing: 0.02em;
      }}
      .policy-country-matrix thead tr:nth-child(2) th {{
        top: 33px;
        color: #FFFFFF;
        background: #00A6A6;
      }}
      .policy-country-matrix .policy-country-matrix__corner {{
        position: sticky;
        left: 0;
        z-index: 4;
        min-width: 150px;
        text-align: left !important;
        vertical-align: middle;
      }}
      .policy-country-matrix .policy-country-matrix__country {{
        position: sticky;
        left: 0;
        z-index: 1;
        min-width: 150px;
        color: #002B5C;
        background: #FFFFFF;
        text-align: left;
        font-weight: 700;
        border-right: 1px solid #C7DDEA;
      }}
      .policy-country-matrix tbody tr:nth-child(even) td,
      .policy-country-matrix tbody tr:nth-child(even) .policy-country-matrix__country {{
        background: #F7FBFE;
      }}
      .policy-country-matrix tbody tr:hover td,
      .policy-country-matrix tbody tr:hover .policy-country-matrix__country {{
        background: #E4F3FA;
      }}
      .policy-country-matrix .policy-country-matrix__month {{
        border-right: 2px solid #C7DDEA;
      }}
      .policy-country-matrix .policy-country-matrix__month-end {{
        border-right: 2px solid #C7DDEA;
      }}
      .policy-country-matrix abbr {{ text-decoration: none; }}
      .policy-country-matrix__key {{
        margin: 0.45rem 0 0;
        color: #4C657A;
        font-size: 0.72rem;
      }}
    </style>
    <div class="policy-country-matrix__scroller">
      <table class="policy-country-matrix" aria-label="Payer country monthly app adoption matrix">
        <thead>
          <tr>
            <th class="policy-country-matrix__corner" rowspan="2" scope="col">Payer country</th>
            {month_headers}
          </tr>
          <tr>{metric_headers}</tr>
        </thead>
        <tbody>{''.join(body_rows)}</tbody>
      </table>
    </div>
    <p class="policy-country-matrix__key">
        <strong>AP</strong> = Active Population &nbsp;&middot;&nbsp;
        <strong>AR</strong> = Active Registered &nbsp;&middot;&nbsp;
        <strong>Pen</strong> = App Penetration
    </p>
    """


def _render_stacked_active_population_trend(
    data: pd.DataFrame,
    *,
    dimension_column: str,
    title: str,
    metric_column: str = "active_population",
    metric_label: str = "Active population",
    metric_value_label: str = "active beneficiaries",
    value_formatter: Callable[[Any], str] = _compact_number,
    height: int = TAB_CHART_HEIGHT,
) -> None:
    """Render a compact, month-end membership trend by one dimension."""

    required_columns = {"month_end", dimension_column, metric_column}
    if data.empty or not required_columns.issubset(data.columns):
        st.info(f"No {metric_label.lower()} history is available for the current selection.")
        return

    chart_data = data.copy()
    chart_data["month_end"] = pd.to_datetime(chart_data["month_end"])
    chart_data[dimension_column] = (
        chart_data[dimension_column].fillna("Unassigned").astype(str)
    )
    display_months = pd.DatetimeIndex(
        chart_data["month_end"].dropna().unique()
    ).sort_values()[-24:]
    chart_data = chart_data.loc[
        chart_data["month_end"].isin(display_months)
    ].sort_values(["month_end", dimension_column])
    if chart_data.empty:
        st.info(f"No {metric_label.lower()} history is available for the current selection.")
        return

    dimension_values = (
        chart_data.groupby(dimension_column, as_index=True)[metric_column]
        .sum()
        .sort_values(ascending=False)
        .index.tolist()
    )
    monthly_totals = (
        chart_data.groupby("month_end", as_index=False)[metric_column]
        .sum()
        .sort_values("month_end")
    )
    largest_total = float(monthly_totals[metric_column].max())
    figure = go.Figure()
    for index, dimension_value in enumerate(dimension_values):
        series = chart_data.loc[
            chart_data[dimension_column].eq(dimension_value)
        ]
        bar_color = STACKED_TREND_COLORS[index % len(STACKED_TREND_COLORS)]
        figure.add_bar(
            name=dimension_value,
            x=series["month_end"],
            y=series[metric_column],
            marker=dict(
                color=bar_color,
                line=dict(color="#FFFFFF", width=1),
            ),
            text=[
                value_formatter(value) if pd.notna(value) and value > 0 else ""
                for value in series[metric_column]
            ],
            texttemplate="%{text}",
            textposition="inside",
            insidetextanchor="end",
            textfont=dict(color=_label_text_color(bar_color), size=11),
            constraintext="inside",
            hovertemplate=(
                "%{x|%b %Y}<br>"
                f"<b>{dimension_value}</b>: %{{y:,.0f}} {metric_value_label}"
                "<extra></extra>"
            ),
        )
    figure.add_scatter(
        name=f"Total {metric_label.lower()}",
        x=monthly_totals["month_end"],
        y=monthly_totals[metric_column],
        mode="text",
        text=[
            f"<b>{value_formatter(value)}</b>"
            for value in monthly_totals[metric_column]
        ],
        textposition="top center",
        textfont=dict(color=PALETTE["navy"], size=11),
        hovertemplate=(
            "%{x|%b %Y}<br>"
            f"<b>Total: %{{y:,.0f}}</b> {metric_value_label}<extra></extra>"
        ),
        showlegend=False,
        cliponaxis=False,
    )
    figure.update_layout(
        barmode="stack",
        height=height,
        margin=dict(l=0, r=10, t=60, b=0),
        title=dict(
            text=title,
            x=0,
            xanchor="left",
            font=dict(size=16, color=PALETTE["navy"]),
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=0.97, xanchor="center", x=0.5
        ),
        uniformtext=dict(minsize=9, mode="hide"),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        xaxis=dict(
            title="",
            tickformat="%b<br>%Y",
            dtick="M1",
            tickangle=0,
            tickfont=dict(size=12),
        ),
        yaxis=dict(
            title=metric_label,
            range=[0, largest_total * 1.16] if largest_total else [0, 1],
        ),
    )
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})


def _render_monthly_active_population_trend(
    data: pd.DataFrame,
    *,
    metric_column: str = "active_population",
    metric_label: str = "Active population",
    metric_value_label: str = "active beneficiaries",
    title: str = "Month-end active population",
    bar_color: str = MONTHLY_TREND_COLORS["active_population"],
    value_formatter: Callable[[Any], str] = _compact_number,
    height: int = TAB_CHART_HEIGHT,
) -> None:
    """Render a single-metric, month-end membership trend for the last 24 months."""

    required_columns = {"month_end", metric_column}
    if data.empty or not required_columns.issubset(data.columns):
        st.info(f"No {metric_label.lower()} history is available for the current selection.")
        return

    chart_data = data.copy()
    chart_data["month_end"] = pd.to_datetime(chart_data["month_end"])
    chart_data = (
        chart_data.dropna(subset=["month_end"])
        .sort_values("month_end")
        .drop_duplicates("month_end", keep="last")
        .tail(24)
    )
    if chart_data.empty:
        st.info(f"No {metric_label.lower()} history is available for the current selection.")
        return

    figure = go.Figure()
    figure.add_bar(
        name=metric_label,
        x=chart_data["month_end"],
        y=chart_data[metric_column],
        marker_color=bar_color,
        opacity=0.82,
        text=[value_formatter(value) for value in chart_data[metric_column]],
        texttemplate="%{text}",
        textposition="outside",
        textfont=dict(color=bar_color),
        cliponaxis=False,
        hovertemplate=(
            "%{x|%b %Y}<br>"
            f"<b>%{{y:,.0f}}</b> {metric_value_label}<extra></extra>"
        ),
    )
    figure.update_layout(
        height=height,
        margin=dict(l=0, r=10, t=60, b=0),
        title=dict(
            text=title,
            x=0,
            xanchor="left",
            font=dict(size=16, color=PALETTE["navy"]),
        ),
        showlegend=False,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        xaxis=dict(
            title="",
            tickformat="%b<br>%Y",
            dtick="M1",
            tickangle=0,
            tickfont=dict(size=12),
        ),
        yaxis=dict(title=metric_label, rangemode="tozero"),
    )
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})


def _render_yearly_premium_view(
    annual: pd.DataFrame,
    *,
    show_table: bool = True,
    chart_key: str = "premium_yearly_view",
    height: int = TAB_CHART_HEIGHT,
) -> None:
    """Render the policy-year premium comparison retained on the Premium page."""

    required_columns = {
        "uw_year",
        "gross_premium_usd",
        "net_premium_usd",
        "tpa_fee_usd",
    }
    if annual.empty or not required_columns.issubset(annual.columns):
        st.info("No underwriting-year premium history is available for the current selection.")
        return

    chart_data = annual.sort_values("uw_year").copy()
    figure = go.Figure()
    for label, column, color in (
        ("Gross premium", "gross_premium_usd", MONTHLY_TREND_COLORS["active_population"]),
        ("Net premium", "net_premium_usd", MONTHLY_TREND_COLORS["registered_users"]),
        ("TPA fee", "tpa_fee_usd", MONTHLY_TREND_COLORS["app_penetration"]),
    ):
        figure.add_bar(
            name=label,
            x=chart_data["uw_year"],
            y=chart_data[column],
            marker_color=color,
            text=[_compact_money(value) for value in chart_data[column]],
            texttemplate="%{text}",
            textposition="outside",
            textfont=dict(color=color),
            cliponaxis=False,
            hovertemplate=(
                "%{x}<br>"
                f"<b>{label}:</b> $%{{y:,.0f}}<extra></extra>"
            ),
        )
    figure.update_layout(
        barmode="group",
        height=height,
        margin=dict(l=0, r=10, t=60, b=0),
        title=dict(
            text="Gross premium, net premium and TPA fee by underwriting year",
            x=0,
            xanchor="left",
            font=dict(size=16, color=PALETTE["navy"]),
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=0.97, xanchor="center", x=0.5
        ),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        xaxis=dict(title="UW year", type="category"),
        yaxis=dict(title="USD", rangemode="tozero"),
    )
    st.plotly_chart(
        figure,
        width="stretch",
        config={"displayModeBar": False},
        key=chart_key,
    )
    # if show_table:
    #     st.dataframe(_format_frame(chart_data), width="stretch", hide_index=True)


def _render_yearly_metric_view(
    annual: pd.DataFrame,
    *,
    metric_column: str,
    metric_label: str,
    metric_value_label: str,
    title: str,
    bar_color: str,
    value_formatter: Callable[[Any], str] = _compact_number,
    height: int = TAB_CHART_HEIGHT,
) -> None:
    """Render a focused underwriting-year view for one page's leading measure."""

    required_columns = {"uw_year", metric_column}
    if annual.empty or not required_columns.issubset(annual.columns):
        st.info(f"No {metric_label.lower()} history is available for the current selection.")
        return

    chart_data = annual.sort_values("uw_year").copy()
    figure = go.Figure()
    figure.add_bar(
        name=metric_label,
        x=chart_data["uw_year"],
        y=chart_data[metric_column],
        marker_color=bar_color,
        opacity=0.84,
        text=[value_formatter(value) for value in chart_data[metric_column]],
        texttemplate="%{text}",
        textposition="outside",
        textfont=dict(color=bar_color),
        cliponaxis=False,
        hovertemplate=(
            "%{x}<br>"
            f"<b>%{{y:,.0f}}</b> {metric_value_label}<extra></extra>"
        ),
    )
    figure.update_layout(
        height=height,
        margin=dict(l=0, r=10, t=60, b=0),
        title=dict(
            text=title,
            x=0,
            xanchor="left",
            font=dict(size=16, color=PALETTE["navy"]),
        ),
        showlegend=False,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        xaxis=dict(title="UW year", type="category"),
        yaxis=dict(title=metric_label, rangemode="tozero"),
    )
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})


def _render_yearly_executive_member_view(
    annual: pd.DataFrame, *, height: int = 390
) -> None:
    """Render the annual member-health counterpart to the executive monthly chart."""

    required_columns = {
        "uw_year",
        "beneficiaries",
        "registered_users",
        "app_penetration",
    }
    if annual.empty or not required_columns.issubset(annual.columns):
        st.info("No underwriting-year member history is available for the current selection.")
        return

    chart_data = annual.sort_values("uw_year").copy()
    figure = go.Figure()
    figure.add_bar(
        name="Active population",
        x=chart_data["uw_year"],
        y=chart_data["beneficiaries"],
        marker_color=MONTHLY_TREND_COLORS["active_population"],
        opacity=0.82,
        offsetgroup="active_population",
        text=[_compact_number(value) for value in chart_data["beneficiaries"]],
        texttemplate="%{text}",
        textposition="outside",
        textfont=dict(color=MONTHLY_TREND_COLORS["active_population"]),
        cliponaxis=False,
        hovertemplate="%{x}<br><b>%{y:,.0f}</b> active beneficiaries<extra></extra>",
    )
    figure.add_bar(
        name="Active Registered",
        x=chart_data["uw_year"],
        y=chart_data["registered_users"],
        marker_color=MONTHLY_TREND_COLORS["registered_users"],
        opacity=0.88,
        offsetgroup="active_registered_users",
        text=[_compact_number(value) for value in chart_data["registered_users"]],
        texttemplate="%{text}",
        textposition="outside",
        textfont=dict(color=MONTHLY_TREND_COLORS["registered_users"]),
        cliponaxis=False,
        hovertemplate="%{x}<br><b>%{y:,.0f}</b> active registered users<extra></extra>",
    )
    figure.add_scatter(
        name="App Penetration",
        x=chart_data["uw_year"],
        y=chart_data["app_penetration"],
        yaxis="y2",
        mode="lines+markers+text",
        line=dict(color=MONTHLY_TREND_COLORS["app_penetration"], width=2.5),
        marker=dict(size=6),
        text=[
            f"{value:.1%}" if pd.notna(value) else ""
            for value in chart_data["app_penetration"]
        ],
        textposition="top center",
        textfont=dict(color=MONTHLY_TREND_COLORS["app_penetration"], size=10),
        cliponaxis=False,
        hovertemplate="%{x}<br><b>%{y:.1%}</b> app penetration<extra></extra>",
    )
    figure.update_layout(
        barmode="group",
        bargap=0.16,
        bargroupgap=0,
        height=height,
        margin=dict(l=0, r=10, t=60, b=0),
        title=dict(
            text="Active population, Active Registered and App Penetration by underwriting year",
            x=0,
            xanchor="left",
            font=dict(size=16, color=PALETTE["navy"]),
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=0.97, xanchor="center", x=0.5
        ),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        xaxis=dict(title="UW year", type="category"),
        yaxis=dict(title="Active population", rangemode="tozero"),
        yaxis2=dict(
            title="App Penetration",
            overlaying="y",
            side="right",
            tickformat=".0%",
            range=[0, 1.09],
            dtick=0.2,
        ),
    )
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})


def _render_monthly_premium_trend(
    data: pd.DataFrame, *, height: int = TAB_CHART_HEIGHT
) -> None:
    """Render the latest 24 months of apportioned GP, NP and TPA fee."""

    required_columns = {
        "month_end",
        "gross_premium_usd",
        "net_premium_usd",
        "tpa_fee_usd",
    }
    if data.empty or not required_columns.issubset(data.columns):
        st.info("No monthly premium history is available for the current selection.")
        return

    chart_data = data.copy()
    chart_data["month_end"] = pd.to_datetime(chart_data["month_end"])
    chart_data = (
        chart_data.dropna(subset=["month_end"])
        .sort_values("month_end")
        .drop_duplicates("month_end", keep="last")
        .tail(24)
    )
    if chart_data.empty:
        st.info("No monthly premium history is available for the current selection.")
        return

    figure = go.Figure()
    for label, column, color in (
        ("Gross premium", "gross_premium_usd", MONTHLY_TREND_COLORS["active_population"]),
        ("Net premium", "net_premium_usd", MONTHLY_TREND_COLORS["registered_users"]),
        ("TPA fee", "tpa_fee_usd", MONTHLY_TREND_COLORS["app_penetration"]),
    ):
        figure.add_bar(
            name=label,
            x=chart_data["month_end"],
            y=chart_data[column],
            marker_color=color,
            opacity=0.86,
            text=[_compact_money(value) for value in chart_data[column]],
            texttemplate="%{text}",
            textposition="outside",
            textfont=dict(color=color, size=10),
            cliponaxis=False,
            hovertemplate=(
                "%{x|%b %Y}<br>"
                f"<b>{label}:</b> $%{{y:,.0f}}<extra></extra>"
            ),
        )
    figure.update_layout(
        barmode="group",
        bargap=0.16,
        bargroupgap=0,
        height=height,
        margin=dict(l=0, r=10, t=60, b=0),
        title=dict(
            text="Month-end gross premium, net premium and TPA fee",
            x=0,
            xanchor="left",
            font=dict(size=16, color=PALETTE["navy"]),
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=0.97, xanchor="center", x=0.5
        ),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        xaxis=dict(
            title="",
            tickformat="%b<br>%Y",
            dtick="M1",
            tickangle=0,
            tickfont=dict(size=12),
        ),
        yaxis=dict(title="USD", rangemode="tozero"),
    )
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})


def _dimension_trend_frame(
    snapshot: dict[str, Any],
    snapshot_key: str,
    dimension_column: str,
    metric_column: str = "active_population",
) -> pd.DataFrame:
    """Return an empty, renderable frame when a legacy cached snapshot is used."""

    frame = snapshot.get(snapshot_key)
    if isinstance(frame, pd.DataFrame):
        return frame
    return pd.DataFrame(
        columns=["month_end", dimension_column, metric_column]
    )


def _format_change(change: float | None) -> tuple[str, str]:
    if change is None or pd.isna(change):
        return "—", "neutral"
    if change > 0:
        return f"+{change:.1%}", "positive"
    if change < 0:
        return f"{change:.1%}", "negative"
    return "0.0%", "neutral"


def _metric_card(
    label: str, value: str, mom_change: float | None, yoy_change: float | None
) -> None:
    mom_text, mom_tone = _format_change(mom_change)
    yoy_text, yoy_tone = _format_change(yoy_change)
    st.markdown(
        f"""
        <div class="insight-card">
          <div class="insight-label">{label}</div>
          <div class="insight-value">{value}</div>
          <div class="insight-trends">
            <span class="insight-trend insight-trend-{mom_tone}">MoM {mom_text}</span>
            <span class="insight-trend insight-trend-{yoy_tone}">YoY {yoy_text}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _change_from_reference(current: Any, reference: Any) -> float | None:
    if current is None or reference is None or pd.isna(current) or pd.isna(reference):
        return None
    reference_value = float(reference)
    return None if reference_value == 0 else float(current) / reference_value - 1


def _latest_kpi_metrics(monthly_kpis: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    """Return the latest values plus exact prior-month and prior-year changes."""

    metrics = (
        "active_population",
        "active_registered_users",
        "app_penetration",
        "gross_premium_usd",
        "net_premium_usd",
        "tpa_fee_usd",
    )
    empty = {metric: {"value": None, "mom": None, "yoy": None} for metric in metrics}
    if monthly_kpis.empty:
        return empty

    trend = monthly_kpis.copy()
    trend["month_end"] = pd.to_datetime(trend["month_end"]).dt.normalize()
    trend = trend.sort_values("month_end").drop_duplicates("month_end", keep="last")
    denominator = trend["active_population"].where(trend["active_population"].ne(0))
    trend["app_penetration"] = trend["active_registered_users"].div(denominator)

    latest = trend.iloc[-1]
    latest_month = latest["month_end"]
    indexed = trend.set_index("month_end")
    mom_reference_date = latest_month - pd.DateOffset(months=1)
    yoy_reference_date = latest_month - pd.DateOffset(years=1)
    mom_reference = indexed.loc[mom_reference_date] if mom_reference_date in indexed.index else None
    yoy_reference = indexed.loc[yoy_reference_date] if yoy_reference_date in indexed.index else None

    return {
        metric: {
            "value": latest[metric],
            "mom": _change_from_reference(
                latest[metric], None if mom_reference is None else mom_reference[metric]
            ),
            "yoy": _change_from_reference(
                latest[metric], None if yoy_reference is None else yoy_reference[metric]
            ),
        }
        for metric in metrics
    }


def _formatted_metric(value: float | None, formatter: Any) -> str:
    return "—" if value is None or pd.isna(value) else formatter(value)


def _scope_text(filters: FilterSpec) -> str:
    # fragments = [f"UW years {filters.year_start}–{filters.year_end}"]
    fragments = [] 
    applied_dimensions: list[str] = []
    for attribute, label in FILTER_LABELS.items():
        values = getattr(filters, attribute)
        if values:
            applied_dimensions.append(
                f"{label.lower()}: {', '.join(values[:3])}{' +' if len(values) > 3 else ''}"
            )
    fragments.extend(applied_dimensions[:3])
    if len(applied_dimensions) > 3:
        fragments.append(f"+{len(applied_dimensions) - 3} more filters")
    return " | ".join(fragments)


def _filter_widget_keys() -> dict[str, str]:
    """Give controls a fresh identity after a reset to clear browser form state."""

    epoch = st.session_state.get(FILTER_RESET_EPOCH_KEY, 0)
    return {name: f"{base_key}_{epoch}" for name, base_key in FILTER_WIDGET_KEYS.items()}


def _filter_widget_defaults(
    filters: FilterSpec, widget_keys: dict[str, str]
) -> dict[str, Any]:
    """Map the applied filter scope to the exact values held by UI widgets."""

    return {
        **{
            widget_keys[attribute]: list(getattr(filters, attribute))
            for attribute in FILTER_COLUMNS
        },
    }


def _initialize_filter_widget_state(
    filters: FilterSpec, widget_keys: dict[str, str]
) -> None:
    defaults = _filter_widget_defaults(filters, widget_keys)
    for widget_key, value in defaults.items():
        st.session_state.setdefault(widget_key, value)


def _reset_filter_state(default: FilterSpec) -> None:
    """Clear both the applied query scope and every persisted widget value."""

    st.session_state["applied_filters"] = default
    old_epoch = st.session_state.get(FILTER_RESET_EPOCH_KEY, 0)
    for base_key in FILTER_WIDGET_KEYS.values():
        st.session_state.pop(f"{base_key}_{old_epoch}", None)
    # New widget identities force Streamlit to discard buffered browser values
    # from the form before the clean controls are rendered.
    st.session_state[FILTER_RESET_EPOCH_KEY] = old_epoch + 1


def _render_sidebar(options: dict[str, list[Any]]) -> FilterSpec:
    years = [int(year) for year in options.get("uw_year", [])]
    if not years:
        st.error("No valid UW years were found in the mart.")
        st.stop()
    default = FilterSpec(year_start=min(years), year_end=max(years))
    applied = st.session_state.get("applied_filters", default)
    if (applied.year_start, applied.year_end) != (
        default.year_start,
        default.year_end,
    ):
        # The year slider was removed. Do not preserve a hidden historical
        # selection from an earlier browser session.
        applied = FilterSpec(
            year_start=default.year_start,
            year_end=default.year_end,
            **{
                attribute: getattr(applied, attribute)
                for attribute in FILTER_COLUMNS
            },
        )
        st.session_state["applied_filters"] = applied
    widget_keys = _filter_widget_keys()
    _initialize_filter_widget_state(applied, widget_keys)

    with st.sidebar:
        if ALLIANZ_LOGO_PATH.is_file():
            st.image(str(ALLIANZ_LOGO_PATH), width="stretch")
            st.divider( width="stretch")
        st.markdown("### Data Filters:")
        # st.caption("Apply a common scope across every page. Filters run only when applied.")
        with st.form("portfolio_filters", border=False):
            selected_dimensions: dict[str, list[str]] = {}
            for group_label, attributes in FILTER_GROUPS:
                with st.expander(
                    group_label,
                    expanded=False,
                    # expanded=group_label == "Portfolio & contract",
                ):
                    for attribute in attributes:
                        selected_dimensions[attribute] = st.multiselect(
                            FILTER_LABELS[attribute],
                            options=options.get(FILTER_COLUMNS[attribute], []),
                            key=widget_keys[attribute],
                            placeholder="All",
                        )
            apply_col, reset_col = st.columns(2, gap="small")
            with apply_col:
                submitted = st.form_submit_button(
                    "Apply Filters",
                    # type="primary",
                    width="stretch",
                )
            with reset_col:
                st.form_submit_button(
                    "Reset Filters",
                    width="stretch",
                    on_click=_reset_filter_state,
                    args=(default,),
                )
        if submitted:
            applied = FilterSpec(
                year_start=default.year_start,
                year_end=default.year_end,
                **{
                    attribute: tuple(selected_dimensions[attribute])
                    for attribute in FILTER_COLUMNS
                },
            )
            st.session_state["applied_filters"] = applied
        # st.divider(width = "stretch")
        st.empty()
        # st.caption("Data controls")
        # st.caption("• Aggregates only leave the mart for Generative BI")
        # st.caption("• The raw source is not loaded into the browser")
        scope = _scope_text(applied) or "None"
        st.markdown(
            f"**Filters applied:**<br>&nbsp;&nbsp;{scope}",
            unsafe_allow_html=True,
            )


    return applied


def _render_header() -> None:
    st.markdown(
        '<div class="qbr-kicker">POLICY PORTFOLIO / BUSINESS REVIEW</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="qbr-title">Generative BI powered portfolio performance center</div>',
        unsafe_allow_html=True,
    )


def _render_top_entity_gross_premium(
    data: pd.DataFrame,
    *,
    entity_column: str,
    title: str,
    empty_message: str,
    chart_key: str,
    height: int = 450,
) -> None:
    """Render a top-ten gross premium comparison, stacked by network type."""

    required_columns = {entity_column, "network_type", "gross_premium_usd"}
    if data.empty or not required_columns.issubset(data.columns):
        st.info(empty_message)
        return

    prepared = data.loc[:, [entity_column, "network_type", "gross_premium_usd"]].copy()
    prepared[entity_column] = prepared[entity_column].fillna("Unassigned").astype(str)
    prepared["network_type"] = prepared["network_type"].fillna("Unassigned").astype(str)
    prepared["gross_premium_usd"] = pd.to_numeric(
        prepared["gross_premium_usd"], errors="coerce"
    ).fillna(0)
    totals = (
        prepared.groupby(entity_column, as_index=False)["gross_premium_usd"]
        .sum()
        .sort_values("gross_premium_usd", ascending=False)
        .head(10)
    )
    if totals.empty:
        st.info(empty_message)
        return

    entity_order = totals[entity_column].tolist()
    stacked = prepared.loc[prepared[entity_column].isin(entity_order)].copy()
    network_type_order = (
        stacked.groupby("network_type", as_index=True)["gross_premium_usd"]
        .sum()
        .sort_values(ascending=False)
        .index.astype(str)
        .tolist()
    )
    figure = px.bar(
        stacked,
        x="gross_premium_usd",
        y=entity_column,
        color="network_type",
        orientation="h",
        barmode="stack",
        category_orders={
            entity_column: entity_order,
            "network_type": network_type_order,
        },
        color_discrete_sequence=STACKED_TREND_COLORS,
    )
    figure.add_scatter(
        x=totals["gross_premium_usd"],
        y=totals[entity_column],
        mode="text",
        text=[
            f"{value:,.0f}" if pd.notna(value) else ""
            for value in totals["gross_premium_usd"]
        ],
        textposition="middle right",
        textfont=dict(color=PALETTE["slate"], size=10),
        hoverinfo="skip",
        showlegend=False,
        cliponaxis=False,
    )
    maximum_total = float(totals["gross_premium_usd"].max())
    figure.update_layout(
        height=height,
        margin=dict(l=0, r=10, t=88, b=0),
        title=dict(
            text=title,
            x=0,
            xanchor="left",
            font=dict(size=16, color=PALETTE["navy"]),
        ),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        xaxis_title="Gross premium (USD)",
        yaxis=dict(
            title="",
            categoryorder="array",
            categoryarray=entity_order,
            # Plotly draws horizontal categories bottom-to-top. Reverse the
            # axis so the descending data order is also descending visually.
            autorange="reversed",
        ),
        legend=dict( orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5,
            title=dict(
                text="Network type:",
                side="left",
                font=dict(size=11, color=PALETTE["slate"])
            ),
        ),
    )
    figure.update_xaxes(
        range=[0, maximum_total * 1.16] if maximum_total > 0 else [0, 1]
    )
    st.plotly_chart(
        figure,
        width="stretch",
        config={"displayModeBar": False},
        key=chart_key,
    )


def _render_overview(snapshot: dict[str, Any]) -> None:
    kpis = _latest_kpi_metrics(snapshot["monthly_kpis"])

    cards = st.columns(6)
    with cards[0]:
        _metric_card(
            "Active population",
            _formatted_metric(kpis["active_population"]["value"], _compact_number),
            kpis["active_population"]["mom"],
            kpis["active_population"]["yoy"],
        )
    with cards[1]:
        _metric_card(
            "Active Registered",
            _formatted_metric(kpis["active_registered_users"]["value"], _compact_number),
            kpis["active_registered_users"]["mom"],
            kpis["active_registered_users"]["yoy"],
        )
    with cards[2]:
        _metric_card(
            "App Penetration",
            _formatted_metric(
                kpis["app_penetration"]["value"], lambda value: f"{value:.1%}"
            ),
            kpis["app_penetration"]["mom"],
            kpis["app_penetration"]["yoy"],
        )
    with cards[3]:
        _metric_card(
            "Gross premium",
            _formatted_metric(kpis["gross_premium_usd"]["value"], _compact_money),
            kpis["gross_premium_usd"]["mom"],
            kpis["gross_premium_usd"]["yoy"],
        )
    with cards[4]:
        _metric_card(
            "Net premium",
            _formatted_metric(kpis["net_premium_usd"]["value"], _compact_money),
            kpis["net_premium_usd"]["mom"],
            kpis["net_premium_usd"]["yoy"],
        )
    with cards[5]:
        _metric_card(
            "TPA fee",
            _formatted_metric(kpis["tpa_fee_usd"]["value"], _compact_money),
            kpis["tpa_fee_usd"]["mom"],
            kpis["tpa_fee_usd"]["yoy"],
        )

    monthly_kpis = snapshot["monthly_kpis"].copy()
    monthly_kpis["month_end"] = pd.to_datetime(monthly_kpis["month_end"])
    monthly_kpis = monthly_kpis.sort_values("month_end").tail(24)
    monthly_kpis["app_penetration"] = monthly_kpis["active_registered_users"].div(
        monthly_kpis["active_population"].where(
            monthly_kpis["active_population"].ne(0)
        )
    )

    st.divider(width = "stretch")

    # st.markdown(
    #     '<div class="section-label">Active population, Registered Users and App Penetration</div>',
    #     unsafe_allow_html=True,
    # )

    annual = snapshot["premium_by_year"].copy()
    yearly_tab, chart_tab, table_tab, payer_tab, contract_tab = st.tabs(
        [
            "Yearly View",
            "Monthly Trend Chart",
            "Country Trend Table",
            "Top Payers",
            "Top Master Contracts",
        ],
        key="overview-portfolio-tabs",
    )
    with yearly_tab:
        member_column, premium_column = st.columns(2)
        with member_column:
            _render_yearly_executive_member_view(annual, height=450)
        with premium_column:
            _render_yearly_premium_view(
                annual,
                show_table=False,
                chart_key="executive_yearly_premium_view",
                height=450,
            )
    with chart_tab:
        if monthly_kpis.empty:
            st.info("No monthly population history is available for the current selection.")
        else:
            figure = go.Figure()
            figure.add_bar(
                name="Active population",
                x=monthly_kpis["month_end"],
                y=monthly_kpis["active_population"],
                marker_color=MONTHLY_TREND_COLORS["active_population"],
                opacity=0.82,
                offsetgroup="active_population",
                text=[_compact_number(value) for value in monthly_kpis["active_population"]],
                texttemplate="%{text}",
                textposition="outside",
                textfont=dict(color=MONTHLY_TREND_COLORS["active_population"]),
                cliponaxis=False,
                hovertemplate="%{x|%b %Y}<br><b>%{y:,.0f}</b> active beneficiaries<extra></extra>",
            )
            figure.add_bar(
                name="Registered users",
                x=monthly_kpis["month_end"],
                y=monthly_kpis["active_registered_users"],
                marker_color=MONTHLY_TREND_COLORS["registered_users"],
                opacity=0.88,
                offsetgroup="registered_users",
                text=[
                    _compact_number(value)
                    for value in monthly_kpis["active_registered_users"]
                ],
                texttemplate="%{text}",
                textposition="outside",
                textfont=dict(color=MONTHLY_TREND_COLORS["registered_users"]),
                cliponaxis=False,
                hovertemplate="%{x|%b %Y}<br><b>%{y:,.0f}</b> registered users<extra></extra>",
            )
            figure.add_scatter(
                name="App Penetration",
                x=monthly_kpis["month_end"],
                y=monthly_kpis["app_penetration"],
                yaxis="y2",
                mode="lines+markers+text",
                line=dict(color=MONTHLY_TREND_COLORS["app_penetration"], width=2.5),
                marker=dict(size=6),
                text=[
                    f"{value:.1%}" if pd.notna(value) else ""
                    for value in monthly_kpis["app_penetration"]
                ],
                textposition="top center",
                textfont=dict(
                    color=MONTHLY_TREND_COLORS["app_penetration"], size=10
                ),
                cliponaxis=False,
                hovertemplate="%{x|%b %Y}<br><b>%{y:.1%}</b> app penetration<extra></extra>",
            )
            figure.update_layout(
                barmode="group",
                bargap=0.16,
                bargroupgap=0,
                height=450,
                margin=dict(l=0, r=10, t=60, b=0),
                title=dict(
                    text="Month-end active population, registered users and app penetration",
                    x=0,
                    xanchor="left",
                    font=dict(size=16, color=PALETTE["navy"]),
                ),
                legend=dict( orientation="h", yanchor="bottom", y=0.97, xanchor="center", x=0.5 ),
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                xaxis=dict(
                    title="",
                    tickformat="%b<br>%Y",
                    dtick="M1",
                    tickangle=0,
                    tickfont=dict(size=12),
                ),
                yaxis=dict(title="Active population", rangemode="tozero"),
                yaxis2=dict(
                    title="App Penetration",
                    overlaying="y",
                    side="right",
                    tickformat=".0%",
                    range=[0, 1.09],
                    dtick=0.2,
                ),
            )
            st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})

    with table_tab:
        country_monthly = snapshot["monthly_country_kpis"].copy()
        country_monthly["month_end"] = pd.to_datetime(country_monthly["month_end"])
        display_months = pd.DatetimeIndex(
            monthly_kpis["month_end"].dropna().unique()
        ).sort_values()
        country_monthly = country_monthly.loc[
            country_monthly["month_end"].isin(display_months)
        ].copy()
        payer_countries = sorted(country_monthly["payer_country"].dropna().unique())
        if country_monthly.empty or display_months.empty or not payer_countries:
            st.info("No country-level population history is available for the current selection.")
        else:
            html = _country_month_matrix_html(country_monthly, display_months)
            st.html(
                f"""
                <div style="height:450px; overflow-y:auto; border:1px solid #ddd;">
                    {html}
                </div>
                """,
                width="stretch",
            )

    with payer_tab:
        _render_top_entity_gross_premium(
            snapshot.get("payer_network_premium", pd.DataFrame()),
            entity_column="payer_name",
            title="Top payers by gross premium",
            empty_message="No payer data for the current selection.",
            chart_key="overview_top_payers_gross_premium",
            height=450,
        )
    with contract_tab:
        _render_top_entity_gross_premium(
            snapshot.get("master_contract_network_premium", pd.DataFrame()),
            entity_column="master_contract",
            title="Top master contracts by gross premium",
            empty_message="No master contract data for the current selection.",
            chart_key="overview_top_master_contracts_gross_premium",
            height=450,
        )

    # st.markdown('<div class="section-label">Payer review</div>', unsafe_allow_html=True)
    # st.dataframe(
    #     _format_frame(
    #         snapshot["payer_review"].copy(),
    #         {"app_penetration_rate", "net_to_gross_ratio", "tpa_to_gross_ratio"},
    #     ),
    #     width="stretch",
    #     hide_index=True,
    #     height=360,
    # )


def _latest_population_dimension(
    frame: pd.DataFrame,
    *,
    dimension_column: str,
    latest_month: pd.Timestamp,
    metric_column: str = "active_population",
) -> pd.DataFrame:
    """Return the latest membership view for one portfolio dimension."""

    required_columns = {"month_end", dimension_column, metric_column}
    if frame.empty or not required_columns.issubset(frame.columns):
        return pd.DataFrame(columns=[dimension_column, metric_column])
    data = frame.copy()
    data["month_end"] = pd.to_datetime(data["month_end"]).dt.normalize()
    data[dimension_column] = data[dimension_column].fillna("Unassigned").astype(str)
    return (
        data.loc[
            data["month_end"].eq(latest_month),
            [dimension_column, metric_column],
        ]
        .groupby(dimension_column, as_index=False)[metric_column]
        .sum()
        .sort_values(metric_column, ascending=False)
        .reset_index(drop=True)
    )


def _render_latest_population_breakdown(
    frame: pd.DataFrame,
    *,
    dimension_column: str,
    latest_month: pd.Timestamp,
    title: str,
    max_items: int = 8,
    metric_column: str = "active_population",
    metric_label: str = "Active population",
    metric_value_label: str = "active beneficiaries",
    value_formatter: Callable[[Any], str] = _compact_number,
    height: int = TAB_CHART_HEIGHT,
    chart_key: str | None = None,
) -> None:
    """Render a compact current-month membership breakdown."""

    chart_data = _latest_population_dimension(
        frame,
        dimension_column=dimension_column,
        latest_month=latest_month,
        metric_column=metric_column,
    ).head(max_items)
    if chart_data.empty:
        st.caption(f"No {metric_label.lower()} breakdown is available for this scope.")
        return
    chart_data = chart_data.sort_values(metric_column, ascending=False)
    figure = px.bar(
        chart_data,
        x=metric_column,
        y=dimension_column,
        color=dimension_column,
        orientation="h",
        text=[value_formatter(value) for value in chart_data[metric_column]],
        color_discrete_sequence=STACKED_TREND_COLORS,
        category_orders={dimension_column: chart_data[dimension_column].tolist()},
    )
    figure.update_traces(
        textposition="outside",
        cliponaxis=False,
        hovertemplate=(
            f"<b>%{{y}}</b><br>%{{x:,.0f}} {metric_value_label}<extra></extra>"
        ),
    )
    figure.update_layout(
        height=height,
        margin=dict(l=0, r=10, t=58, b=0),
        title=dict(
            text=title,
            x=0,
            xanchor="left",
            font=dict(size=15, color=PALETTE["navy"]),
        ),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        showlegend=False,
        xaxis_title="",
        yaxis=dict(
            title="",
            categoryorder="array",
            categoryarray=chart_data[dimension_column].tolist(),
            # A horizontal category axis is drawn bottom-to-top. Reversing the
            # axis keeps the largest (first) category at the top of the chart.
            autorange="reversed",
        ),
    )
    st.plotly_chart(
        figure,
        width="stretch",
        config={"displayModeBar": False},
        key=chart_key,
    )


def _render_gross_premium_breakdown(
    snapshot: dict[str, Any],
    *,
    latest_month: pd.Timestamp,
    source_table: str,
    dimension_column: str,
    title: str,
    chart_key: str,
) -> None:
    """Render one latest-month gross-premium breakdown in a page tab."""

    _render_latest_population_breakdown(
        _dimension_trend_frame(
            snapshot,
            source_table,
            dimension_column,
            "gross_premium_usd",
        ),
        dimension_column=dimension_column,
        latest_month=latest_month,
        title=title,
        metric_column="gross_premium_usd",
        metric_label="Gross premium",
        metric_value_label="gross premium (USD)",
        value_formatter=_compact_money,
        height=TAB_CHART_HEIGHT,
        chart_key=chart_key,
    )


def _render_population(snapshot: dict[str, Any], filters: FilterSpec) -> None:
    monthly_kpis = snapshot["monthly_kpis"].copy()
    if monthly_kpis.empty:
        st.warning("No active memberships match the current scope.")
        return
    monthly_kpis["month_end"] = pd.to_datetime(monthly_kpis["month_end"]).dt.normalize()
    monthly_kpis = (
        monthly_kpis.sort_values("month_end")
        .drop_duplicates("month_end", keep="last")
        .reset_index(drop=True)
    )
    latest_month = pd.Timestamp(monthly_kpis.iloc[-1]["month_end"])
    kpis = _latest_kpi_metrics(monthly_kpis)
    country_monthly = snapshot["monthly_country_kpis"].copy()

    # st.markdown('<div class="qbr-kicker">POPULATION / MEMBER LENS</div>', unsafe_allow_html=True)
    # st.subheader("Active population", anchor=False)
    # st.caption(
    #     "Distinct beneficiary keys active on the final calendar day of each month. "
    #     f"Reporting month: {latest_month:%b %Y} · Scope: {_scope_text(filters) or 'All portfolio'}"
    # )

    # st.markdown('<div class="section-label">Population health</div>', unsafe_allow_html=True)
    cards = st.columns(3)
    with cards[0]:
        _metric_card(
            "Active population",
            _formatted_metric(kpis["active_population"]["value"], _compact_number),
            kpis["active_population"]["mom"],
            kpis["active_population"]["yoy"],
        )
    with cards[1]:
        _metric_card(
            "Active Registered",
            _formatted_metric(kpis["active_registered_users"]["value"], _compact_number),
            kpis["active_registered_users"]["mom"],
            kpis["active_registered_users"]["yoy"],
        )
    with cards[2]:
        _metric_card(
            "App Penetration",
            _formatted_metric(
                kpis["app_penetration"]["value"], lambda value: f"{value:.1%}"
            ),
            kpis["app_penetration"]["mom"],
            kpis["app_penetration"]["yoy"],
        )

    st.divider(width="stretch")
    st.markdown('<div class="section-label">Population trends</div>', unsafe_allow_html=True)
    (
        yearly_tab,
        monthly_tab,
        network_tab,
        network_group_tab,
        policy_type_tab,
        gross_premium_country_tab,
        gross_premium_network_tab,
        gross_premium_policy_tab,
        population_country_tab,
        population_network_tab,
        population_policy_tab,
    ) = st.tabs(
        [
            "Yearly View",
            "Monthly Trend Chart",
            "Network Trend Chart",
            "Network Group Trend Chart",
            "Policy Type Trend Chart",
            "GP by Payer Country",
            "GP by Network Type",
            "GP by Policy Type",
            "Active Population by Payer Country",
            "Active Population by Network Type",
            "Active Population by Policy Type",
        ],
        key="active-population-portfolio-tabs",
    )
    with yearly_tab:
        _render_yearly_metric_view(
            snapshot["premium_by_year"].copy(),
            metric_column="beneficiaries",
            metric_label="Active population",
            metric_value_label="active beneficiaries",
            title="Active population by underwriting year",
            bar_color=MONTHLY_TREND_COLORS["active_population"],
            height=TAB_CHART_HEIGHT,
        )
    with monthly_tab:
        _render_monthly_active_population_trend(
            monthly_kpis,
            height=TAB_CHART_HEIGHT,
        )
    with network_tab:
        _render_stacked_active_population_trend(
            _dimension_trend_frame(
                snapshot, "monthly_network_type_kpis", "network_type"
            ),
            dimension_column="network_type",
            title="Month-end active population by network type",
            height=TAB_CHART_HEIGHT,
        )
    with network_group_tab:
        _render_stacked_active_population_trend(
            _dimension_trend_frame(
                snapshot, "monthly_network_group_kpis", "network_group"
            ),
            dimension_column="network_group",
            title="Month-end active population by network group",
            height=TAB_CHART_HEIGHT,
        )
    with policy_type_tab:
        _render_stacked_active_population_trend(
            _dimension_trend_frame(
                snapshot, "monthly_policy_type_kpis", "policy_type"
            ),
            dimension_column="policy_type",
            title="Month-end active population by policy type",
            height=TAB_CHART_HEIGHT,
        )
    with gross_premium_country_tab:
        _render_gross_premium_breakdown(
            snapshot,
            latest_month=latest_month,
            source_table="monthly_country_kpis",
            dimension_column="payer_country",
            title="Gross premium by payer country",
            chart_key="active-population-portfolio-tabs-gp-payer-country",
        )
    with gross_premium_network_tab:
        _render_gross_premium_breakdown(
            snapshot,
            latest_month=latest_month,
            source_table="monthly_network_type_kpis",
            dimension_column="network_type",
            title="Gross premium by network type",
            chart_key="active-population-portfolio-tabs-gp-network-type",
        )
    with gross_premium_policy_tab:
        _render_gross_premium_breakdown(
            snapshot,
            latest_month=latest_month,
            source_table="monthly_policy_type_kpis",
            dimension_column="policy_type",
            title="Gross premium by policy type",
            chart_key="active-population-portfolio-tabs-gp-policy-type",
        )
    with population_country_tab:
        _render_latest_population_breakdown(
            country_monthly,
            dimension_column="payer_country",
            latest_month=latest_month,
            title="Active population by payer country",
            height=TAB_CHART_HEIGHT,
            chart_key="active-population-portfolio-tabs-population-payer-country",
        )
    with population_network_tab:
        _render_latest_population_breakdown(
            _dimension_trend_frame(
                snapshot, "monthly_network_type_kpis", "network_type"
            ),
            dimension_column="network_type",
            latest_month=latest_month,
            title="Active population by network type",
            height=TAB_CHART_HEIGHT,
            chart_key="active-population-portfolio-tabs-population-network-type",
        )
    with population_policy_tab:
        _render_latest_population_breakdown(
            _dimension_trend_frame(
                snapshot, "monthly_policy_type_kpis", "policy_type"
            ),
            dimension_column="policy_type",
            latest_month=latest_month,
            title="Active population by policy type",
            height=TAB_CHART_HEIGHT,
            chart_key="active-population-portfolio-tabs-population-policy-type",
        )



def _render_mobile(snapshot: dict[str, Any]) -> None:
    monthly_kpis = snapshot["monthly_kpis"].copy()
    if monthly_kpis.empty:
        st.warning("No active registrations match the current scope.")
        return
    monthly_kpis["month_end"] = pd.to_datetime(monthly_kpis["month_end"]).dt.normalize()
    monthly_kpis = (
        monthly_kpis.sort_values("month_end")
        .drop_duplicates("month_end", keep="last")
        .reset_index(drop=True)
    )
    latest_month = pd.Timestamp(monthly_kpis.iloc[-1]["month_end"])
    kpis = _latest_kpi_metrics(monthly_kpis)
    country_monthly = snapshot["monthly_country_kpis"].copy()

    cards = st.columns(3)
    with cards[0]:
        _metric_card(
            "Active Registered",
            _formatted_metric(kpis["active_registered_users"]["value"], _compact_number),
            kpis["active_registered_users"]["mom"],
            kpis["active_registered_users"]["yoy"],
        )
    with cards[1]:
        _metric_card(
            "Active population",
            _formatted_metric(kpis["active_population"]["value"], _compact_number),
            kpis["active_population"]["mom"],
            kpis["active_population"]["yoy"],
        )
    with cards[2]:
        _metric_card(
            "App Penetration",
            _formatted_metric(
                kpis["app_penetration"]["value"], lambda value: f"{value:.1%}"
            ),
            kpis["app_penetration"]["mom"],
            kpis["app_penetration"]["yoy"],
        )

    st.divider(width="stretch")
    st.markdown(
        '<div class="section-label">Active registered trends</div>',
        unsafe_allow_html=True,
    )
    (
        yearly_tab,
        monthly_tab,
        network_tab,
        network_group_tab,
        policy_type_tab,
        gross_premium_country_tab,
        gross_premium_network_tab,
        gross_premium_policy_tab,
        registered_country_tab,
        registered_network_tab,
        registered_policy_tab,
    ) = st.tabs(
        [
            "Yearly View",
            "Monthly Trend Chart",
            "Network Trend Chart",
            "Network Group Trend Chart",
            "Policy Type Trend Chart",
            "GP by Payer Country",
            "GP by Network Type",
            "GP by Policy Type",
            "Active Registered by Payer Country",
            "Active Registered by Network Type",
            "Active Registered by Policy Type",
        ],
        key="active-registered-portfolio-tabs",
    )
    with yearly_tab:
        _render_yearly_metric_view(
            snapshot["premium_by_year"].copy(),
            metric_column="registered_users",
            metric_label="Active Registered",
            metric_value_label="active registered users",
            title="Active registered users by underwriting year",
            bar_color=MONTHLY_TREND_COLORS["registered_users"],
            height=TAB_CHART_HEIGHT,
        )
    with monthly_tab:
        _render_monthly_active_population_trend(
            monthly_kpis,
            metric_column="active_registered_users",
            metric_label="Active Registered",
            metric_value_label="active registered users",
            title="Month-end active registered users",
            bar_color=MONTHLY_TREND_COLORS["registered_users"],
            height=TAB_CHART_HEIGHT,
        )
    with network_tab:
        _render_stacked_active_population_trend(
            _dimension_trend_frame(
                snapshot,
                "monthly_network_type_kpis",
                "network_type",
                "active_registered_users",
            ),
            dimension_column="network_type",
            title="Month-end active registered users by network type",
            metric_column="active_registered_users",
            metric_label="Active Registered",
            metric_value_label="active registered users",
            height=TAB_CHART_HEIGHT,
        )
    with network_group_tab:
        _render_stacked_active_population_trend(
            _dimension_trend_frame(
                snapshot,
                "monthly_network_group_kpis",
                "network_group",
                "active_registered_users",
            ),
            dimension_column="network_group",
            title="Month-end active registered users by network group",
            metric_column="active_registered_users",
            metric_label="Active Registered",
            metric_value_label="active registered users",
            height=TAB_CHART_HEIGHT,
        )
    with policy_type_tab:
        _render_stacked_active_population_trend(
            _dimension_trend_frame(
                snapshot,
                "monthly_policy_type_kpis",
                "policy_type",
                "active_registered_users",
            ),
            dimension_column="policy_type",
            title="Month-end active registered users by policy type",
            metric_column="active_registered_users",
            metric_label="Active Registered",
            metric_value_label="active registered users",
            height=TAB_CHART_HEIGHT,
        )
    with gross_premium_country_tab:
        _render_gross_premium_breakdown(
            snapshot,
            latest_month=latest_month,
            source_table="monthly_country_kpis",
            dimension_column="payer_country",
            title="Gross premium by payer country",
            chart_key="active-registered-portfolio-tabs-gp-payer-country",
        )
    with gross_premium_network_tab:
        _render_gross_premium_breakdown(
            snapshot,
            latest_month=latest_month,
            source_table="monthly_network_type_kpis",
            dimension_column="network_type",
            title="Gross premium by network type",
            chart_key="active-registered-portfolio-tabs-gp-network-type",
        )
    with gross_premium_policy_tab:
        _render_gross_premium_breakdown(
            snapshot,
            latest_month=latest_month,
            source_table="monthly_policy_type_kpis",
            dimension_column="policy_type",
            title="Gross premium by policy type",
            chart_key="active-registered-portfolio-tabs-gp-policy-type",
        )
    with registered_country_tab:
        _render_latest_population_breakdown(
            country_monthly,
            dimension_column="payer_country",
            latest_month=latest_month,
            title="Active registered users by payer country",
            metric_column="active_registered_users",
            metric_label="Active Registered",
            metric_value_label="active registered users",
            height=TAB_CHART_HEIGHT,
            chart_key="active-registered-portfolio-tabs-registered-payer-country",
        )
    with registered_network_tab:
        _render_latest_population_breakdown(
            _dimension_trend_frame(
                snapshot,
                "monthly_network_type_kpis",
                "network_type",
                "active_registered_users",
            ),
            dimension_column="network_type",
            latest_month=latest_month,
            title="Active registered users by network type",
            metric_column="active_registered_users",
            metric_label="Active Registered",
            metric_value_label="active registered users",
            height=TAB_CHART_HEIGHT,
            chart_key="active-registered-portfolio-tabs-registered-network-type",
        )
    with registered_policy_tab:
        _render_latest_population_breakdown(
            _dimension_trend_frame(
                snapshot,
                "monthly_policy_type_kpis",
                "policy_type",
                "active_registered_users",
            ),
            dimension_column="policy_type",
            latest_month=latest_month,
            title="Active registered users by policy type",
            metric_column="active_registered_users",
            metric_label="Active Registered",
            metric_value_label="active registered users",
            height=TAB_CHART_HEIGHT,
            chart_key="active-registered-portfolio-tabs-registered-policy-type",
        )


def _render_premium(snapshot: dict[str, Any]) -> None:
    monthly_kpis = snapshot["monthly_kpis"].copy()
    if monthly_kpis.empty:
        st.warning("No premium records match the current scope.")
        return
    monthly_kpis["month_end"] = pd.to_datetime(monthly_kpis["month_end"]).dt.normalize()
    monthly_kpis = (
        monthly_kpis.sort_values("month_end")
        .drop_duplicates("month_end", keep="last")
        .reset_index(drop=True)
    )
    latest_month = pd.Timestamp(monthly_kpis.iloc[-1]["month_end"])
    kpis = _latest_kpi_metrics(monthly_kpis)

    cards = st.columns(3)
    with cards[0]:
        _metric_card(
            "GP",
            _formatted_metric(kpis["gross_premium_usd"]["value"], _compact_money),
            kpis["gross_premium_usd"]["mom"],
            kpis["gross_premium_usd"]["yoy"],
        )
    with cards[1]:
        _metric_card(
            "NP",
            _formatted_metric(kpis["net_premium_usd"]["value"], _compact_money),
            kpis["net_premium_usd"]["mom"],
            kpis["net_premium_usd"]["yoy"],
        )
    with cards[2]:
        _metric_card(
            "TPA Fee",
            _formatted_metric(kpis["tpa_fee_usd"]["value"], _compact_money),
            kpis["tpa_fee_usd"]["mom"],
            kpis["tpa_fee_usd"]["yoy"],
        )

    st.divider(width="stretch")
    st.markdown('<div class="section-label">Premium trends</div>', unsafe_allow_html=True)
    (
        yearly_tab,
        monthly_tab,
        network_tab,
        network_group_tab,
        policy_type_tab,
        gross_premium_country_tab,
        gross_premium_network_tab,
        gross_premium_policy_tab,
    ) = st.tabs(
        [
            "Yearly View",
            "Monthly Trend Chart",
            "Network Trend Chart",
            "Network Group Trend Chart",
            "Policy Type Trend Chart",
            "GP by Payer Country",
            "GP by Network Type",
            "GP by Policy Type",
        ],
        key="premium-portfolio-tabs",
    )
    with yearly_tab:
        _render_yearly_premium_view(
            snapshot["premium_by_year"].copy(),
            height=TAB_CHART_HEIGHT,
        )
    with monthly_tab:
        _render_monthly_premium_trend(monthly_kpis, height=TAB_CHART_HEIGHT)
    with network_tab:
        _render_stacked_active_population_trend(
            _dimension_trend_frame(
                snapshot,
                "monthly_network_type_kpis",
                "network_type",
                "gross_premium_usd",
            ),
            dimension_column="network_type",
            title="Month-end gross premium by network type",
            metric_column="gross_premium_usd",
            metric_label="Gross premium",
            metric_value_label="gross premium (USD)",
            value_formatter=_compact_money,
            height=TAB_CHART_HEIGHT,
        )
    with network_group_tab:
        _render_stacked_active_population_trend(
            _dimension_trend_frame(
                snapshot,
                "monthly_network_group_kpis",
                "network_group",
                "gross_premium_usd",
            ),
            dimension_column="network_group",
            title="Month-end gross premium by network group",
            metric_column="gross_premium_usd",
            metric_label="Gross premium",
            metric_value_label="gross premium (USD)",
            value_formatter=_compact_money,
            height=TAB_CHART_HEIGHT,
        )
    with policy_type_tab:
        _render_stacked_active_population_trend(
            _dimension_trend_frame(
                snapshot,
                "monthly_policy_type_kpis",
                "policy_type",
                "gross_premium_usd",
            ),
            dimension_column="policy_type",
            title="Month-end gross premium by policy type",
            metric_column="gross_premium_usd",
            metric_label="Gross premium",
            metric_value_label="gross premium (USD)",
            value_formatter=_compact_money,
            height=TAB_CHART_HEIGHT,
        )
    with gross_premium_country_tab:
        _render_gross_premium_breakdown(
            snapshot,
            latest_month=latest_month,
            source_table="monthly_country_kpis",
            dimension_column="payer_country",
            title="Gross premium by payer country",
            chart_key="premium-portfolio-tabs-gp-payer-country",
        )
    with gross_premium_network_tab:
        _render_gross_premium_breakdown(
            snapshot,
            latest_month=latest_month,
            source_table="monthly_network_type_kpis",
            dimension_column="network_type",
            title="Gross premium by network type",
            chart_key="premium-portfolio-tabs-gp-network-type",
        )
    with gross_premium_policy_tab:
        _render_gross_premium_breakdown(
            snapshot,
            latest_month=latest_month,
            source_table="monthly_policy_type_kpis",
            dimension_column="policy_type",
            title="Gross premium by policy type",
            chart_key="premium-portfolio-tabs-gp-policy-type",
        )

_EVIDENCE_PERCENT_COLUMNS = {
    "app_penetration_rate",
    "net_to_gross_ratio",
    "tpa_to_gross_ratio",
    "registered_user_penetration",
    "linked_beneficiary_coverage",
}


def _render_question_evidence(evidence: QuestionEvidence) -> None:
    with st.expander("Evidence used for this answer", expanded=False):
        st.caption(
            f"Focus: {evidence.focus} · Sources: "
            f"{', '.join(evidence_table_label(source) for source in evidence.source_tables)}"
        )
        for limitation in evidence.limitations:
            st.warning(limitation)
        for source, frame in evidence.tables.items():
            st.markdown(f"**{evidence_table_label(source)}**")
            percent_columns = _EVIDENCE_PERCENT_COLUMNS.intersection(frame.columns)
            st.dataframe(
                _format_frame(frame.copy(), percent_columns),
                width="stretch",
                hide_index=True,
                height=min(300, 74 + 35 * max(len(frame), 1)),
            )


def _render_ollama_serve_status(
    status_slot: Any,
    status: OllamaRuntimeStatus,
) -> None:
    """Present passive Ollama serve health without invoking the configured model."""

    availability = "Available" if status.is_ready else "Unavailable"
    indicator = "🟢" if status.is_ready else "🟠"
    status_slot.caption(
        f"{indicator} **LLM {availability}:** {status.message}"
    )


def _apply_gen_bi_suggestion() -> None:
    """Prefill the Generative BI form from a selected native suggestion pill."""

    suggested_question = st.session_state.get(GEN_BI_SUGGESTION_KEY)
    if suggested_question:
        st.session_state[GEN_BI_QUESTION_KEY] = str(suggested_question)


def _render_gen_bi(
    snapshot: dict[str, Any],
    filters: FilterSpec,
    entity_catalog: dict[str, list[Any]],
    evaluation_dir: Path,
) -> None:
    ollama_config = get_ollama_config(ROOT / ".env")
    # Remove the persisted result produced by the retired response-check control.
    st.session_state.pop("gen_bi_ollama_model_status", None)

    st.subheader("Generative BI — Insights", anchor=False)
    st.caption(
        "Ask a focused business question. A deterministic semantic layer selects only relevant, "
        "aggregate-only evidence; the configured Ollama model turns it into a concise business narrative."
    )
    ollama_status_slot = st.empty()
    _render_ollama_serve_status(
        ollama_status_slot,
        _cached_ollama_serve_status(ollama_config.host),
    )

    st.pills(
        "**Executive generative BI question examples:**",
        GEN_BI_SUGGESTIONS,
        selection_mode="single",
        key=GEN_BI_SUGGESTION_KEY,
        on_change=_apply_gen_bi_suggestion,
    )
    with st.form("gen_bi_question", border=False):
        question = st.text_area(
            "**Ask your data specific question:**",
            placeholder="Which payer needs attention, why, and what should we do next?",
            height=120,
            max_chars=800,
            key=GEN_BI_QUESTION_KEY,
        )
        st.caption(
            "Choose a example question above or write your own payer, master-contract, or age-bucket demographic question."
        )
        submitted = st.form_submit_button(
            "Generate Insights",
            type="primary",
            width="content",
        )
    # st.caption(
    #     "Every submitted interaction is saved locally as an aggregate-only Parquet evaluation record."
    # )

    if not submitted:
        return
    if not question.strip():
        st.warning("Enter a focused business question before generating an answer.")
        return

    scope = _scope_text(filters) or "All portfolio"
    planning_started = time.perf_counter()
    try:
        evidence = build_question_evidence(
            question,
            snapshot,
            scope,
            entity_catalog=entity_catalog,
        )
    except Exception as exc:
        st.error(f"The question-to-evidence step could not be completed: {exc}")
        return
    planning_ms = round((time.perf_counter() - planning_started) * 1000, 1)

    answer: str | None = None
    response_status = "success"
    response_engine = f"{OLLAMA_ENGINE}:{ollama_config.model}"
    model_check_status = "narrative_responded"
    error_message: str | None = None
    response_ms: float | None = None
    model_status: OllamaModelStatus | None = None
    answer_started = time.perf_counter()
    with st.spinner("Ensuring the configured Ollama runtime is available..."):
        runtime_status = ensure_ollama(ollama_config.host)
    _cached_ollama_serve_status.clear()
    _render_ollama_serve_status(ollama_status_slot, runtime_status)
    if runtime_status.is_ready:
        try:
            with st.spinner(f"Generating a business response with model: {ollama_config.model}..."):
                answer = _cached_ollama_answer(
                    ollama_config.host,
                    ollama_config.model,
                    evidence.question,
                    evidence.context_json,
                    OLLAMA_REPRODUCIBILITY_PROFILE,
                )
        except Exception as exc:
            with st.spinner("Preparing the evidence-bound fallback..."):
                model_status = check_ollama_model_response(
                    ollama_config.host,
                    ollama_config.model,
                )
            model_check_status = (
                "simple_probe_responded"
                if model_status.is_responding
                else "simple_probe_failed"
            )
            error_message = (
                f"Generative BI model call failed: {exc} Ollama diagnostic: {model_status.message}"
            )
    else:
        model_status = OllamaModelStatus(
            is_responding=False,
            message=runtime_status.message,
        )
        model_check_status = "runtime_unavailable"
        error_message = (
            f"Generative BI runtime is unavailable: {runtime_status.message}"
        )

    if answer is None:
        try:
            answer = generate_executive_answer(evidence)
            response_status = "fallback"
            response_engine = DETERMINISTIC_ENGINE
        except Exception as fallback_exc:
            response_status = "failed"
            error_message = f"{error_message} Deterministic fallback failed: {fallback_exc}"
    response_ms = round((time.perf_counter() - answer_started) * 1000, 1)

    try:
        record_path = record_evaluation(
            evaluation_dir=evaluation_dir,
            evidence=evidence,
            answer=answer,
            response_engine=response_engine,
            response_status=response_status,
            configured_model=ollama_config.model,
            generation_profile=OLLAMA_REPRODUCIBILITY_PROFILE,
            model_check_status=model_check_status,
            filter_spec_json=json.dumps(filters.as_dict(), sort_keys=True),
            planning_ms=planning_ms,
            response_ms=response_ms,
            dashboard_query_ms=snapshot.get("query_ms"),
            error_message=error_message,
        )
    except Exception as exc:
        record_path = None
        st.warning(f"The answer could not be recorded to the evaluation dataset: {exc}")

    if answer is None:
        st.error(f"The Generative BI response could not be completed: {error_message}")
    else:
        if response_status == "fallback":
            st.warning(
                "The configured model did not produce a Generative BI response. "
                f"{model_status.message if model_status else error_message} "
                "Showing the evidence-bound deterministic fallback instead."
            )
        st.markdown('<div class="section-label">Insight:</div>', unsafe_allow_html=True)
        st.markdown(answer)

    if record_path is not None:
        st.caption(
            f"Evaluation record saved: `{record_path.name}` · Engine `{response_engine}` · "
            f"Evidence planning {planning_ms:,.1f} ms · Response {response_ms:,.1f} ms"
        )
    _render_question_evidence(evidence)



def _render_data_guide(snapshot: dict[str, Any]) -> None:
    metadata = snapshot["metadata"]
    st.subheader("Metric definitions and performance guardrails", anchor=False)
    st.markdown(
        """
        - **Active population:** distinct `beneficiarykey` where member start date is on or before a calendar month end and stop date is on or after it.
        - **Mobile App penetration:** distinct `registereduserkey` divided by distinct `beneficiarykey`, as requested. The dashboard separately reports beneficiary linkage for interpretation.
        - **KPI-card MoM / YoY:** calculated against the prior calendar month and the same month one year earlier. GP / NP / TPA are evenly allocated across each policy's active month-end coverage months for these comparisons; detailed premium tables remain policy-year sums.
        - **Generative BI:** a deterministic semantic layer maps each question to relevant aggregate evidence, then the configured Ollama model produces the business narrative. It never writes SQL or receives beneficiary-level rows. Generation uses a fixed seed and greedy sampling profile, so identical question, scope, evidence, and model build return the same response. The page shows a passive Ollama serve status; a local endpoint is started only when an insight is requested, remote endpoints are only checked, and models are never pulled automatically. If a narration fails, an internal diagnostic precedes the evidence-bound fallback. Each submitted interaction is saved as an aggregate-only Parquet evaluation record.
        """
    )
    st.markdown('<div class="section-label">Mart metadata</div>', unsafe_allow_html=True)
    metadata_frame = pd.DataFrame(
        [{"Property": key, "Value": value} for key, value in metadata.items()]
    )
    st.dataframe(metadata_frame, width="stretch", hide_index=True)
    st.caption(
        f"Latest dashboard query: {snapshot['query_ms']:,.1f} ms. This measures DuckDB aggregation only, "
        "not browser rendering or the deterministic Generative BI response step."
    )


def main() -> None:
    _inject_css()
    db_path = Path(
        os.getenv("POLICY_MART_PATH", str(ROOT / "data" / "policy_mart.duckdb"))
    ).expanduser()
    if not db_path.exists():
        st.title("Policy Portfolio QBR")
        st.warning("The analytics mart has not been built yet.")
        st.code("python prepare_data.py --source .\\data\\policies.csv", language="powershell")
        st.markdown(
            "Stage the 3M-row CSV/Parquet extract in `data/`, then run the one-time mart builder. "
            "The dashboard reads the resulting DuckDB file rather than the raw extract."
        )
        return
    try:
        modified_ns = db_path.stat().st_mtime_ns
        metadata = get_mart_metadata(db_path)
        options = _cached_filter_options(str(db_path), modified_ns)
    except Exception as exc:
        st.error(f"Unable to read the policy mart: {exc}")
        st.info("Rebuild it with prepare_data.py after checking source columns and date formats.")
        return
    filters = _render_sidebar(options)
    filter_json = json.dumps(filters.as_dict(), sort_keys=True)
    try:
        with st.spinner("Refreshing portfolio view…"):
            snapshot = _cached_snapshot(
                str(db_path),
                modified_ns,
                filter_json,
                SNAPSHOT_CACHE_VERSION,
            )
            if any(table not in snapshot for table in REQUIRED_SNAPSHOT_TABLES):
                # Recover from a Streamlit hot reload that retained a snapshot
                # generated before a new aggregate was added to query_dashboard.
                _cached_snapshot.clear()
                snapshot = query_dashboard(db_path, filters)
    except Exception as exc:
        st.error(f"Unable to query the policy mart: {exc}")
        st.info("Rebuild it with prepare_data.py after checking source columns and date formats.")
        return
    if not metadata:
        st.error("The mart metadata is missing. Rebuild the mart before continuing.")
        return
    _render_header()
    tabs = st.tabs(
        ["Overview", "Active Population", "Mobile App", "Premium & economics", "Generative BI", "Data guide"], key="page-portfolio-tabs",
    )
    with tabs[0]:
        _render_overview(snapshot)
    with tabs[1]:
        _render_population(snapshot, filters)
    with tabs[2]:
        _render_mobile(snapshot)
    with tabs[3]:
        _render_premium(snapshot)
    with tabs[4]:
        _render_gen_bi(
            snapshot,
            filters,
            options,
            GEN_BI_EVALUATION_DIR,
        )
    with tabs[5]:
        _render_data_guide(snapshot)


if __name__ == "__main__":
    main()
