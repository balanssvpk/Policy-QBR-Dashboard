"""CXO-level Policy QBR dashboard."""

from __future__ import annotations

from html import escape
import json
import os
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
ALLIANZ_LOGO_PATH = ROOT / "AZ_Partners_Attached_Descriptor_Positive_RGB_.png"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from policy_dashboard.data import (
    FILTER_COLUMNS,
    FilterSpec,
    get_filter_options,
    get_mart_metadata,
    query_dashboard,
)
from policy_dashboard.gen_bi import (
    OllamaConfig,
    OllamaServiceStatus,
    ask_ollama,
    available_models,
    build_context,
    deterministic_insight,
    ensure_ollama_server,
    get_ollama_config,
)


st.set_page_config(
    page_title="Policy Portfolio QBR",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = {
    "navy": "#11263E",
    "teal": "#00A6A6",
    "mint": "#8ED6C8",
    "amber": "#F2B84B",
    "coral": "#E9795D",
    "mist": "#E9EEF2",
    "slate": "#607084",
}

#Remove Deply and the three-dots menu
st.markdown("""
    <style>
        /* Remove the three-dots menu */
        #MainMenu {visibility: hidden;}

        /* Remove the Deploy button */
        .stDeployButton {display: none !important;}

        /* Optional: remove Streamlit header and footer */
        header {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

FILTER_WIDGET_KEYS = {
    **{
        attribute: f"portfolio_filter_{attribute}"
        for attribute in FILTER_COLUMNS
    },
}
FILTER_RESET_EPOCH_KEY = "portfolio_filter_reset_epoch"

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
          .stTabs [data-baseweb="tab-list"] { gap: 1.25rem; border-bottom: 1px solid #D9E0E5; font-weight: 900; font-size: 18px }
          .stTabs [data-baseweb="tab"] { height: 42px; padding: 0; color: #607084; font-weight: 900; font-size: 18px }
          .stTabs [aria-selected="true"] { color: #11263E; border-bottom-color: #00A6A6; }
          .stButton > button { border-radius: 6px; font-weight: 650; }
          [data-testid="stDataFrame"] { border: 1px solid #E1E6EA; border-radius: 8px; overflow: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def _cached_filter_options(db_path: str, modified_ns: int) -> dict[str, list[Any]]:
    del modified_ns
    return get_filter_options(db_path)


@st.cache_data(show_spinner=False)
def _cached_snapshot(db_path: str, modified_ns: int, filters_json: str) -> dict[str, Any]:
    del modified_ns
    return query_dashboard(db_path, FilterSpec.from_dict(json.loads(filters_json)))


@st.cache_data(show_spinner=False, ttl=900)
def _cached_ollama_answer(host: str, model: str, question: str, context: str) -> str:
    return ask_ollama(host=host, model=model, question=question, context=context)


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
        max-height: 420px;
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
        background: #003781;
        letter-spacing: 0.02em;
      }}
      .policy-country-matrix thead tr:nth-child(2) th {{
        top: 33px;
        color: #003781;
        background: #E4F3FA;
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
        border-right: 2px solid #0067B1;
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
    <p class="policy-country-matrix__key">AP = Active Population &nbsp;&middot;&nbsp; AR = Active Registered &nbsp;&middot;&nbsp; Pen = App Penetration</p>
    """


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
        st.markdown("### Data Filters:")
        # st.caption("Apply a common scope across every page. Filters run only when applied.")
        with st.form("portfolio_filters", border=False):
            selected_dimensions: dict[str, list[str]] = {}
            for group_label, attributes in FILTER_GROUPS:
                with st.expander(
                    group_label,
                    expanded=group_label == "Portfolio & contract",
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
                    type="primary",
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
        st.divider()
        st.caption("Data controls")
        st.caption("• Aggregates only leave the mart for Gen BI")
        st.caption("• The raw source is not loaded into the browser")
        scope = _scope_text(applied) or "None"
        st.markdown(f"**Filters applied:** {scope}")
    return applied


def _render_header() -> None:
    st.markdown(
        '<div class="qbr-kicker">POLICY PORTFOLIO / QUARTERLY BUSINESS REVIEW</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="qbr-title">Executive policy command centre</div>',
        unsafe_allow_html=True,
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
    st.markdown(
        '<div class="section-label">Active population, Registered Users and App Penetration</div>',
        unsafe_allow_html=True,
    )
    chart_tab, table_tab = st.tabs(["Monthly Trend Chart", "Country Trend Table"])
    with chart_tab:
        if monthly_kpis.empty:
            st.info("No monthly population history is available for the current selection.")
        else:
            figure = go.Figure()
            figure.add_bar(
                name="Active population",
                x=monthly_kpis["month_end"],
                y=monthly_kpis["active_population"],
                marker_color=PALETTE["teal"],
                opacity=0.82,
                offsetgroup="active_population",
                text=[_compact_number(value) for value in monthly_kpis["active_population"]],
                texttemplate="%{text}",
                textposition="outside",
                textfont=dict(color=PALETTE["teal"]),
                cliponaxis=False,
                hovertemplate="%{x|%b %Y}<br><b>%{y:,.0f}</b> active beneficiaries<extra></extra>",
            )
            figure.add_bar(
                name="Registered users",
                x=monthly_kpis["month_end"],
                y=monthly_kpis["active_registered_users"],
                marker_color=PALETTE["navy"],
                opacity=0.88,
                offsetgroup="registered_users",
                text=[
                    _compact_number(value)
                    for value in monthly_kpis["active_registered_users"]
                ],
                texttemplate="%{text}",
                textposition="outside",
                textfont=dict(color=PALETTE["navy"]),
                cliponaxis=False,
                hovertemplate="%{x|%b %Y}<br><b>%{y:,.0f}</b> registered users<extra></extra>",
            )
            figure.add_scatter(
                name="App Penetration",
                x=monthly_kpis["month_end"],
                y=monthly_kpis["app_penetration"],
                yaxis="y2",
                mode="lines+markers+text",
                line=dict(color=PALETTE["coral"], width=2.5),
                marker=dict(size=6),
                text=[
                    f"{value:.1%}" if pd.notna(value) else ""
                    for value in monthly_kpis["app_penetration"]
                ],
                textposition="top center",
                textfont=dict(color=PALETTE["coral"], size=10),
                cliponaxis=False,
                hovertemplate="%{x|%b %Y}<br><b>%{y:.1%}</b> app penetration<extra></extra>",
            )
            figure.update_layout(
                barmode="group",
                bargap=0.16,
                bargroupgap=0,
                height=390,
                margin=dict(l=0, r=10, t=28, b=0),
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0
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
                yaxis=dict(title="Active population", rangemode="tozero"),
                yaxis2=dict(
                    title="App Penetration",
                    overlaying="y",
                    side="right",
                    tickformat=".0%",
                    range=[0, 1],
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
            st.html(
                _country_month_matrix_html(country_monthly, display_months),
                width="stretch",
            )

    st.markdown('<div class="section-label">Portfolio trajectory</div>', unsafe_allow_html=True)
    left, right = st.columns((1.25, 1))
    annual = snapshot["premium_by_year"].copy()
    with left:
        if annual.empty:
            st.info("No premium data for the current selection.")
        else:
            melted = annual.melt(
                id_vars="uw_year",
                value_vars=["gross_premium_usd", "net_premium_usd", "tpa_fee_usd"],
                var_name="metric",
                value_name="usd",
            )
            melted["metric"] = melted["metric"].map(
                {
                    "gross_premium_usd": "Gross premium",
                    "net_premium_usd": "Net premium",
                    "tpa_fee_usd": "TPA fee",
                }
            )
            figure = px.line(
                melted,
                x="uw_year",
                y="usd",
                color="metric",
                markers=True,
                color_discrete_map={
                    "Gross premium": PALETTE["navy"],
                    "Net premium": PALETTE["teal"],
                    "TPA fee": PALETTE["amber"],
                },
            )
            figure.update_layout(
                height=390,
                margin=dict(l=0, r=10, t=25, b=0),
                legend_title_text="",
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                yaxis_title="USD",
                xaxis_title="UW year",
            )
            st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
    with right:
        payers = snapshot["payer_review"].head(8).copy()
        if payers.empty:
            st.info("No payer data for the current selection.")
        else:
            figure = px.bar(
                payers.sort_values("gross_premium_usd"),
                x="gross_premium_usd",
                y="payer_name",
                orientation="h",
                color_discrete_sequence=[PALETTE["teal"]],
            )
            figure.update_layout(
                height=315,
                margin=dict(l=0, r=10, t=25, b=0),
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                xaxis_title="Gross premium (USD)",
                yaxis_title="",
            )
            st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})

    st.markdown('<div class="section-label">Payer review</div>', unsafe_allow_html=True)
    st.dataframe(
        _format_frame(
            snapshot["payer_review"].copy(),
            {"app_penetration_rate", "net_to_gross_ratio", "tpa_to_gross_ratio"},
        ),
        width="stretch",
        hide_index=True,
        height=360,
    )


def _render_population(snapshot: dict[str, Any]) -> None:
    population = snapshot["active_population"].copy()
    st.subheader("Active population at each month end", anchor=False)
    st.caption(
        "Distinct beneficiary keys active on the final calendar day of each month, "
        "based on member start and stop dates."
    )
    if population.empty:
        st.warning("No active memberships match the current scope.")
        return
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=population["month_end"],
            y=population["active_population"],
            mode="lines+markers",
            line=dict(color=PALETTE["teal"], width=3),
            marker=dict(size=6),
            fill="tozeroy",
            fillcolor="rgba(0, 166, 166, 0.12)",
            hovertemplate="%{x|%b %Y}<br><b>%{y:,.0f}</b> active beneficiaries<extra></extra>",
        )
    )
    figure.update_layout(
        height=390,
        margin=dict(l=0, r=10, t=10, b=0),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        xaxis_title="Month end",
        yaxis_title="Active beneficiaries",
        showlegend=False,
    )
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
    table = population.rename(
        columns={"month_end": "Month end", "active_population": "Active population"}
    )
    st.dataframe(_format_frame(table), width="stretch", hide_index=True, height=310)


def _render_mobile(snapshot: dict[str, Any]) -> None:
    summary = snapshot["summary"].iloc[0]
    rate = _percent(summary["registered_users"], summary["unique_beneficiaries"])
    linked_rate = _percent(summary["registered_beneficiaries"], summary["unique_beneficiaries"])
    st.subheader("Mobile App penetration", anchor=False)
    st.caption(
        "Primary KPI = distinct registereduserkey ÷ distinct beneficiarykey. "
        "Linked-beneficiary coverage is shown alongside it because a registered user can represent more than one beneficiary."
    )
    cols = st.columns(4)
    cols[0].metric("Unique beneficiaries", _compact_number(summary["unique_beneficiaries"]))
    cols[1].metric("Unique registered users", _compact_number(summary["registered_users"]))
    cols[2].metric("Registered-user penetration", f"{rate:.1%}" if rate is not None else "—")
    cols[3].metric("Linked-beneficiary coverage", f"{linked_rate:.1%}" if linked_rate is not None else "—")

    mobile = snapshot["mobile_by_payer"].copy()
    if not mobile.empty:
        chart_data = mobile.sort_values("registered_user_penetration", ascending=True).tail(15)
        figure = px.bar(
            chart_data,
            x="registered_user_penetration",
            y="payer_name",
            orientation="h",
            color_discrete_sequence=[PALETTE["coral"]],
            labels={"registered_user_penetration": "Registered-user penetration", "payer_name": ""},
        )
        figure.update_xaxes(tickformat=".0%")
        figure.update_layout(
            height=390,
            margin=dict(l=0, r=10, t=25, b=0),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
        )
        st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
    st.markdown('<div class="section-label">Adoption table by payer</div>', unsafe_allow_html=True)
    st.dataframe(
        _format_frame(
            mobile,
            {"registered_user_penetration", "linked_beneficiary_coverage"},
        ),
        width="stretch",
        hide_index=True,
        height=400,
    )


def _render_premium(snapshot: dict[str, Any]) -> None:
    st.subheader("GP, NP and TPA fee evaluation", anchor=False)
    st.caption("All amounts are USD. Ratios are calculated only when gross premium is non-zero.")
    annual = snapshot["premium_by_year"].copy()
    if not annual.empty:
        figure = go.Figure()
        figure.add_bar(
            name="Gross premium",
            x=annual["uw_year"],
            y=annual["gross_premium_usd"],
            marker_color=PALETTE["navy"],
        )
        figure.add_bar(
            name="Net premium",
            x=annual["uw_year"],
            y=annual["net_premium_usd"],
            marker_color=PALETTE["teal"],
        )
        figure.add_bar(
            name="TPA fee",
            x=annual["uw_year"],
            y=annual["tpa_fee_usd"],
            marker_color=PALETTE["amber"],
        )
        figure.update_layout(
            barmode="group",
            height=360,
            margin=dict(l=0, r=10, t=15, b=0),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            yaxis_title="USD",
            xaxis_title="UW year",
            legend_title_text="",
        )
        st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
        st.dataframe(_format_frame(annual), width="stretch", hide_index=True)
    st.markdown('<div class="section-label">Policy mix</div>', unsafe_allow_html=True)
    st.dataframe(
        _format_frame(snapshot["policy_type_review"].copy()),
        width="stretch",
        hide_index=True,
        height=320,
    )


def _render_gen_bi(
    snapshot: dict[str, Any],
    filters: FilterSpec,
    ollama_config: OllamaConfig,
    ollama_status: OllamaServiceStatus,
) -> None:
    st.subheader("Gen BI — payer and policy review", anchor=False)
    st.caption(
        "The metric engine calculates the evidence in DuckDB first. Ollama receives only aggregate tables; "
        "it cannot generate or execute SQL."
    )

    question = st.text_area(
        "Ask question:",
        placeholder="Which payer needs the most attention, and what action should we take?",
        height=100,
    )

    with st.expander("Instant evidence-led portfolio readout", expanded=True):
        try:
            st.markdown(deterministic_insight(snapshot))
        except Exception as exc:
            st.error(f"Insight generation failed: {exc}")

    host = ollama_config.host
    configured_model = ollama_config.model
    models = available_models(host) if ollama_status.is_available else []
    model_is_available = configured_model in models

    if ollama_status.is_available and model_is_available:
        st.caption(
            f"Using configured local model `{configured_model}`. The model is kept warm for 30 minutes; "
            "identical questions and scope are cached for 15 minutes."
        )
    elif ollama_status.is_available:
        st.warning(
            f"Configured model `{configured_model}` is not installed. Run `ollama pull {configured_model}` "
            "to enable narrative answers."
        )
    else:
        st.info(ollama_status.message)

    scope = _scope_text(filters)
    if not scope:
        scope = "None"

    if st.button(
        "Generate insights",
        type="primary",
        disabled=not bool(question.strip()) or not model_is_available,
    ):
        if not model_is_available:
            st.error("The configured local Ollama model is not available yet.")
            return

        context = build_context(question, snapshot, scope)

        try:
            with st.spinner("Drafting a concise, evidence-backed answer locally…"):
                answer = _cached_ollama_answer(
                    host, configured_model, question, context
                )
            st.markdown(answer)
        except Exception as exc:
            st.error(f"Ollama could not complete the request: {exc}")

    st.markdown('<div class="section-label">Evidence sent to the narrative layer</div>', unsafe_allow_html=True)

    evidence_columns = [
        "payer_name",
        "beneficiaries",
        "gross_premium_usd",
        "net_premium_usd",
        "tpa_fee_usd",
        "app_penetration_rate",
    ]

    evidence = snapshot["payer_review"].loc[:, evidence_columns]

    st.dataframe(
        _format_frame(evidence, {"app_penetration_rate"}),
        width="stretch",
        hide_index=True,
        height=300,
    )



def _render_data_guide(snapshot: dict[str, Any]) -> None:
    metadata = snapshot["metadata"]
    st.subheader("Metric definitions and performance guardrails", anchor=False)
    st.markdown(
        """
        - **Active population:** distinct `beneficiarykey` where member start date is on or before a calendar month end and stop date is on or after it.
        - **Mobile App penetration:** distinct `registereduserkey` divided by distinct `beneficiarykey`, as requested. The dashboard separately reports beneficiary linkage for interpretation.
        - **KPI-card MoM / YoY:** calculated against the prior calendar month and the same month one year earlier. GP / NP / TPA are evenly allocated across each policy's active month-end coverage months for these comparisons; detailed premium tables remain policy-year sums.
        - **Gen BI:** a fixed metric layer produces aggregate evidence; Ollama narrates it and is never allowed to write SQL or receive beneficiary-level rows.
        """
    )
    st.markdown('<div class="section-label">Mart metadata</div>', unsafe_allow_html=True)
    metadata_frame = pd.DataFrame(
        [{"Property": key, "Value": value} for key, value in metadata.items()]
    )
    st.dataframe(metadata_frame, width="stretch", hide_index=True)
    st.caption(
        f"Latest dashboard query: {snapshot['query_ms']:,.1f} ms. This measures DuckDB aggregation only, "
        "not browser rendering or Ollama inference."
    )


def main() -> None:
    _inject_css()
    ollama_config = get_ollama_config(ROOT / ".env")
    ollama_status = ensure_ollama_server(ollama_config.host)
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
            snapshot = _cached_snapshot(str(db_path), modified_ns, filter_json)
    except Exception as exc:
        st.error(f"Unable to query the policy mart: {exc}")
        st.info("Rebuild it with prepare_data.py after checking source columns and date formats.")
        return
    if not metadata:
        st.error("The mart metadata is missing. Rebuild the mart before continuing.")
        return
    _render_header()
    tabs = st.tabs(
        ["Executive overview", "Population", "Mobile App", "Premium & economics", "Gen BI", "Data guide"]
    )
    with tabs[0]:
        _render_overview(snapshot)
    with tabs[1]:
        _render_population(snapshot)
    with tabs[2]:
        _render_mobile(snapshot)
    with tabs[3]:
        _render_premium(snapshot)
    with tabs[4]:
        _render_gen_bi(snapshot, filters, ollama_config, ollama_status)
    with tabs[5]:
        _render_data_guide(snapshot)


if __name__ == "__main__":
    main()
