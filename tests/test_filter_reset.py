from __future__ import annotations

import json
from pathlib import Path
import sys

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from policy_dashboard.data import FILTER_COLUMNS, FilterSpec, build_mart
from test_data_layer import SAMPLE


def test_reset_filters_clears_widget_values_and_applied_scope(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "policies.csv"
    mart = tmp_path / "policy_mart.duckdb"
    source.write_text(SAMPLE, encoding="utf-8")
    build_mart(source, mart)
    monkeypatch.setenv("POLICY_MART_PATH", str(mart))

    app = AppTest.from_file(str(ROOT / "app.py"))
    app.run(timeout=30)
    tab_labels = [tab.label for tab in app.tabs]
    assert "Monthly Trend Chart" in tab_labels
    assert "Network Trend Chart" in tab_labels
    assert "Network Group Trend Chart" in tab_labels
    assert "Policy Type Trend Chart" in tab_labels
    assert "Country Trend Table" in tab_labels
    overview_tabs = [
        "Yearly View",
        "Monthly Trend Chart",
        "Country Trend Table",
        "Top Payers",
        "Top Master Contracts",
    ]
    population_page_tabs = [
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
    ]
    mobile_page_tabs = [
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
    ]
    premium_page_tabs = [
        "Yearly View",
        "Monthly Trend Chart",
        "Network Trend Chart",
        "Network Group Trend Chart",
        "Policy Type Trend Chart",
        "GP by Payer Country",
        "GP by Network Type",
        "GP by Policy Type",
    ]
    assert tab_labels.count("Yearly View") == 4
    assert any(
        tab_labels[index : index + len(overview_tabs)] == overview_tabs
        for index in range(len(tab_labels))
    )
    assert all(
        any(
            tab_labels[index : index + len(page_tabs)] == page_tabs
            for index in range(len(tab_labels))
        )
        for page_tabs in (population_page_tabs, mobile_page_tabs, premium_page_tabs)
    )
    assert not any(
        tab_labels[index : index + 3] == ["Payer Country", "Network Type", "Policy Type"]
        for index in range(len(tab_labels))
    )
    trend_specs = [
        json.loads(chart.proto.spec)
        for chart in app.get("plotly_chart")
        if chart.proto.spec
        and json.loads(chart.proto.spec)
        .get("layout", {})
        .get("title", {})
        .get("text", "")
        .startswith("Month-end active population by ")
    ]
    assert len(trend_specs) == 3
    for trend_spec in trend_specs:
        assert trend_spec["layout"]["height"] == 425
        bars = [
            trace
            for trace in trend_spec["data"]
            if trace.get("type") == "bar"
        ]
        assert bars
        assert all(trace.get("textposition") == "inside" for trace in bars)
        assert all(trace.get("insidetextanchor") == "end" for trace in bars)
        total_trace = next(
            trace
            for trace in trend_spec["data"]
            if trace.get("name") == "Total active population"
        )
        assert total_trace["textposition"] == "top center"
    registered_trend_specs = [
        json.loads(chart.proto.spec)
        for chart in app.get("plotly_chart")
        if chart.proto.spec
        and json.loads(chart.proto.spec)
        .get("layout", {})
        .get("title", {})
        .get("text", "")
        .startswith("Month-end active registered users by ")
    ]
    assert len(registered_trend_specs) == 3
    for trend_spec in registered_trend_specs:
        assert trend_spec["layout"]["height"] == 425
        total_trace = next(
            trace
            for trace in trend_spec["data"]
            if trace.get("name") == "Total active registered"
        )
        assert total_trace["textposition"] == "top center"
    premium_trend_specs = [
        json.loads(chart.proto.spec)
        for chart in app.get("plotly_chart")
        if chart.proto.spec
        and json.loads(chart.proto.spec)
        .get("layout", {})
        .get("title", {})
        .get("text", "")
        .startswith("Month-end gross premium by ")
    ]
    assert len(premium_trend_specs) == 3
    for trend_spec in premium_trend_specs:
        assert trend_spec["layout"]["height"] == 425
        total_trace = next(
            trace
            for trace in trend_spec["data"]
            if trace.get("name") == "Total gross premium"
        )
        assert total_trace["textposition"] == "top center"
    chart_titles = {
        json.loads(chart.proto.spec)
        .get("layout", {})
        .get("title", {})
        .get("text")
        for chart in app.get("plotly_chart")
        if chart.proto.spec
    }
    assert {
        "Active population by payer country",
        "Active population by network type",
        "Active population by policy type",
        "Month-end active population",
        "Active registered users by payer country",
        "Active registered users by network type",
        "Active registered users by policy type",
        "Month-end active registered users",
        "Gross premium, net premium and TPA fee by underwriting year",
        "Month-end gross premium, net premium and TPA fee",
        "Gross premium by payer country",
        "Gross premium by network type",
        "Gross premium by policy type",
        "Active population by underwriting year",
        "Active registered users by underwriting year",
        "Active population, Active Registered and App Penetration by underwriting year",
        "Top payers by gross premium",
        "Top master contracts by gross premium",
    }.issubset(chart_titles)
    assert "Premium and TPA fee by underwriting year" not in chart_titles
    overview_chart_titles = {
        "Active population, Active Registered and App Penetration by underwriting year",
        "Month-end active population, registered users and app penetration",
        "Top payers by gross premium",
        "Top master contracts by gross premium",
    }
    for chart in app.get("plotly_chart"):
        if not chart.proto.spec:
            continue
        chart_spec = json.loads(chart.proto.spec)
        title = chart_spec.get("layout", {}).get("title", {}).get("text")
        if title in overview_chart_titles:
            assert chart_spec["layout"]["height"] == 450
    yearly_premium_heights = [
        json.loads(chart.proto.spec)["layout"]["height"]
        for chart in app.get("plotly_chart")
        if chart.proto.spec
        and json.loads(chart.proto.spec)
        .get("layout", {})
        .get("title", {})
        .get("text")
        == "Gross premium, net premium and TPA fee by underwriting year"
    ]
    assert sorted(yearly_premium_heights) == [425, 450]
    latest_breakdown_titles = {
        "Gross premium by payer country",
        "Gross premium by network type",
        "Gross premium by policy type",
        "Active population by payer country",
        "Active population by network type",
        "Active population by policy type",
        "Active registered users by payer country",
        "Active registered users by network type",
        "Active registered users by policy type",
    }
    latest_breakdown_specs = [
        json.loads(chart.proto.spec)
        for chart in app.get("plotly_chart")
        if chart.proto.spec
        and json.loads(chart.proto.spec)
        .get("layout", {})
        .get("title", {})
        .get("text")
        in latest_breakdown_titles
    ]
    assert len(latest_breakdown_specs) == 15
    for breakdown_spec in latest_breakdown_specs:
        assert breakdown_spec["layout"]["height"] == 425
        category_order = breakdown_spec["layout"]["yaxis"]["categoryarray"]
        assert category_order
        assert breakdown_spec["layout"]["yaxis"]["autorange"] == "reversed"
    population_monthly_trend_spec = next(
        json.loads(chart.proto.spec)
        for chart in app.get("plotly_chart")
        if chart.proto.spec
        and json.loads(chart.proto.spec)
        .get("layout", {})
        .get("title", {})
        .get("text")
        == "Month-end active population"
    )
    assert [trace["name"] for trace in population_monthly_trend_spec["data"]] == [
        "Active population"
    ]
    assert population_monthly_trend_spec["data"][0]["type"] == "bar"
    assert population_monthly_trend_spec["data"][0]["marker"]["color"] == "#00A6A6"
    assert population_monthly_trend_spec["layout"]["height"] == 425
    registered_monthly_trend_spec = next(
        json.loads(chart.proto.spec)
        for chart in app.get("plotly_chart")
        if chart.proto.spec
        and json.loads(chart.proto.spec)
        .get("layout", {})
        .get("title", {})
        .get("text")
        == "Month-end active registered users"
    )
    assert [trace["name"] for trace in registered_monthly_trend_spec["data"]] == [
        "Active Registered"
    ]
    assert registered_monthly_trend_spec["data"][0]["type"] == "bar"
    assert registered_monthly_trend_spec["data"][0]["marker"]["color"] == "#11263E"
    assert registered_monthly_trend_spec["layout"]["height"] == 425
    yearly_population_spec = next(
        json.loads(chart.proto.spec)
        for chart in app.get("plotly_chart")
        if chart.proto.spec
        and json.loads(chart.proto.spec)
        .get("layout", {})
        .get("title", {})
        .get("text")
        == "Active population by underwriting year"
    )
    assert [trace["name"] for trace in yearly_population_spec["data"]] == [
        "Active population"
    ]
    assert yearly_population_spec["layout"]["height"] == 425
    yearly_registered_spec = next(
        json.loads(chart.proto.spec)
        for chart in app.get("plotly_chart")
        if chart.proto.spec
        and json.loads(chart.proto.spec)
        .get("layout", {})
        .get("title", {})
        .get("text")
        == "Active registered users by underwriting year"
    )
    assert [trace["name"] for trace in yearly_registered_spec["data"]] == [
        "Active Registered"
    ]
    assert yearly_registered_spec["layout"]["height"] == 425
    executive_yearly_spec = next(
        json.loads(chart.proto.spec)
        for chart in app.get("plotly_chart")
        if chart.proto.spec
        and json.loads(chart.proto.spec)
        .get("layout", {})
        .get("title", {})
        .get("text")
        == "Active population, Active Registered and App Penetration by underwriting year"
    )
    assert [trace["name"] for trace in executive_yearly_spec["data"]] == [
        "Active population",
        "Active Registered",
        "App Penetration",
    ]
    yearly_premium_page_spec = next(
        json.loads(chart.proto.spec)
        for chart in app.get("plotly_chart")
        if chart.proto.spec
        and json.loads(chart.proto.spec)
        .get("layout", {})
        .get("title", {})
        .get("text")
        == "Gross premium, net premium and TPA fee by underwriting year"
    )
    assert [trace["name"] for trace in yearly_premium_page_spec["data"]] == [
        "Gross premium",
        "Net premium",
        "TPA fee",
    ]
    assert {
        trace["name"]: trace["marker"]["color"]
        for trace in yearly_premium_page_spec["data"]
    } == {
        "Gross premium": "#00A6A6",
        "Net premium": "#11263E",
        "TPA fee": "#E9795D",
    }
    monthly_premium_page_spec = next(
        json.loads(chart.proto.spec)
        for chart in app.get("plotly_chart")
        if chart.proto.spec
        and json.loads(chart.proto.spec)
        .get("layout", {})
        .get("title", {})
        .get("text")
        == "Month-end gross premium, net premium and TPA fee"
    )
    assert [trace["name"] for trace in monthly_premium_page_spec["data"]] == [
        "Gross premium",
        "Net premium",
        "TPA fee",
    ]
    assert monthly_premium_page_spec["layout"]["height"] == 425
    assert {
        "Active population trajectory",
        "App penetration within active population",
        "Month-on-month movement by payer country",
    }.isdisjoint(chart_titles)
    assert not any(
        "Leading payer country" in markdown.value for markdown in app.markdown
    )
    payer_trend_spec = next(
        json.loads(chart.proto.spec)
        for chart in app.get("plotly_chart")
        if chart.proto.spec
        and json.loads(chart.proto.spec)
        .get("layout", {})
        .get("title", {})
        .get("text")
        == "Top payers by gross premium"
    )
    assert payer_trend_spec["layout"]["barmode"] == "stack"
    payer_legend = payer_trend_spec["layout"]["legend"]
    assert payer_legend["orientation"] == "h"
    assert payer_legend["x"] == 0.5
    assert payer_legend["xanchor"] == "center"
    assert payer_legend["y"] == 1.05
    assert payer_legend["yanchor"] == "bottom"
    assert payer_legend["title"]["text"] == "Network type:"
    assert payer_trend_spec["layout"]["yaxis"]["categoryarray"]
    assert payer_trend_spec["layout"]["yaxis"]["autorange"] == "reversed"
    assert any(
        trace.get("type") == "bar" and trace.get("name") == "GN"
        for trace in payer_trend_spec["data"]
    )
    master_contract_trend_spec = next(
        json.loads(chart.proto.spec)
        for chart in app.get("plotly_chart")
        if chart.proto.spec
        and json.loads(chart.proto.spec)
        .get("layout", {})
        .get("title", {})
        .get("text")
        == "Top master contracts by gross premium"
    )
    assert master_contract_trend_spec["layout"]["barmode"] == "stack"
    master_contract_legend = master_contract_trend_spec["layout"]["legend"]
    assert master_contract_legend["orientation"] == "h"
    assert master_contract_legend["x"] == 0.5
    assert master_contract_legend["xanchor"] == "center"
    assert master_contract_legend["y"] == 1.05
    assert master_contract_legend["yanchor"] == "bottom"
    assert master_contract_legend["title"]["text"] == "Network type:"
    assert master_contract_trend_spec["layout"]["yaxis"]["categoryarray"]
    assert master_contract_trend_spec["layout"]["yaxis"]["autorange"] == "reversed"
    assert any(
        trace.get("type") == "bar" and trace.get("name") == "GN"
        for trace in master_contract_trend_spec["data"]
    )
    top_premium_titles = [
        json.loads(chart.proto.spec).get("layout", {}).get("title", {}).get("text")
        for chart in app.get("plotly_chart")
        if chart.proto.spec
        and json.loads(chart.proto.spec).get("layout", {}).get("title", {}).get("text")
        in {"Top payers by gross premium", "Top master contracts by gross premium"}
    ]
    assert top_premium_titles == [
        "Top payers by gross premium",
        "Top master contracts by gross premium",
    ]
    assert "Local Ollama model" not in [
        selectbox.label for selectbox in app.selectbox
    ]
    assert "Generative BI — Insights" in [
        subheader.value for subheader in app.subheader
    ]
    assert (
        "Choose a example question above or write your own payer, master-contract, or age-bucket demographic question."
    ) in [caption.value for caption in app.caption]
    assert "Generate Insights" in [button.label for button in app.button]
    assert "Check model response" not in [button.label for button in app.button]
    assert "Generate CXO answer" not in [button.label for button in app.button]
    assert any("LLM" in caption.value for caption in app.caption)
    assert not any("Configured model:" in caption.value for caption in app.caption)
    suggestion_pill = app.pills(key="gen_bi_suggested_question")
    assert len(suggestion_pill.options) == 9
    assert suggestion_pill.options[0] == (
        "Top active‑population drivers and the action to prioritise."
    )
    suggestion_pill.select(suggestion_pill.options[0])
    app.run(timeout=30)
    assert app.text_area(key="gen_bi_question_text").value == suggestion_pill.options[0]
    dashboard_css = next(
        markdown.value
        for markdown in app.markdown
        if "sidebar collapse/expand control" in markdown.value
    )
    assert 'button[kind="header"] { display: none !important; }' not in dashboard_css
    assert '[data-testid="stToolbar"] { display: none !important; }' not in dashboard_css
    assert '[data-testid="stAppDeployButton"] { display: none !important; }' in dashboard_css
    assert '.st-key-overview-portfolio-tabs [data-baseweb="tab"]' in dashboard_css
    assert "font-size: 14px;" in dashboard_css
    assert "font-weight: 700;" in dashboard_css
    assert any(
        "Filters applied:" in markdown.value and "None" in markdown.value
        for markdown in app.sidebar.markdown
    )
    assert not app.sidebar.select_slider
    assert len(app.sidebar.multiselect) == len(FILTER_COLUMNS)
    assert [expander.label for expander in app.sidebar.expander] == [
        "Portfolio & contract",
        "Policy & network",
        "Member profile",
        "Digital adoption",
    ]
    country_matrix = next(
        element.proto.body
        for element in app.get("html")
        if "policy-country-matrix" in element.proto.body
    )
    assert country_matrix.count('scope="colgroup"') == 24
    assert country_matrix.count('title="Active population">AP</abbr>') == 24
    assert country_matrix.count('title="Active Registered">AR</abbr>') == 24
    assert country_matrix.count('title="App Penetration">Pen</abbr>') == 24
    assert "height:450px" in country_matrix
    assert "height: 450px;" in country_matrix
    assert "background: #00A6A6;" in country_matrix
    assert "color: #002B5C;" in country_matrix
    assert "vertical-align: middle;" in country_matrix
    assert "font-variant-numeric: tabular-nums;" in country_matrix
    app.sidebar.multiselect[0].set_value(["Egypt"])
    app.sidebar.multiselect[1].set_value(["Allianz"])
    app.sidebar.multiselect[18].set_value(["Lumi"])
    app.sidebar.button[0].click()
    app.run(timeout=30)

    applied = app.session_state["applied_filters"]
    assert applied.payer_countries == ("Egypt",)
    assert applied.payers == ("Allianz",)
    assert applied.app_names == ("Lumi",)
    assert (applied.year_start, applied.year_end) == (2024, 2026)
    assert any(
        "payer country: Egypt" in markdown.value
        and "payer: Allianz" in markdown.value
        and "app name: Lumi" in markdown.value
        for markdown in app.sidebar.markdown
    )

    app.sidebar.button[1].click()
    app.run(timeout=30)

    assert app.session_state["portfolio_filter_reset_epoch"] == 1
    assert not app.sidebar.select_slider
    assert app.sidebar.multiselect[0].value == []
    assert app.sidebar.multiselect[1].value == []
    assert app.sidebar.multiselect[18].value == []
    assert not app.exception


def test_removed_year_slider_discards_legacy_year_scope(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "policies.csv"
    mart = tmp_path / "policy_mart.duckdb"
    source.write_text(SAMPLE, encoding="utf-8")
    build_mart(source, mart)
    monkeypatch.setenv("POLICY_MART_PATH", str(mart))

    app = AppTest.from_file(str(ROOT / "app.py"))
    app.run(timeout=30)

    # A prior release could retain both a widget value and an applied
    # year-only scope in an open browser tab. The removed control must not
    # leave an invisible historical filter behind.
    app.session_state["portfolio_filter_years_0"] = 2024
    app.session_state["applied_filters"] = FilterSpec(
        year_start=2025,
        year_end=2025,
        payers=("Allianz",),
    )
    app.run(timeout=30)

    applied = app.session_state["applied_filters"]
    assert (applied.year_start, applied.year_end) == (2024, 2026)
    assert applied.payers == ("Allianz",)
    assert not app.sidebar.select_slider
    assert not app.error
    assert not app.exception
