from __future__ import annotations

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
    assert "Country Trend Table" in tab_labels
    assert "Local Ollama model" not in [
        selectbox.label for selectbox in app.selectbox
    ]
    assert any(
        markdown.value == "**Filters applied:** None"
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
    assert "background: #003781;" in country_matrix
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
