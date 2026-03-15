"""Streamlit entry point for futures seasonal analytics."""

from __future__ import annotations

import inspect
import tomllib
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import streamlit as st
import yaml
import plotly.graph_objects as go

from engine import FuturesComboEngine, StrategySpec, LegSpec, FillPolicy
from data_provider import (
    load_dataset,
    append_bloomberg_snapshot_live,
    DataSource,
    BloombergPullConfig,
    CacheConfig,
    default_cache_dir,
)
from ui_components import (
    plot_lifecycle_overlay,
    resolve_display_years,
    build_lifecycle_heatmap_table,
    bucket_sort_key,
    compact_heatmap_table,
    to_period_log_return,
    plot_lifecycle_heatmap,
    lifecycle_stats_at_asof,
    summarize_forward_trade_metrics,
)


APP_DIR = Path(__file__).resolve().parent
DEFAULT_STRATEGIES_PATH = APP_DIR / "strategies.yaml"
APP_NAME = "Seasonal Forge"
APP_TAGLINE = "Forged futures seasonality, spreads, and heatmaps"


def _read_project_version() -> str:
    pyproject_path = APP_DIR / "pyproject.toml"
    try:
        raw = pyproject_path.read_text(encoding="utf-8")
        data = tomllib.loads(raw)
        return str(data.get("project", {}).get("version", "0.0.0"))
    except Exception:
        return "0.0.0"


APP_VERSION = _read_project_version()
CATEGORY_ORDER = ["Grains", "Oilseeds", "Energy", "Biofuels", "Examples", "Other"]
VALUE_SOURCE_LABELS = {
    "contract": "Futures contract value",
    "calculated": "Calculated from legs",
}
STRATEGY_CARD_STYLE = """
<style>
.strategy-card {
    border: 1px solid rgba(255, 75, 75, 0.45);
    background: rgba(255, 75, 75, 0.10);
    padding: 0.55rem 0.65rem;
    border-radius: 0.6rem;
    margin-bottom: 0.45rem;
}
.strategy-card .title {
    font-weight: 700;
    color: #ff6b6b;
    margin-bottom: 0.15rem;
}
.strategy-card .sub {
    font-size: 0.85rem;
    opacity: 0.9;
}
</style>
"""


# -------------------------
# Caching
# -------------------------

@st.cache_data(show_spinner=False)
def load_strategies_yaml(path: str, reload_token: int) -> Tuple[Dict[str, StrategySpec], Dict[str, str]]:
    """
    Load strategies from YAML. reload_token busts cache when user clicks "Reload strategies".
    """
    _ = reload_token
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Strategies file not found: {p}")

    cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    items = cfg.get("strategies", [])
    specs: Dict[str, StrategySpec] = {}
    categories: Dict[str, str] = {}

    for item in items:
        legs = [LegSpec(**leg) for leg in item["legs"]]
        raw_value_source = item.get("value_source")
        if raw_value_source is None:
            raw_value_source = "calculated" if item.get("expression") else "contract"

        spec = StrategySpec(
            name=item["name"],
            legs=legs,
            expression=item.get("expression"),
            output_currency=item.get("output_currency", "USD"),
            min_obs=int(item.get("min_obs", 5)),
            value_source=str(raw_value_source),
        )
        specs[spec.name] = spec
        categories[spec.name] = str(item.get("category", "Other")).strip() or "Other"

    if not specs:
        raise ValueError("No strategies found in YAML. Expected: strategies: [ ... ]")

    return specs, categories


@st.cache_data(show_spinner=False)
def load_data_cached(
    reload_token: int,
    *,
    source: DataSource,
    specs: List[StrategySpec],
    bbg_cfg: Optional[BloombergPullConfig],
    cache_cfg: Optional[CacheConfig],
    csv_path: Optional[str],
    parquet_path: Optional[str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load the raw wide dataset (wide dataframe).
    reload_token busts Streamlit cache when user clicks "Reload data".
    """
    res = load_dataset(
        reload_token,
        source=source,
        specs=specs,
        bbg_cfg=bbg_cfg,
        cache_cfg=cache_cfg,
        csv_path=csv_path,
        parquet_path=parquet_path,
    )
    return res.df, res.meta


@st.cache_resource(show_spinner=False)
def get_engine(df: pd.DataFrame) -> FuturesComboEngine:
    """Build and cache the parsing engine for the current dataset."""
    return FuturesComboEngine(df, date_col="date" if "date" in df.columns else None)


def overlay_lifecycle_window_compat(
    engine: FuturesComboEngine,
    curves: pd.DataFrame,
    *,
    window_months: Tuple[int, int],
    asof: pd.Timestamp,
    reference: str,
    axis: str,
    alignment: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Call overlay_lifecycle_window with backward compatibility for engines that
    do not yet expose the `alignment` keyword.
    """
    fn = engine.overlay_lifecycle_window
    params = inspect.signature(fn).parameters
    kwargs: Dict[str, Any] = {
        "window_months": window_months,
        "asof": asof,
        "reference": reference,
        "axis": axis,
    }
    has_alignment = "alignment" in params
    if has_alignment:
        kwargs["alignment"] = alignment

    matrix, calendar, meta = fn(curves, **kwargs)
    meta = dict(meta or {})
    if not has_alignment:
        meta.setdefault("alignment", "first_valid")
    return matrix, calendar, meta


def clear_streamlit_caches(*, reset_asof_input: bool = False) -> None:
    """Clear Streamlit caches and optionally reset the ASOF widget state."""
    st.cache_data.clear()
    st.cache_resource.clear()
    if reset_asof_input:
        st.session_state.pop("asof_date_input", None)


@st.cache_data(show_spinner=False)
def build_trading_insights_table(
    df_raw: pd.DataFrame,
    specs: Tuple[StrategySpec, ...],
    strategy_categories: Dict[str, str],
    *,
    asof: pd.Timestamp,
    window_months: Tuple[int, int],
    alignment: str,
    fill_enabled: bool,
    fill_limit: int,
) -> pd.DataFrame:
    """Compute cross-strategy forward seasonal metrics for the current ASOF/window."""
    engine = FuturesComboEngine(df_raw, date_col="date" if "date" in df_raw.columns else None)
    fill = FillPolicy(method="ffill", limit=int(fill_limit), apply_to_fx=True) if fill_enabled else None

    rows: List[Dict[str, Any]] = []
    for spec in specs:
        try:
            curves, report = engine.build_with_report(spec, fill=fill, multiindex_cols=False)
            curves = curves.dropna(how="all")
            if curves.empty:
                raise ValueError("No curve values available")

            matrix, calendar, meta = overlay_lifecycle_window_compat(
                engine,
                curves,
                window_months=(int(window_months[0]), int(window_months[1])),
                asof=pd.Timestamp(asof),
                reference="front",
                axis="t",
                alignment=alignment,
            )
            metrics = summarize_forward_trade_metrics(
                matrix,
                calendar,
                reference_anchor_year=int(meta["reference_anchor_year"]),
            )
            stats = lifecycle_stats_at_asof(
                matrix,
                calendar,
                int(meta["reference_anchor_year"]),
                [int(y) for y in matrix.columns],
            )
            ok_years = int((report["status"] == "ok").sum()) if not report.empty else 0

            row: Dict[str, Any] = {
                "strategy": spec.name,
                "category": strategy_categories.get(spec.name, "Other"),
                "value_source": VALUE_SOURCE_LABELS.get(spec.value_source, spec.value_source),
                "status": "OK" if metrics.get("ok") else "Insufficient sample",
                "status_detail": metrics.get("reason", ""),
                "anchor_years": ok_years,
                "sample": metrics.get("sample_size", 0),
                "bias": metrics.get("bias", "NA"),
                "success_pct": metrics.get("success_rate", np.nan),
                "up_pct": metrics.get("up_rate", np.nan),
                "avg_change": metrics.get("avg_change", np.nan),
                "median_change": metrics.get("median_change", np.nan),
                "avg_abs_change": metrics.get("avg_abs_change", np.nan),
                "mae_p75": metrics.get("mae_p75", np.nan),
                "mae_p90": metrics.get("mae_p90", np.nan),
                "reward_risk": metrics.get("reward_risk", np.nan),
                "hist_vol": metrics.get("hist_vol", np.nan),
                "zscore": metrics.get("zscore", np.nan),
                "percentile": metrics.get("percentile", np.nan),
                "current_value": stats.get("ref_val", np.nan) if stats.get("ok") else np.nan,
                "asof": pd.Timestamp(meta["asof"]),
                "window_end": pd.Timestamp(meta["window_end_date"]),
            }
            stop = row["mae_p75"]
            median_edge = abs(float(row["median_change"])) if np.isfinite(row["median_change"]) else np.nan
            if np.isfinite(median_edge) and np.isfinite(stop) and stop > 0:
                row["edge_score"] = float((row["success_pct"] / 100.0) * (median_edge / stop))
            else:
                row["edge_score"] = np.nan
            rows.append(row)
        except Exception as exc:
            rows.append(
                {
                    "strategy": spec.name,
                    "category": strategy_categories.get(spec.name, "Other"),
                    "value_source": VALUE_SOURCE_LABELS.get(spec.value_source, spec.value_source),
                    "status": "Error",
                    "status_detail": str(exc),
                    "anchor_years": 0,
                    "sample": 0,
                    "bias": "NA",
                    "success_pct": np.nan,
                    "up_pct": np.nan,
                    "avg_change": np.nan,
                    "median_change": np.nan,
                    "avg_abs_change": np.nan,
                    "mae_p75": np.nan,
                    "mae_p90": np.nan,
                    "reward_risk": np.nan,
                    "hist_vol": np.nan,
                    "zscore": np.nan,
                    "percentile": np.nan,
                    "current_value": np.nan,
                    "asof": pd.Timestamp(asof),
                    "window_end": pd.Timestamp(asof),
                    "edge_score": np.nan,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.sort_values(
        by=["edge_score", "success_pct", "sample", "strategy"],
        ascending=[False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    return out


# -------------------------
# UI
# -------------------------

def main() -> None:
    st.set_page_config(page_title=APP_NAME, layout="wide")

    # Session keys
    if "reload_data_token" not in st.session_state:
        st.session_state["reload_data_token"] = 0
    if "reload_strat_token" not in st.session_state:
        st.session_state["reload_strat_token"] = 0

    st.title(APP_NAME)
    st.caption(f"{APP_TAGLINE} | v{APP_VERSION}")
    st.markdown(STRATEGY_CARD_STYLE, unsafe_allow_html=True)

    # Sidebar controls
    with st.sidebar:
        st.markdown(
            """
            <div class="strategy-card">
              <div class="title">Active Strategy</div>
              <div class="sub">Pick strategy first, then tune data and window settings.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        strategies_path = st.text_input("Strategies YAML", value=str(DEFAULT_STRATEGIES_PATH))

        if st.button("Reload strategies", width="stretch"):
            st.session_state["reload_strat_token"] += 1
            clear_streamlit_caches()
            st.rerun()

        try:
            specs, strategy_categories = load_strategies_yaml(
                strategies_path,
                st.session_state["reload_strat_token"],
            )
        except Exception as e:
            st.error(f"Unable to load strategies: {e}")
            st.stop()

        unique_categories = sorted(
            set(strategy_categories.values()),
            key=lambda c: (
                CATEGORY_ORDER.index(c) if c in CATEGORY_ORDER else len(CATEGORY_ORDER),
                str(c).lower(),
            ),
        )
        selected_category = st.selectbox("Category", ["All"] + unique_categories)
        strategy_options = [
            name
            for name in specs.keys()
            if selected_category == "All" or strategy_categories.get(name) == selected_category
        ]
        if not strategy_options:
            st.error("No strategies found for the selected category.")
            st.stop()

        strategy_name = st.selectbox("Select strategy", strategy_options)
        spec = specs[strategy_name]
        value_source_label = VALUE_SOURCE_LABELS.get(spec.value_source, str(spec.value_source))
        st.caption(
            f"Category: {strategy_categories.get(strategy_name, 'Other')} | "
            f"Type: {value_source_label}"
        )
        st.divider()

        if st.button("Reload data", width="stretch"):
            st.session_state["reload_data_token"] += 1
            clear_streamlit_caches(reset_asof_input=True)
            st.rerun()
        st.divider()

        st.header("Data source")
        src_label = st.radio("Source", options=["Bloomberg", "Parquet", "CSV"], index=0)
        source: DataSource = {"Bloomberg": "bloomberg", "Parquet": "parquet", "CSV": "csv"}[src_label]
        include_today_live = False

        # Bloomberg settings
        bbg_cfg: Optional[BloombergPullConfig] = None
        cache_cfg: Optional[CacheConfig] = None

        csv_path = None
        parquet_path = None

        if source == "bloomberg":
            st.subheader("Bloomberg config")
            host = st.text_input("Host", value="localhost")
            port = st.number_input("Port", min_value=1, max_value=65535, value=8194, step=1)

            history_start_date = st.date_input(
                "History start",
                value=pd.Timestamp("2010-01-01").date(),
                help="Start date for historical price pulls.",
            )
            generic_history_start_date = history_start_date
            with st.expander("Advanced universe options", expanded=False):
                use_separate_generic_start = st.checkbox(
                    "Use separate generic ticker history start",
                    value=False,
                    help="Only needed if you want a different start date for FUT_CUR_GEN_TICKER discovery.",
                )
                if use_separate_generic_start:
                    generic_history_start_date = st.date_input(
                        "Generic ticker history start",
                        value=history_start_date,
                    )

            history_behavior = st.radio(
                "History and behavior",
                options=[
                    "Previous business day only",
                    "Previous business day + append latest snapshot for today",
                ],
                index=0,
                help="Use snapshot mode for the latest intraday PX_LAST. If market is not open, today's snapshot may still match prior close.",
            )
            include_today = "append latest snapshot" in history_behavior.lower()
            include_today_live = bool(include_today)
            end_mode = "yesterday"
            snapshot_mode = "overwrite"
            if include_today:
                st.caption("Snapshot mode uses overwrite, so pressing Reload data always refreshes today's row from Bloomberg.")

            st.subheader("Universe cache")
            force_refresh_universe = st.checkbox("Force refresh FUT_CHAIN / generics", value=False)
            ttl_days = st.number_input("Ticker cache TTL (days)", min_value=1, max_value=365, value=30, step=1)

            st.subheader("Local cache")
            cache_dir = st.text_input("Cache directory", value=str(default_cache_dir()))

            start_date = pd.Timestamp(history_start_date).strftime("%Y%m%d")
            generic_history_start = pd.Timestamp(generic_history_start_date).strftime("%Y%m%d")

            bbg_cfg = BloombergPullConfig(
                host=str(host),
                port=int(port),
                start_date_yyyymmdd=str(start_date),
                generic_history_start_yyyymmdd=(
                    str(generic_history_start).strip() or str(start_date).strip()
                ),
                end_mode=str(end_mode),
                include_today_snapshot=False,
                snapshot_mode=str(snapshot_mode),
                force_refresh_universe=bool(force_refresh_universe),
                ticker_cache_ttl_days=int(ttl_days),
            )
            cache_cfg = CacheConfig(cache_dir=str(cache_dir))

        else:
            st.subheader("File paths")
            if source == "parquet":
                parquet_path = st.text_input("Parquet path", value="")
            else:
                csv_path = st.text_input("CSV path", value="")

    # Load data once for the full strategy library.
    df_raw, data_meta = load_data_cached(
        st.session_state["reload_data_token"],
        source=source,
        specs=list(specs.values()),
        bbg_cfg=bbg_cfg,
        cache_cfg=cache_cfg,
        csv_path=csv_path or None,
        parquet_path=parquet_path or None,
    )

    # In live-snapshot mode, refresh today's snapshot on every rerun/reload
    # without forcing a full history pull.
    if source == "bloomberg" and include_today_live and bbg_cfg is not None:
        try:
            df_raw, snap_meta = append_bloomberg_snapshot_live(df_raw, bbg_cfg=bbg_cfg)
            data_meta = {**dict(data_meta or {}), **dict(snap_meta or {})}
        except Exception as e:
            data_meta = {**dict(data_meta or {}), "snapshot_live_error": str(e)}

    # Build engine
    engine = get_engine(df_raw)

    # Build options
    st.sidebar.header("Spread construction")
    fill_enabled = st.sidebar.checkbox(
        "Fill interior business-day gaps (ffill)",
        value=True,
        help="Only interior gaps are filled. Trailing stale tails are kept as NaN.",
    )
    fill_limit = st.sidebar.number_input(
        "Fill limit (business days)",
        min_value=0,
        max_value=5,
        value=1,
        step=1,
    )
    fill = FillPolicy(method="ffill", limit=int(fill_limit), apply_to_fx=True) if fill_enabled else None

    curves, report = engine.build_with_report(spec, fill=fill, multiindex_cols=False)
    curves = curves.dropna(how="all")
    asof_default = pd.Timestamp(curves.index.max()) if len(curves.index) else pd.Timestamp(engine.df.index.max())

    # As-of selection.
    with st.sidebar:
        st.header("As-of")
        min_d = pd.Timestamp(curves.index.min()).date() if len(curves.index) else asof_default.date()
        max_d = pd.Timestamp(curves.index.max()).date() if len(curves.index) else asof_default.date()
        asof_mode = st.radio(
            "ASOF mode",
            options=["Latest available (recommended)", "Manual date"],
            index=0,
        )
        if asof_mode.startswith("Latest"):
            asof_ts = pd.Timestamp(max_d)
            st.caption(f"ASOF locked to latest available date: {asof_ts.date()}")
        else:
            asof_date = st.date_input(
                "As-of date",
                value=asof_default.date(),
                min_value=min_d,
                max_value=max_d,
                key="asof_date_input",
            )
            asof_ts = pd.Timestamp(asof_date)

    # Seasonal window selection
    with st.sidebar:
        st.header("Seasonal window")
        start_m, end_m = st.slider(
            "Window in months relative to ASOF",
            min_value=-24,
            max_value=24,
            value=(-9, 6),
            step=1,
        )
        st.caption("Window is in months relative to ASOF, for example -9 to +6.")
        alignment_label = st.radio(
            "Lifecycle alignment",
            options=[
                "ASOF-aligned window (recommended)",
                "Absolute day-count since first valid",
            ],
            index=0,
            help=(
                "ASOF-aligned aligns each curve on its own ASOF point. "
                "Absolute mode compares the same raw lifecycle day-count since first valid."
            ),
        )
        alignment_mode = (
            "asof_aligned" if alignment_label.startswith("ASOF-aligned") else "first_valid"
        )
        if alignment_mode == "asof_aligned":
            st.caption("ASOF-aligned mode keeps month buckets comparable across anchor years.")
        else:
            st.caption("Absolute mode may omit years that do not reach the selected lifecycle window.")

    # Life-cycle overlay view
    matrix, calendar, meta = overlay_lifecycle_window_compat(
        engine,
        curves,
        window_months=(int(start_m), int(end_m)),
        asof=asof_ts,
        reference="front",
        axis="t",
        alignment=alignment_mode,
    )
    ref_ay = int(meta["reference_anchor_year"])

    # Select which anchor years to display
    with st.sidebar:
        st.header("Display")
        available_years = [int(c) for c in matrix.columns]
        available_years_sorted = sorted(available_years)
        default_years = available_years_sorted
        selected_years = st.multiselect("Anchor years", options=available_years_sorted, default=default_years)
        show_asof_line = st.checkbox("Show ASOF marker", value=True)
        display_years = resolve_display_years(
            matrix,
            selected_years=[int(y) for y in selected_years],
            reference_anchor_year=ref_ay,
        )

    insights_df = build_trading_insights_table(
        df_raw,
        tuple(specs.values()),
        strategy_categories,
        asof=asof_ts,
        window_months=(int(start_m), int(end_m)),
        alignment=alignment_mode,
        fill_enabled=fill_enabled,
        fill_limit=int(fill_limit),
    )

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Seasonal (life-cycle)",
        "Trading insights",
        "Full timeline",
        "Diagnostics",
    ])

    with tab1:
        st.subheader(strategy_name)
        st.caption(
            f"Value source: {value_source_label} | "
            f"Output currency: {spec.output_currency} | "
            f"ASOF (selected): {meta['asof'].date()} | "
            f"Reference anchor year: {ref_ay} | "
            f"Window: {start_m}m -> {end_m}m | "
            f"Alignment: {meta.get('alignment', 'first_valid')}"
        )

        stats = lifecycle_stats_at_asof(matrix, calendar, ref_ay, display_years)
        m1, m2, m3, m4 = st.columns(4)

        if stats.get("ok"):
            m1.metric("ASOF label", stats["asof_label"])
            m2.metric("Current (ref) value", f"{stats['ref_val']:.2f}" if np.isfinite(stats["ref_val"]) else "NA")
            if "hist_mean" in stats:
                m3.metric("Hist mean @ ASOF", f"{stats['hist_mean']:.2f}")
                m4.metric("Z-score / Percentile", f"{stats['zscore']:.2f} / {stats['percentile']:.0f}p")
            else:
                m3.metric("Historical samples @ ASOF", f"{stats.get('hist_n', 0)}")
                m4.metric(" ", " ")
        else:
            st.info(f"Stats not available: {stats.get('reason')}")

        fig = plot_lifecycle_overlay(
            matrix,
            calendar,
            reference_anchor_year=ref_ay,
            selected_years=display_years,
            title=f"{strategy_name} - Seasonal overlay (life-cycle)",
            show_asof_line=show_asof_line,
            axis_title=(
                "Seasonal axis (ASOF-aligned trading-day index)"
                if str(meta.get("alignment")) == "asof_aligned"
                else "Seasonal axis (trading-day count since each curve first valid)"
            ),
        )
        st.plotly_chart(fig, width="stretch")

        st.caption(
            f"Reference-year leg futures prices (raw contracts) | "
            f"Anchor year: {ref_ay} | Last 10 trading sessions up to ASOF {meta['asof'].date()}"
        )
        try:
            leg_contracts = engine.build_leg_contract_values(spec, anchor_year=ref_ay, fill=fill)
        except Exception as e:
            st.info(f"Leg contract table unavailable: {e}")
        else:
            leg_contracts = leg_contracts.loc[:pd.Timestamp(meta["asof"])].dropna(how="all")
            if leg_contracts.empty:
                st.info("No leg contract values available up to the selected ASOF.")
            else:
                last_10 = leg_contracts.tail(10)
                tbl = last_10.T
                tbl.columns = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in last_10.index]
                tbl.index.name = "Leg (contract [source])"
                st.dataframe(tbl.round(2))

        st.subheader("Seasonal heatmap")
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            heat_bucket_label = st.radio(
                "Bucketing",
                options=[
                    "Per month",
                    "Per month decade (1-9 / 10-19 / 20-EOM)",
                ],
                horizontal=True,
            )
        with c2:
            heat_agg_label = st.selectbox(
                "Aggregation",
                options=[
                    "Mean",
                    "Last available (EOM in month mode)",
                    "First available",
                    "Median",
                    "Min",
                    "Max",
                ],
                index=0,
            )
        with c3:
            heat_value_label = st.selectbox(
                "Values",
                options=["Price", "Log return"],
                index=0,
            )

        bucket_mode = (
            "month_decade" if heat_bucket_label.startswith("Per month decade") else "month"
        )
        agg_mode_map = {
            "Mean": "mean",
            "Last available (EOM in month mode)": "last",
            "First available": "first",
            "Median": "median",
            "Min": "min",
            "Max": "max",
        }
        agg_mode = agg_mode_map[heat_agg_label]
        value_mode = "log_return" if heat_value_label == "Log return" else "price"

        if bucket_mode == "month_decade":
            st.caption("Month-decade view is limited to M-3..M+3 around ASOF.")

        if value_mode == "log_return":
            st.caption("Log return is computed as current bucket vs previous bucket.")

        heatmap_years = [int(y) for y in display_years if int(y) <= int(ref_ay)]
        excluded_future_heatmap_years = sorted(
            {int(y) for y in display_years if int(y) > int(ref_ay)}
        )
        if excluded_future_heatmap_years:
            st.caption(
                "Heatmap excludes future anchor years: "
                + ", ".join(str(y) for y in excluded_future_heatmap_years)
            )

        heat_table_base = build_lifecycle_heatmap_table(
            matrix,
            calendar,
            years=heatmap_years,
            asof=pd.Timestamp(meta["asof"]),
            bucket_mode=bucket_mode,
            agg_mode=agg_mode,
        )
        heat_table_base, dropped_rows_base, dropped_cols_base = compact_heatmap_table(
            heat_table_base,
            drop_empty_rows=False,
            drop_empty_cols=True,
            reference_anchor_year=ref_ay,
        )
        if value_mode == "log_return":
            heat_table = to_period_log_return(heat_table_base)
            heat_table, dropped_rows_ret, dropped_cols_ret = compact_heatmap_table(
                heat_table,
                drop_empty_rows=False,
                drop_empty_cols=True,
                reference_anchor_year=ref_ay,
            )
            dropped_rows = sorted(set(dropped_rows_base + dropped_rows_ret), reverse=True)
            dropped_cols = sorted(set(dropped_cols_base + dropped_cols_ret), key=bucket_sort_key)
        else:
            heat_table = heat_table_base
            dropped_rows = dropped_rows_base
            dropped_cols = dropped_cols_base

        empty_years = [
            int(y)
            for y in heat_table.index
            if heat_table.loc[y].isna().all()
        ]

        if heat_table.empty or heat_table.shape[1] == 0:
            st.info("Heatmap not available for the current selection.")
        else:
            if empty_years or dropped_rows or dropped_cols:
                st.caption(
                    f"Years with no values in window: {', '.join(str(y) for y in empty_years) if empty_years else 'none'} | "
                    f"Omitted empty years: {', '.join(str(y) for y in dropped_rows) if dropped_rows else 'none'} | "
                    f"Omitted empty buckets: {len(dropped_cols)}"
                )
            if not np.isfinite(heat_table.to_numpy(dtype=float)).any():
                st.info("Heatmap values are unavailable for this selection.")
            else:
                fig_hm = plot_lifecycle_heatmap(
                    heat_table,
                    reference_anchor_year=ref_ay,
                    title=(
                        f"{strategy_name} - Heatmap (Period log return)"
                        if value_mode == "log_return"
                        else f"{strategy_name} - Heatmap ({heat_agg_label}, {heat_value_label})"
                    ),
                    value_mode=value_mode,
                )
                st.plotly_chart(fig_hm, width="stretch")

        with st.expander("Calendar mapping (t -> label)", expanded=False):
            st.dataframe(calendar)

    with tab2:
        st.subheader("Trading insights")
        st.caption("Cross-strategy seasonal trade metrics from the selected ASOF to the end of the current forward window.")

        insight_categories = ["All"] + [
            c for c in unique_categories if c in set(insights_df["category"].dropna().astype(str))
        ]
        selected_insight_category = st.selectbox("Insights category", insight_categories, index=0)
        only_actionable = st.checkbox(
            "Show only actionable rows",
            value=True,
            help="Hide strategies without enough forward-history sample or analytics.",
        )

        filtered_insights = insights_df.copy()
        if selected_insight_category != "All":
            filtered_insights = filtered_insights.loc[
                filtered_insights["category"] == selected_insight_category
            ].copy()
        if only_actionable:
            filtered_insights = filtered_insights.loc[filtered_insights["status"] == "OK"].copy()

        filtered_insights = filtered_insights.reset_index(drop=True)

        c1, c2, c3, c4 = st.columns(4)
        actionable_count = int((filtered_insights["status"] == "OK").sum()) if not filtered_insights.empty else 0
        bullish_count = int((filtered_insights["bias"] == "Long").sum()) if not filtered_insights.empty else 0
        bearish_count = int((filtered_insights["bias"] == "Short").sum()) if not filtered_insights.empty else 0
        best_row = filtered_insights.iloc[0] if not filtered_insights.empty else None
        c1.metric("Strategies", f"{len(filtered_insights)}")
        c2.metric("Actionable", f"{actionable_count}")
        c3.metric("Long / Short", f"{bullish_count} / {bearish_count}")
        c4.metric(
            "Top edge",
            best_row["strategy"] if best_row is not None else "NA",
            f"{best_row['edge_score']:.2f}"
            if best_row is not None and np.isfinite(best_row["edge_score"])
            else "NA",
        )

        if filtered_insights.empty:
            st.info("No strategy matches the current insights filters.")
        else:
            display_cols = [
                "strategy",
                "category",
                "bias",
                "success_pct",
                "up_pct",
                "median_change",
                "avg_change",
                "mae_p75",
                "mae_p90",
                "reward_risk",
                "hist_vol",
                "zscore",
                "percentile",
                "edge_score",
                "sample",
                "status",
            ]
            table = filtered_insights[display_cols].copy()
            styler = table.style.format(
                {
                    "success_pct": "{:.1f}%",
                    "up_pct": "{:.1f}%",
                    "median_change": "{:.2f}",
                    "avg_change": "{:.2f}",
                    "mae_p75": "{:.2f}",
                    "mae_p90": "{:.2f}",
                    "reward_risk": "{:.2f}",
                    "hist_vol": "{:.2f}",
                    "zscore": "{:.2f}",
                    "percentile": "{:.0f}p",
                    "edge_score": "{:.2f}",
                },
                na_rep="NA",
            )
            styler = styler.background_gradient(
                subset=["success_pct", "reward_risk", "edge_score"], cmap="Greens"
            )
            styler = styler.background_gradient(
                subset=["zscore"], cmap="RdYlGn_r", vmin=-2.5, vmax=2.5
            )
            styler = styler.background_gradient(
                subset=["mae_p75", "mae_p90", "hist_vol"], cmap="Reds"
            )
            st.dataframe(
                styler,
                width="stretch",
                column_config={
                    "strategy": st.column_config.TextColumn("Strategy", width="large"),
                    "category": st.column_config.TextColumn("Category", width="medium"),
                    "bias": st.column_config.TextColumn("Bias", width="small"),
                    "success_pct": st.column_config.NumberColumn(
                        "Success %",
                        help="Hit rate in the direction of the inferred seasonal bias.",
                    ),
                    "up_pct": st.column_config.NumberColumn(
                        "Up %",
                        help="Share of historical windows that finished higher than ASOF.",
                    ),
                    "median_change": st.column_config.NumberColumn(
                        "Median dP",
                        help="Median point move from ASOF to the end of the forward window.",
                    ),
                    "avg_change": st.column_config.NumberColumn("Average dP"),
                    "mae_p75": st.column_config.NumberColumn(
                        "Stop P75",
                        help="75th percentile adverse excursion in the bias direction.",
                    ),
                    "mae_p90": st.column_config.NumberColumn(
                        "Stop P90",
                        help="90th percentile adverse excursion in the bias direction.",
                    ),
                    "reward_risk": st.column_config.NumberColumn("Reward/Risk"),
                    "hist_vol": st.column_config.NumberColumn(
                        "Hist vol",
                        help="Median annualized realized volatility over the forward path.",
                    ),
                    "zscore": st.column_config.NumberColumn(
                        "Z-score",
                        help="Current ASOF value versus historical ASOF distribution.",
                    ),
                    "percentile": st.column_config.NumberColumn("Percentile"),
                    "edge_score": st.column_config.NumberColumn("Edge"),
                    "sample": st.column_config.NumberColumn(
                        "N",
                        help="Historical anchor years used in the forward metric sample.",
                    ),
                    "status": st.column_config.TextColumn("Status", width="small"),
                },
                hide_index=True,
            )

            with st.expander("Metric definitions", expanded=False):
                st.markdown(
                    """
                    - `Bias`: inferred trade direction from the median historical forward move.
                    - `Success %`: historical hit rate in that bias direction.
                    - `Stop P75 / P90`: adverse excursion percentile to frame statistical stop placement.
                    - `Z-score / Percentile`: how stretched the current spread is versus its historical ASOF distribution.
                    - `Edge`: simple ranking score using hit rate and median move versus stop distance.
                    """
                )

            with st.expander("Full insights dataset", expanded=False):
                st.dataframe(filtered_insights, width="stretch", hide_index=True)

    with tab3:
        st.subheader("Full timeline (real dates)")
        st.caption("Full series on calendar dates.")

        fig2 = go.Figure()
        for y in sorted(set(int(y) for y in selected_years)):
            if y not in curves.columns:
                continue
            fig2.add_trace(go.Scatter(x=curves.index, y=curves[y], mode="lines", name=str(y)))

        fig2.update_layout(height=500, hovermode="x unified", title=strategy_name)
        fig2.update_xaxes(title_text="Date")
        fig2.update_yaxes(title_text="Value")
        st.plotly_chart(fig2, width="stretch")

        with st.expander("Availability table", expanded=False):
            st.dataframe(engine.availability(curves))

    with tab4:
        st.subheader("Diagnostics")
        st.caption("Use this tab to inspect missing data, FX, and contract mapping.")

        with st.expander("Build report (skipped anchor years)", expanded=True):
            st.dataframe(report)

        with st.expander("Data meta", expanded=False):
            st.json(data_meta)

        with st.expander("Contract resolution details", expanded=False):
            st.write("Shows how a 1-digit year ticker is mapped to a full contract year.")
            if spec.legs:
                leg0 = spec.legs[0]
                years = engine.available_contract_years(leg0.ticker_root, leg0.month_code)
                if years:
                    y = years[-1]
                    st.json(engine.explain_contract(leg0.ticker_root, leg0.month_code, y))
                else:
                    st.info("No contracts found for the first leg.")

    st.caption("Bloomberg mode uses local Parquet caching with incremental updates.")


if __name__ == "__main__":
    main()










