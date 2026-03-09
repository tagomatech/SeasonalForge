from __future__ import annotations

from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def plot_lifecycle_overlay(
    matrix: pd.DataFrame,
    calendar: pd.DataFrame,
    *,
    reference_anchor_year: int,
    selected_years: List[int],
    title: str,
    show_asof_line: bool = True,
    axis_title: str = "Seasonal axis (trading-day count since each curve first valid)",
) -> go.Figure:
    """
    Plot lifecycle overlay matrix and highlight the reference anchor-year curve.
    """
    years = [int(y) for y in selected_years if int(y) in matrix.columns]
    if int(reference_anchor_year) in matrix.columns and int(reference_anchor_year) not in years:
        years.append(int(reference_anchor_year))
    years = sorted(set(years))

    # Keep non-reference curves readable without overpowering the reference curve.
    other_colors = [
        "#1f77b4",  # blue
        "#2ca02c",  # green
        "#9467bd",  # purple
        "#17becf",  # cyan
        "#bcbd22",  # olive
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#7f7f7f",  # gray
    ]

    fig = go.Figure()
    other_idx = 0
    for y in years:
        s = matrix[y]
        is_current = int(y) == int(reference_anchor_year)
        if is_current:
            line_width = 4
            line_color = "rgb(255,0,0)"
            opacity = 1.0
        else:
            line_width = 2.2
            line_color = other_colors[other_idx % len(other_colors)]
            opacity = 0.9
            other_idx += 1
        fig.add_trace(
            go.Scatter(
                x=matrix.index,
                y=s,
                mode="lines",
                name=str(y),
                line={"width": line_width, "color": line_color},
                opacity=opacity,
                hovertemplate="t=%{x}<br>value=%{y:.4f}<extra>" + str(y) + "</extra>",
            )
        )

    if show_asof_line and "is_asof" in calendar.columns:
        asof_rows = calendar.loc[calendar["is_asof"]]
        if not asof_rows.empty:
            t_asof = int(asof_rows["t"].iloc[0])
            if t_asof >= int(matrix.index.min()) and t_asof <= int(matrix.index.max()):
                fig.add_vline(x=t_asof, line_width=2, line_dash="dot")
                fig.add_annotation(x=t_asof, y=1.02, xref="x", yref="paper", text="ASOF", showarrow=False)

    # Tick labels
    cal = calendar.set_index("t")
    tmin, tmax = int(matrix.index.min()), int(matrix.index.max())
    candidate = list(range(tmin, tmax + 1))
    step = max(int(len(candidate) / 10), 1)  # ~10 ticks
    tickvals = candidate[::step]
    tickvals = sorted(set(tickvals + [tmin, tmax] + cal.index[cal.get("is_asof", False)].tolist()))
    ticktext = [cal.loc[t, "label"] if t in cal.index else str(t) for t in tickvals]

    fig.update_layout(
        title=title,
        height=600,
        legend_title_text="Anchor year",
        margin=dict(l=10, r=10, t=50, b=10),
        hovermode="x unified",
    )
    fig.update_xaxes(
        title_text=str(axis_title),
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext,
        showgrid=True,
    )
    fig.update_yaxes(title_text="Value", showgrid=True)
    return fig


def resolve_display_years(
    matrix: pd.DataFrame,
    *,
    selected_years: List[int],
    reference_anchor_year: int,
) -> List[int]:
    """Return plotted years, forcing inclusion of the reference anchor year."""
    years = [int(y) for y in selected_years if int(y) in matrix.columns]
    if int(reference_anchor_year) in matrix.columns and int(reference_anchor_year) not in years:
        years.append(int(reference_anchor_year))
    return sorted(set(years))


def build_lifecycle_heatmap_table(
    matrix: pd.DataFrame,
    calendar: pd.DataFrame,
    *,
    years: List[int],
    asof: pd.Timestamp,
    bucket_mode: str,   # "month" | "month_decade"
    agg_mode: str,      # "mean" | "last" | "first" | "median" | "min" | "max"
) -> pd.DataFrame:
    """Aggregate lifecycle values into a year x bucket table for heatmap rendering."""
    if matrix.empty or calendar.empty:
        return pd.DataFrame()

    years_in_matrix = [int(y) for y in years if int(y) in matrix.columns]
    if not years_in_matrix:
        return pd.DataFrame()

    cal = calendar.set_index("t")
    base = pd.DataFrame(index=matrix.index)
    base["date"] = pd.to_datetime(cal.reindex(matrix.index)["date"], errors="coerce")
    base = base.dropna(subset=["date"])
    if base.empty:
        return pd.DataFrame()

    asof_ts = pd.Timestamp(asof)
    base["month_offset"] = (
        (base["date"].dt.year - int(asof_ts.year)) * 12
        + (base["date"].dt.month - int(asof_ts.month))
    ).astype(int)

    if bucket_mode == "month_decade":
        # Keep decade view compact regardless of the wider lifecycle window.
        base = base[(base["month_offset"] >= -3) & (base["month_offset"] <= 3)]
        if base.empty:
            return pd.DataFrame()
        day = base["date"].dt.day
        base["bucket"] = np.select(
            [day <= 9, day <= 19],
            ["01-09", "10-19"],
            default="20-EOM",
        )
        base["column"] = base["month_offset"].map(lambda m: f"M{int(m):+d}") + " " + base["bucket"]

        month_vals = sorted(set(int(m) for m in base["month_offset"].tolist()))
        decade_labels = ["01-09", "10-19", "20-EOM"]
        ordered_cols = [f"M{m:+d} {d}" for m in month_vals for d in decade_labels]
    else:
        base["column"] = base["month_offset"].map(lambda m: f"M{int(m):+d}")
        month_vals = sorted(set(int(m) for m in base["month_offset"].tolist()))
        ordered_cols = [f"M{m:+d}" for m in month_vals]

    out = pd.DataFrame(index=years_in_matrix, columns=ordered_cols, dtype=float)

    for y in years_in_matrix:
        vals = pd.to_numeric(matrix.loc[base.index, int(y)], errors="coerce")
        tmp = pd.DataFrame(
            {
                "date": base["date"].values,
                "column": base["column"].values,
                "value": vals.values,
            },
            index=base.index,
        )
        tmp = tmp.dropna(subset=["column"])
        if tmp.empty:
            continue

        tmp_sorted = tmp.sort_values("date")
        nonnull = tmp_sorted.dropna(subset=["value"])

        if agg_mode == "last":
            agg = tmp_sorted.groupby("column")["value"].last()
        elif agg_mode == "first":
            agg = tmp_sorted.groupby("column")["value"].first()
        elif agg_mode == "median":
            agg = tmp.groupby("column")["value"].median()
        elif agg_mode == "min":
            agg = tmp.groupby("column")["value"].min()
        elif agg_mode == "max":
            agg = tmp.groupby("column")["value"].max()
        else:
            agg = tmp.groupby("column")["value"].mean()

        # Historical rows can still have mapped post-ASOF observations even when the
        # underlying contract expires before the next calendar month begins. In that
        # case, surface the last mapped post-ASOF value in M+1 instead of leaving the
        # first forward month blank.
        if bucket_mode == "month" and not nonnull.empty and "M+1" in ordered_cols and pd.isna(agg.get("M+1")):
            last_valid = nonnull.iloc[-1]
            last_date = pd.Timestamp(last_valid["date"])
            if last_date > asof_ts:
                last_month_offset = int(
                    (int(last_date.year) - int(asof_ts.year)) * 12
                    + (int(last_date.month) - int(asof_ts.month))
                )
                if last_month_offset == 0:
                    agg.loc["M+1"] = float(last_valid["value"])

        out.loc[int(y), agg.index] = agg.values

    return out


def bucket_sort_key(label: str) -> Tuple[int, int]:
    """Sort key for lifecycle heatmap bucket labels (month then in-month decade)."""
    s = str(label).strip()
    parts = s.split()
    m = int(parts[0].replace("M", ""))
    if len(parts) == 1:
        return (m, 0)
    decade_rank = {"01-09": 0, "10-19": 1, "20-EOM": 2}.get(parts[1], 9)
    return (m, decade_rank)


def _bucket_sort_key(label: str) -> Tuple[int, int]:
    """Backward-compatible alias for older imports."""
    return bucket_sort_key(label)

def _ordered_anchor_years(years: List[int], reference_anchor_year: Optional[int]) -> List[int]:
    """Order years so the reference year appears first, then older years, then future years."""
    uniq = sorted(set(int(y) for y in years))
    if reference_anchor_year is None:
        return sorted(uniq, reverse=True)

    ref = int(reference_anchor_year)
    present_and_past = sorted((y for y in uniq if y <= ref), reverse=True)
    future = sorted((y for y in uniq if y > ref))
    return present_and_past + future


def compact_heatmap_table(
    heatmap_table: pd.DataFrame,
    *,
    drop_empty_rows: bool = True,
    drop_empty_cols: bool = True,
    reference_anchor_year: Optional[int] = None,
) -> Tuple[pd.DataFrame, List[int], List[str]]:
    """Optionally drop empty rows/columns and enforce stable row/column ordering."""
    if heatmap_table.empty:
        return heatmap_table.copy(), [], []

    row_before = [int(i) for i in heatmap_table.index]
    col_before = [str(c) for c in heatmap_table.columns]

    out = heatmap_table.copy()
    if drop_empty_rows:
        out = out.loc[~out.isna().all(axis=1), :]
    if drop_empty_cols:
        out = out.loc[:, ~out.isna().all(axis=0)]

    if out.empty:
        return out, row_before, col_before

    ordered_rows = _ordered_anchor_years([int(i) for i in out.index], reference_anchor_year)
    out = out.reindex(index=ordered_rows)
    out = out.reindex(columns=sorted(list(out.columns), key=bucket_sort_key))

    row_after = [int(i) for i in out.index]
    col_after = [str(c) for c in out.columns]
    dropped_rows = [r for r in row_before if r not in set(row_after)]
    dropped_cols = [c for c in col_before if c not in set(col_after)]
    return out, dropped_rows, dropped_cols


def to_period_log_return(heatmap_table: pd.DataFrame) -> pd.DataFrame:
    """
    Compute period-over-period signed log return by row:
    current bucket versus previous bucket.
    """
    cur = pd.DataFrame(heatmap_table, copy=True)
    prev = cur.shift(axis=1)
    eps = 1e-12

    mag = np.log((cur.abs() + eps) / (prev.abs() + eps))
    direction = np.sign(cur - prev)
    out = np.sign(direction) * mag.abs()
    valid = np.isfinite(cur) & np.isfinite(prev)
    out = out.where(valid)
    out = out.where(direction != 0.0, 0.0)
    out = out.where(~(out.abs() < 1e-12), 0.0)
    return out


def plot_lifecycle_heatmap(
    heatmap_table: pd.DataFrame,
    *,
    reference_anchor_year: int,
    title: str,
    value_mode: str,  # "price" | "log_return"
) -> go.Figure:
    """Plot lifecycle heatmap from a year x bucket table."""
    row_ids = [int(y) for y in heatmap_table.index]
    y_labels = [f"{y} (ref)" if y == int(reference_anchor_year) else str(y) for y in row_ids]
    x_labels = [str(c) for c in heatmap_table.columns]
    x_vals = list(range(len(x_labels)))
    y_vals = list(range(len(y_labels)))

    zvals_raw = heatmap_table.to_numpy(dtype=float)
    zvals = (zvals_raw * 100.0) if value_mode == "log_return" else zvals_raw
    if value_mode == "log_return":
        zvals = np.where(np.abs(zvals) < 0.5, 0.0, zvals)

    colorscale = [
        [0.0, "#67000d"],   # dark red (lowest)
        [0.25, "#cb181d"],
        [0.50, "#f7f7f7"],
        [0.75, "#31a354"],
        [1.0, "#00441b"],   # dark green (highest)
    ]

    hover_label = "Log return (%)" if value_mode == "log_return" else "Price"

    zmax_abs = float(np.nanmax(np.abs(zvals))) if np.isfinite(zvals).any() else 0.0
    customdata = np.empty((len(y_labels), len(x_labels), 2), dtype=object)
    for i, ylab in enumerate(y_labels):
        for j, xlab in enumerate(x_labels):
            customdata[i, j, 0] = ylab
            customdata[i, j, 1] = xlab

    zmin_val = None
    zmax_val = None
    if value_mode == "log_return" and zmax_abs > 0.0:
        zmin_val = -zmax_abs
        zmax_val = zmax_abs

    fig = go.Figure(
        data=go.Heatmap(
            z=zvals,
            x=x_vals,
            y=y_vals,
            customdata=customdata,
            colorscale=colorscale,
            colorbar={
                "title": hover_label,
                "tickformat": ".1f",
                "ticksuffix": "%" if value_mode == "log_return" else "",
            },
            hovertemplate=f"Year=%{{customdata[0]}}<br>Bucket=%{{customdata[1]}}<br>{hover_label}=%{{z:.1f}}<extra></extra>",
            zmid=0 if value_mode == "log_return" else None,
            zmin=zmin_val,
            zmax=zmax_val,
            xgap=1,
            ygap=1,
        )
    )
    fig.update_layout(
        title=title,
        height=max(360, 220 + 34 * len(y_labels)),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    fig.update_xaxes(
        title_text="Bucket",
        tickmode="array",
        tickvals=x_vals,
        ticktext=x_labels,
        tickangle=-35,
    )
    fig.update_yaxes(
        title_text="Anchor year",
        tickmode="array",
        tickvals=y_vals,
        ticktext=y_labels,
        autorange="reversed",
    )

    finite = np.isfinite(zvals)
    if finite.any():
        zmin = float(np.nanmin(zvals))
        zmax = float(np.nanmax(zvals))
        zspan = max(zmax - zmin, 1e-12)
        font_size = int(max(9, min(13, 360 / max(len(x_labels), 1))))
        fmt = "{:.1f}%" if value_mode == "log_return" else "{:.1f}"

        for i, y in enumerate(y_labels):
            for j, x in enumerate(x_labels):
                v = zvals[i, j]
                if not np.isfinite(v):
                    continue
                norm = (float(v) - zmin) / zspan
                txt_color = "white" if (norm <= 0.22 or norm >= 0.78) else "#1f1f1f"
                fig.add_annotation(
                    x=j,
                    y=i,
                    xref="x",
                    yref="y",
                    text=f"<b>{fmt.format(float(v))}</b>",
                    showarrow=False,
                    font=dict(size=font_size, color=txt_color),
                )

    return fig


def lifecycle_stats_at_asof(
    matrix: pd.DataFrame,
    calendar: pd.DataFrame,
    reference_anchor_year: int,
    years: List[int],
) -> Dict[str, Any]:
    """
    Compute summary stats at the ASOF lifecycle point.
    """
    cal = calendar.set_index("t")
    if "is_asof" not in cal.columns:
        return {"ok": False, "reason": "ASOF not in window"}
    asof_rows = cal.loc[cal["is_asof"]]
    if asof_rows.empty:
        return {"ok": False, "reason": "ASOF not in window"}
    t_asof = int(asof_rows.index[0])
    if t_asof not in matrix.index:
        return {"ok": False, "reason": "ASOF not in window"}

    cols = [y for y in years if y in matrix.columns]
    row = matrix.loc[t_asof, cols].dropna()
    if row.empty:
        return {"ok": False, "reason": "No values at ASOF"}

    ref_val = matrix.loc[t_asof, reference_anchor_year] if reference_anchor_year in matrix.columns else np.nan
    hist = row.drop(labels=[reference_anchor_year], errors="ignore")

    out: Dict[str, Any] = {
        "ok": True,
        "t_asof": t_asof,
        "asof_label": cal.loc[t_asof, "label"],
        "ref_val": ref_val,
        "hist_n": int(len(hist)),
    }

    if len(hist) >= 2 and np.isfinite(ref_val):
        mean = float(hist.mean())
        std = float(hist.std(ddof=1))
        z = (float(ref_val) - mean) / std if std > 0 else np.nan
        pct = float((hist < ref_val).mean() * 100.0)
        out.update({"hist_mean": mean, "hist_std": std, "zscore": z, "percentile": pct})

    return out





