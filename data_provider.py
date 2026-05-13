
"""Data loading and caching for Bloomberg, CSV, and Parquet sources."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import re
from typing import Optional, Dict, Any, Iterable, List, Set, Tuple, Literal
import json
import hashlib

import pandas as pd
import numpy as np

from engine import StrategySpec

# Bloomberg access is required only in Bloomberg mode.
from bbg import BloombergClient, BloombergConnection


DataSource = Literal["bloomberg", "csv", "parquet"]


MONTH_CODE_TO_NUM: Dict[str, int] = {
    "F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
    "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12,
}

_FUTURES_TICKER_RE = re.compile(
    r"^\s*(?P<root>[A-Z0-9]{1,15})\s*(?P<month>[FGHJKMNQUVXZ])(?P<year>\d{1,2})\s+Comdty\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class DataLoadResult:
    df: pd.DataFrame
    meta: Dict[str, Any]


def default_cache_dir() -> Path:
    """Return a portable default cache directory outside the repo."""
    override = str(os.getenv("FUTURES_SEASONALS_CACHE_DIR", "")).strip()
    if override:
        return Path(override).expanduser()

    local_appdata = str(os.getenv("LOCALAPPDATA", "")).strip()
    if local_appdata:
        return Path(local_appdata) / "FuturesSeasonals"

    xdg_cache_home = str(os.getenv("XDG_CACHE_HOME", "")).strip()
    if xdg_cache_home:
        return Path(xdg_cache_home) / "futures_seasonals"

    return Path.home() / ".futures_seasonals"


@dataclass(frozen=True, slots=True)
class BloombergPullConfig:
    host: str = "localhost"
    port: int = 8194

    field: str = "PX_LAST"
    start_date_yyyymmdd: str = "20100101"          # Earliest date for price history.
    generic_history_start_yyyymmdd: Optional[str] = None  # Earliest date for generic ticker history.

    # Historical end date mode.
    end_mode: Literal["yesterday", "today"] = "yesterday"

    # Optionally append today's PX_LAST snapshot row.
    include_today_snapshot: bool = False
    snapshot_mode: Literal["append_if_missing", "overwrite"] = "append_if_missing"

    # Request chunk size.
    chunk_size: int = 200  # tickers per request
    # Re-fetch trailing business days on every incremental pull to capture
    # late settlements and patchy column-level updates.
    incremental_backfill_bdays: int = 3
    # Once a contract month is at least this many months behind the current
    # historical window and has been quiet for `expired_contract_quiet_bdays`,
    # keep serving it from local cache instead of re-pulling from Bloomberg.
    expired_contract_month_lag: int = 1
    expired_contract_quiet_bdays: int = 10

    # Futures universe cache controls.
    ticker_cache_ttl_days: int = 30
    force_refresh_universe: bool = False


@dataclass(frozen=True, slots=True)
class CacheConfig:
    """Local cache paths for dataset and metadata."""
    cache_dir: str = str(default_cache_dir())
    dataset_parquet: str = "bbg_dataset.parquet"
    dataset_meta_json: str = "bbg_dataset_meta.json"

    # Per-(root, month_code) ticker-list cache.
    tickers_dir: str = "tickers"


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _today_yyyymmdd(*, now: Optional[pd.Timestamp] = None) -> str:
    base = pd.Timestamp.today().normalize() if now is None else pd.Timestamp(now).normalize()
    return base.strftime("%Y%m%d")


def _previous_business_day_yyyymmdd(*, now: Optional[pd.Timestamp] = None) -> str:
    base = pd.Timestamp.today().normalize() if now is None else pd.Timestamp(now).normalize()
    return (base - pd.offsets.BDay(1)).strftime("%Y%m%d")


def _next_business_day_yyyymmdd(ts: pd.Timestamp) -> str:
    return (pd.Timestamp(ts).normalize() + pd.offsets.BDay(1)).strftime("%Y%m%d")


def _resolve_hist_end_yyyymmdd(
    bbg_cfg: BloombergPullConfig,
    *,
    now: Optional[pd.Timestamp] = None,
) -> str:
    """
    Resolve historical end date.
    - `end_mode=today` without snapshot: use today.
    - Otherwise: cap history at previous business day to avoid stale/partial same-day rows.
    """
    if bbg_cfg.end_mode == "today" and not bbg_cfg.include_today_snapshot:
        return _today_yyyymmdd(now=now)
    return _previous_business_day_yyyymmdd(now=now)


def _resolve_incremental_start_yyyymmdd(
    *,
    last_dt: Optional[pd.Timestamp],
    hist_end_yyyymmdd: str,
    start_date_yyyymmdd: str,
    backfill_business_days: int,
) -> Optional[str]:
    """
    Resolve incremental start date for cached datasets.
    Always re-fetch a trailing business-day window to backfill late rows and
    per-column gaps, while still catching up if cache is behind.
    """
    hist_end_ts = pd.Timestamp(hist_end_yyyymmdd).normalize()
    hist_start_ts = pd.Timestamp(start_date_yyyymmdd).normalize()
    if hist_start_ts > hist_end_ts:
        return None

    rewind = max(int(backfill_business_days), 0)
    backfill_start_ts = (hist_end_ts - pd.offsets.BDay(rewind)).normalize()
    candidate = backfill_start_ts

    if last_dt is not None and pd.notna(last_dt):
        next_after_last = (pd.Timestamp(last_dt).normalize() + pd.offsets.BDay(1)).normalize()
        candidate = min(candidate, next_after_last)

    candidate = max(candidate, hist_start_ts)
    if candidate > hist_end_ts:
        return None
    return candidate.strftime("%Y%m%d")


def _cap_history_to_end(df: Optional[pd.DataFrame], hist_end_yyyymmdd: str) -> Tuple[Optional[pd.DataFrame], bool]:
    """
    Cap frame rows at historical end date.
    Returns (capped_frame, was_trimmed).
    """
    if df is None:
        return None, False
    if df.empty:
        return df, False
    end_ts = pd.Timestamp(hist_end_yyyymmdd).normalize()
    capped = df.loc[df.index <= end_ts]
    return capped, len(capped.index) != len(df.index)


def _to_yyyymmdd(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y%m%d")


def _nearest_year_with_last_digit(digit: int, ref_year: int) -> int:
    """Nearest year to ref_year whose last digit equals digit."""
    decade = (ref_year // 10) * 10
    candidates = [decade - 10 + digit, decade + digit, decade + 10 + digit]
    return int(min(candidates, key=lambda y: abs(y - ref_year)))


def _expand_contract_year(token: str, ref_year: int) -> int:
    """Expand Bloomberg 1- or 2-digit contract year tokens into full years."""
    token = str(token).strip()
    if len(token) == 1:
        return _nearest_year_with_last_digit(int(token), ref_year)
    yy = int(token)
    return (2000 + yy) if yy < 80 else (1900 + yy)


def _parse_futures_contract_ticker(ticker: str, *, ref_year: int) -> Optional[Tuple[str, int, int]]:
    """Parse a Bloomberg futures contract ticker into (root, month_num, full_year)."""
    match = _FUTURES_TICKER_RE.match(str(ticker).strip())
    if not match:
        return None
    root = str(match.group("root")).upper().strip()
    month_code = str(match.group("month")).upper().strip()
    year = _expand_contract_year(str(match.group("year")), ref_year)
    month_num = MONTH_CODE_TO_NUM[month_code]
    return root, month_num, year


def _last_non_null_timestamp(series: pd.Series) -> Optional[pd.Timestamp]:
    """Return the most recent non-null timestamp in a column, if any."""
    valid = series.dropna()
    if valid.empty:
        return None
    return pd.Timestamp(valid.index.max()).normalize()


def _select_refresh_tickers(
    tickers: List[str],
    df_cached: Optional[pd.DataFrame],
    *,
    hist_end_yyyymmdd: str,
    backfill_business_days: int,
    expired_contract_month_lag: int,
    expired_contract_quiet_bdays: int,
) -> Tuple[List[str], List[str]]:
    """
    Split tickers into:
    - refresh_tickers: still worth asking Bloomberg for during incremental/snapshot pulls
    - expired_cached_futures: already-cached futures contracts that are old/quiet enough
      to serve entirely from local storage
    """
    ordered = list(dict.fromkeys(str(t) for t in tickers))
    if df_cached is None or df_cached.empty:
        return ordered, []

    hist_end_ts = pd.Timestamp(hist_end_yyyymmdd).normalize()
    hist_month_start = hist_end_ts.replace(day=1)
    quiet_cutoff = (
        hist_end_ts
        - pd.offsets.BDay(max(int(backfill_business_days), 0) + max(int(expired_contract_quiet_bdays), 0))
    ).normalize()

    refresh_tickers: List[str] = []
    expired_cached_futures: List[str] = []

    for ticker in ordered:
        if ticker not in df_cached.columns:
            refresh_tickers.append(ticker)
            continue

        parsed = _parse_futures_contract_ticker(ticker, ref_year=hist_end_ts.year)
        if parsed is None:
            refresh_tickers.append(ticker)
            continue

        _, contract_month, contract_year = parsed
        contract_month_start = pd.Timestamp(year=contract_year, month=contract_month, day=1)
        months_behind = (
            (hist_month_start.year - contract_month_start.year) * 12
            + (hist_month_start.month - contract_month_start.month)
        )
        if months_behind <= int(expired_contract_month_lag):
            refresh_tickers.append(ticker)
            continue

        last_valid = _last_non_null_timestamp(df_cached[ticker])
        if last_valid is None or last_valid >= quiet_cutoff:
            refresh_tickers.append(ticker)
            continue

        expired_cached_futures.append(ticker)

    return refresh_tickers, expired_cached_futures


def _hash_universe(tickers: List[str]) -> str:
    h = hashlib.sha1()
    for t in sorted(set(tickers)):
        h.update(t.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:12]


def required_pairs_and_fx(specs: Iterable[StrategySpec]) -> Tuple[Set[Tuple[str, str]], Set[str]]:
    """
    Extract all required (ticker_root, month_code) pairs + FX tickers from the strategy library.
    """
    pairs: Set[Tuple[str, str]] = set()
    fx: Set[str] = set()

    for spec in specs:
        for leg in spec.legs:
            pairs.add((leg.ticker_root, leg.month_code))
            if leg.fx_ticker:
                fx.add(leg.fx_ticker)

    return pairs, fx


def _tickers_cache_path(cache_cfg: CacheConfig, root: str, month_code: str) -> Path:
    cache_dir = Path(cache_cfg.cache_dir).expanduser()
    tick_dir = cache_dir / cache_cfg.tickers_dir
    _ensure_dir(tick_dir)
    fname = f"tickers_{root.upper().strip()}_{month_code.upper().strip()}.json"
    return tick_dir / fname


def _load_cached_ticker_list(
    path: Path,
    ttl_days: int,
    *,
    generic_history_start_yyyymmdd: Optional[str] = None,
) -> Optional[List[str]]:
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        created_at = pd.Timestamp(obj.get("created_at"))
        if (pd.Timestamp.now() - created_at).days > int(ttl_days):
            return None
        if generic_history_start_yyyymmdd is not None:
            cached_meta = obj.get("meta", {}) or {}
            cached_start = cached_meta.get("generic_history_start")
            if str(cached_start) != str(generic_history_start_yyyymmdd):
                return None
        tickers = obj.get("tickers", [])
        if not isinstance(tickers, list) or not tickers:
            return None
        return [str(t) for t in tickers]
    except Exception:
        return None


def _save_cached_ticker_list(path: Path, tickers: List[str], meta: Dict[str, Any]) -> None:
    obj = {
        "created_at": pd.Timestamp.now().isoformat(),
        "tickers": list(sorted(set(tickers))),
        "meta": meta,
    }
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def resolve_futures_universe(
    pairs: Set[Tuple[str, str]],
    *,
    bbg_cfg: BloombergPullConfig,
    cache_cfg: CacheConfig,
) -> List[str]:
    """Resolve futures tickers for each (root, month_code), using a local JSON cache."""
    futures: List[str] = []
    generic_start = (
        bbg_cfg.generic_history_start_yyyymmdd
        if bbg_cfg.generic_history_start_yyyymmdd
        else bbg_cfg.start_date_yyyymmdd
    )

    conn = BloombergConnection(host=bbg_cfg.host, port=bbg_cfg.port)
    with BloombergClient(conn) as bbg:
        for root, month_code in sorted(pairs):
            cache_path = _tickers_cache_path(cache_cfg, root, month_code)

            if not bbg_cfg.force_refresh_universe:
                cached = _load_cached_ticker_list(
                    cache_path,
                    ttl_days=bbg_cfg.ticker_cache_ttl_days,
                    generic_history_start_yyyymmdd=generic_start,
                )
            else:
                cached = None

            if cached is not None:
                futures.extend(cached)
                continue

            tickers = bbg.get_combined_futures(
                root,
                month_code=month_code,
                generic_history_start_yyyymmdd=generic_start,
            )
            futures.extend(tickers)
            _save_cached_ticker_list(
                cache_path,
                tickers,
                meta={"root": root, "month_code": month_code, "generic_history_start": generic_start},
            )

    return list(sorted(set(futures)))


def _cache_paths(cache_cfg: CacheConfig) -> Tuple[Path, Path]:
    base = Path(cache_cfg.cache_dir).expanduser()
    _ensure_dir(base)
    return base / cache_cfg.dataset_parquet, base / cache_cfg.dataset_meta_json


def _load_cache(cache_cfg: CacheConfig) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    parquet_path, meta_path = _cache_paths(cache_cfg)

    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        # Ensure a DatetimeIndex.
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated(keep="last")]
        df = df.sort_index()
    else:
        df = None

    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    else:
        meta = {}

    return df, meta


def _save_cache(cache_cfg: CacheConfig, df: pd.DataFrame, meta: Dict[str, Any]) -> None:
    parquet_path, meta_path = _cache_paths(cache_cfg)
    df.to_parquet(parquet_path)
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")


def _save_cache_meta(cache_cfg: CacheConfig, meta: Dict[str, Any]) -> None:
    """Persist cache metadata without rewriting the cached dataset parquet."""
    _, meta_path = _cache_paths(cache_cfg)
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")


def _merge_wide_frames(base: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    """
    Outer-merge two wide time-series frames on index and columns.
    For overlapping cells, incoming non-null values overwrite base values.
    """
    if incoming is None or incoming.empty:
        return base.sort_index()
    if base is None or base.empty:
        return incoming.sort_index()

    idx = base.index.union(incoming.index)
    cols = base.columns.union(incoming.columns)

    left = base.reindex(index=idx, columns=cols)
    right = incoming.reindex(index=idx, columns=cols)
    merged = left.where(right.isna(), right)
    return merged.sort_index()


def _append_today_snapshot(
    df: pd.DataFrame,
    snapshot: pd.Series,
    *,
    mode: Literal["append_if_missing", "overwrite"] = "append_if_missing",
) -> pd.DataFrame:
    """Append or overwrite today's row using snapshot values."""
    if snapshot.empty:
        return df

    today = pd.Timestamp.today().normalize()
    # Align snapshot to existing columns (adding missing ones).
    new_cols = [c for c in snapshot.index if c not in df.columns]
    if new_cols:
        df = df.copy()
        for c in new_cols:
            df[c] = pd.Series(index=df.index, dtype=float)

    if mode == "append_if_missing":
        if today in df.index:
            # Fill only missing values.
            df.loc[today, snapshot.index] = df.loc[today, snapshot.index].combine_first(snapshot)
            return df
        else:
            row = pd.DataFrame([snapshot], index=[today])
            return pd.concat([df, row], axis=0).sort_index()

    # Overwrite mode.
    df = df.copy()
    if today not in df.index:
        df.loc[today] = np.nan  # type: ignore
    df.loc[today, snapshot.index] = snapshot
    return df.sort_index()


def append_bloomberg_snapshot_live(
    df: pd.DataFrame,
    *,
    bbg_cfg: BloombergPullConfig,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Append/overwrite today's snapshot from Bloomberg on top of an existing
    history frame. This call is intentionally uncached by Streamlit so users
    can refresh intraday values without forcing a full history pull.
    """
    if df is None or df.empty:
        return df, {"snapshot_requested": True, "snapshot_non_null_count": 0, "snapshot_pulled_at": pd.Timestamp.now().isoformat()}

    tickers = [str(c) for c in df.columns]
    conn = BloombergConnection(host=bbg_cfg.host, port=bbg_cfg.port)
    with BloombergClient(conn) as bbg:
        snap = bbg.get_snapshot(tickers, field=bbg_cfg.field, chunk_size=max(500, bbg_cfg.chunk_size))
    out = _append_today_snapshot(df, snap, mode=bbg_cfg.snapshot_mode)
    meta = {
        "snapshot_requested": True,
        "snapshot_non_null_count": int(pd.to_numeric(snap, errors="coerce").notna().sum()),
        "snapshot_pulled_at": pd.Timestamp.now().isoformat(),
    }
    return out, meta


def load_bloomberg_dataset(
    reload_token: int,
    *,
    specs: Iterable[StrategySpec],
    bbg_cfg: BloombergPullConfig,
    cache_cfg: CacheConfig,
) -> DataLoadResult:
    """Load a wide Bloomberg dataset with cache and incremental updates."""
    _ = reload_token  # used by Streamlit to bust cache

    # Build required ticker universe.
    pairs, fx = required_pairs_and_fx(specs)
    futures_tickers = resolve_futures_universe(pairs, bbg_cfg=bbg_cfg, cache_cfg=cache_cfg)
    tickers = sorted(set(futures_tickers) | set(fx))

    universe_hash = _hash_universe(tickers)

    # Determine historical end date.
    hist_end = _resolve_hist_end_yyyymmdd(bbg_cfg)

    # Load cache.
    df_cached, meta_cached = _load_cache(cache_cfg)
    df = df_cached
    meta: Dict[str, Any] = dict(meta_cached or {})
    changed = False
    incremental_start_used: Optional[str] = None
    snapshot_non_null_count = 0
    snapshot_requested = bool(bbg_cfg.include_today_snapshot)
    snapshot_pulled_at: Optional[str] = None
    refresh_tickers: List[str] = tickers
    expired_cached_futures: List[str] = []

    # Ensure cached frame does not exceed the requested historical end.
    df, was_trimmed = _cap_history_to_end(df, hist_end)
    if was_trimmed:
        changed = True

    # Pull full or incremental history as needed.
    conn = BloombergConnection(host=bbg_cfg.host, port=bbg_cfg.port)
    with BloombergClient(conn) as bbg:
        if df is None:
            # Full pull.
            df = bbg.get_historical_timeseries(
                tickers,
                field=bbg_cfg.field,
                start_date_yyyymmdd=bbg_cfg.start_date_yyyymmdd,
                end_date_yyyymmdd=hist_end,
                chunk_size=bbg_cfg.chunk_size,
            )
            changed = True
        else:
            # Pull any missing ticker columns.
            missing_cols = [t for t in tickers if t not in df.columns]
            if missing_cols:
                df_missing = bbg.get_historical_timeseries(
                    missing_cols,
                    field=bbg_cfg.field,
                    start_date_yyyymmdd=bbg_cfg.start_date_yyyymmdd,
                    end_date_yyyymmdd=hist_end,
                    chunk_size=bbg_cfg.chunk_size,
                )
                df = _merge_wide_frames(df, df_missing)
                changed = True

            refresh_tickers, expired_cached_futures = _select_refresh_tickers(
                tickers,
                df,
                hist_end_yyyymmdd=hist_end,
                backfill_business_days=bbg_cfg.incremental_backfill_bdays,
                expired_contract_month_lag=bbg_cfg.expired_contract_month_lag,
                expired_contract_quiet_bdays=bbg_cfg.expired_contract_quiet_bdays,
            )

            # Incremental update.
            last_dt = pd.Timestamp(df.index.max()) if len(df.index) else None
            inc_start = _resolve_incremental_start_yyyymmdd(
                last_dt=last_dt,
                hist_end_yyyymmdd=hist_end,
                start_date_yyyymmdd=bbg_cfg.start_date_yyyymmdd,
                backfill_business_days=bbg_cfg.incremental_backfill_bdays,
            )
            incremental_start_used = inc_start
            if inc_start is not None and refresh_tickers:
                df_inc = bbg.get_historical_timeseries(
                    refresh_tickers,
                    field=bbg_cfg.field,
                    start_date_yyyymmdd=inc_start,
                    end_date_yyyymmdd=hist_end,
                    chunk_size=bbg_cfg.chunk_size,
                )
                if not df_inc.empty:
                    df = _merge_wide_frames(df, df_inc)
                    changed = True

        assert df is not None
        df = df.sort_index()
        # History-only frame to persist locally.
        df_cache, _ = _cap_history_to_end(df, hist_end)
        assert df_cache is not None

        refresh_tickers, expired_cached_futures = _select_refresh_tickers(
            tickers,
            df_cache,
            hist_end_yyyymmdd=hist_end,
            backfill_business_days=bbg_cfg.incremental_backfill_bdays,
            expired_contract_month_lag=bbg_cfg.expired_contract_month_lag,
            expired_contract_quiet_bdays=bbg_cfg.expired_contract_quiet_bdays,
        )

        # Optional today's snapshot append.
        if bbg_cfg.include_today_snapshot:
            snap = bbg.get_snapshot(refresh_tickers, field=bbg_cfg.field, chunk_size=max(500, bbg_cfg.chunk_size))
            snapshot_non_null_count = int(pd.to_numeric(snap, errors="coerce").notna().sum())
            snapshot_pulled_at = pd.Timestamp.now().isoformat()
            df = _append_today_snapshot(df, snap, mode=bbg_cfg.snapshot_mode)
            # Snapshot updates are intentionally not persisted in local cache.

    # Save cache if data changed.
    meta.update(
        {
            "source": "bloomberg",
            "universe_hash": universe_hash,
            "tickers_count": len(tickers),
            "tickers_sample": tickers[:20],
            "last_updated_at": pd.Timestamp.now().isoformat(),
            "hist_end_date": hist_end,
            "start_date": bbg_cfg.start_date_yyyymmdd,
            "generic_history_start": bbg_cfg.generic_history_start_yyyymmdd,
            "end_mode": bbg_cfg.end_mode,
            "include_today_snapshot": bool(bbg_cfg.include_today_snapshot),
            "incremental_backfill_bdays": int(bbg_cfg.incremental_backfill_bdays),
            "incremental_start_used": incremental_start_used,
            "expired_contract_month_lag": int(bbg_cfg.expired_contract_month_lag),
            "expired_contract_quiet_bdays": int(bbg_cfg.expired_contract_quiet_bdays),
            "refresh_tickers_count": len(refresh_tickers),
            "refresh_tickers_sample": refresh_tickers[:20],
            "expired_cached_futures_count": len(expired_cached_futures),
            "expired_cached_futures": expired_cached_futures,
            "expired_cached_futures_sample": expired_cached_futures[:20],
            "snapshot_requested": snapshot_requested,
            "snapshot_non_null_count": int(snapshot_non_null_count),
            "snapshot_pulled_at": snapshot_pulled_at,
        }
    )

    if changed:
        _save_cache(cache_cfg, df_cache, meta)
    else:
        _save_cache_meta(cache_cfg, meta)

    return DataLoadResult(df=df, meta=meta)


def load_file_dataset(
    reload_token: int,
    *,
    source: DataSource,
    csv_path: Optional[str] = None,
    parquet_path: Optional[str] = None,
) -> DataLoadResult:
    _ = reload_token
    if source == "parquet":
        if not parquet_path:
            raise ValueError("parquet_path is required")
        p = Path(parquet_path)
        if not p.exists():
            raise FileNotFoundError(f"Parquet not found: {p}")
        df = pd.read_parquet(p)
        return DataLoadResult(df=df, meta={"source": str(p), "format": "parquet"})
    if source == "csv":
        if not csv_path:
            raise ValueError("csv_path is required")
        p = Path(csv_path)
        if not p.exists():
            raise FileNotFoundError(f"CSV not found: {p}")
        df = pd.read_csv(p)
        return DataLoadResult(df=df, meta={"source": str(p), "format": "csv"})

    raise ValueError(f"Unsupported file source: {source}")


def load_dataset(
    reload_token: int,
    *,
    source: DataSource,
    specs: Optional[Iterable[StrategySpec]] = None,
    bbg_cfg: Optional[BloombergPullConfig] = None,
    cache_cfg: Optional[CacheConfig] = None,
    csv_path: Optional[str] = None,
    parquet_path: Optional[str] = None,
) -> DataLoadResult:
    """Main data-loading entry point used by the app."""
    if source == "bloomberg":
        if specs is None or bbg_cfg is None or cache_cfg is None:
            raise ValueError("Bloomberg mode requires specs, bbg_cfg, and cache_cfg.")
        return load_bloomberg_dataset(reload_token, specs=specs, bbg_cfg=bbg_cfg, cache_cfg=cache_cfg)

    return load_file_dataset(reload_token, source=source, csv_path=csv_path, parquet_path=parquet_path)
