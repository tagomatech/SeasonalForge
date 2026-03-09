
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Dict, Tuple, List, Iterable, Union, Any
from collections import defaultdict, deque
import re

import numpy as np
import pandas as pd


# =========================
# Parsing helpers / constants
# =========================

MONTH_CODE_TO_NUM: Dict[str, int] = {
    "F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
    "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12,
}

# Bloomberg futures column examples:
#   "IJG25 Comdty"
#   "IJG6 Comdty"
#   "S H25 Comdty"
_FUT_COL_RE = re.compile(
    r"^\s*(?P<root>[A-Z0-9]{1,15})\s*(?P<month>[FGHJKMNQUVXZ])(?P<year>\d{1,2})\s+Comdty\s*$",
    re.IGNORECASE,
)

# Bloomberg FX column example:
#   "EURUSD Curncy"
_FX_COL_RE = re.compile(
    r"^\s*(?P<base>[A-Z]{3})(?P<quote>[A-Z]{3})\s+Curncy\s*$",
    re.IGNORECASE,
)


def _as_datetime_index(df: pd.DataFrame, date_col: Optional[str]) -> pd.DataFrame:
    """Ensure df has a DatetimeIndex; if date_col is given, set it as index."""
    if date_col is not None and date_col in df.columns:
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)

    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)

    return df.sort_index()


def _infer_asof_year(idx: pd.DatetimeIndex) -> int:
    return int(idx.max().year) if len(idx) else int(pd.Timestamp.utcnow().year)


def _expand_year_2digit(yy: int, pivot: int = 80) -> int:
    """Map 2-digit year to 19xx/20xx. Default pivot=80 -> 79->2079, 80->1980."""
    return (2000 + yy) if yy < pivot else (1900 + yy)


def _nearest_year_with_last_digit(digit: int, ref_year: int) -> int:
    """Nearest year to ref_year whose last digit equals digit."""
    decade = (ref_year // 10) * 10
    candidates = [decade - 10 + digit, decade + digit, decade + 10 + digit]
    return int(min(candidates, key=lambda y: abs(y - ref_year)))


def _sanitize_alias(s: str) -> str:
    alias = re.sub(r"[^A-Za-z0-9_]+", "_", str(s).strip())
    alias = re.sub(r"_{2,}", "_", alias).strip("_")
    if not alias:
        alias = "leg"
    if alias[0].isdigit():
        alias = f"l_{alias}"
    return alias


# =========================
# User-facing specs
# =========================

@dataclass(frozen=True, slots=True)
class FillPolicy:
    """Holiday-gap fill policy for legs and FX rates."""
    method: Optional[str] = None
    limit: int = 1
    apply_to_fx: bool = True

    def __post_init__(self) -> None:
        if self.method is None:
            return
        m = str(self.method).lower().strip()
        if m not in {"ffill"}:
            raise ValueError(f"Unsupported FillPolicy.method: {self.method!r}")
        if self.limit < 0:
            raise ValueError("FillPolicy.limit must be >= 0")


@dataclass(frozen=True, slots=True)
class LegSpec:
    """One strategy leg with contract mapping and conversion settings."""
    alias: str
    ticker_root: str
    month_code: str
    multiplier: float = 1.0
    year_shift: int = 0
    uom_mul: float = 1.0
    currency: Optional[str] = None
    fx_ticker: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "alias", _sanitize_alias(self.alias))
        object.__setattr__(self, "ticker_root", self.ticker_root.strip().upper())
        object.__setattr__(self, "month_code", self.month_code.strip().upper())
        if self.currency is not None:
            object.__setattr__(self, "currency", self.currency.strip().upper())

        if self.month_code not in MONTH_CODE_TO_NUM:
            raise ValueError(f"Invalid month_code: {self.month_code!r}")


@dataclass(frozen=True, slots=True)
class StrategySpec:
    """Strategy definition for linear or expression-based combinations."""
    name: str
    legs: Sequence[LegSpec]
    expression: Optional[str] = None
    output_currency: str = "USD"
    min_obs: int = 5
    value_source: str = "contract"  # "contract" | "calculated"

    def __post_init__(self) -> None:
        aliases = [l.alias for l in self.legs]
        if len(set(aliases)) != len(aliases):
            raise ValueError(f"Leg aliases must be unique within a strategy: {aliases}")
        if self.min_obs < 1:
            raise ValueError("min_obs must be >= 1")

        source = str(self.value_source).strip().lower()
        if source not in {"contract", "calculated"}:
            raise ValueError("value_source must be 'contract' or 'calculated'")
        object.__setattr__(self, "value_source", source)


# =========================
# Internal structures
# =========================

@dataclass(frozen=True, slots=True)
class _ContractKey:
    root: str
    month: str
    year: int  # full year (e.g. 2026)


@dataclass(frozen=True, slots=True)
class _Block:
    col: str
    start: pd.Timestamp
    end: pd.Timestamp
    nobs: int


# =========================
# Engine
# =========================

class FuturesComboEngine:
    """Futures combination engine for spread and multi-leg seasonal analysis."""

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        date_col: Optional[str] = None,
        asof_year: Optional[int] = None,
        ambiguous_gap_days: int = 180,
        year_pivot: int = 80,
    ) -> None:
        self.df = _as_datetime_index(df, date_col=date_col)
        self.asof_year = int(asof_year) if asof_year is not None else _infer_asof_year(self.df.index)
        self.ambiguous_gap_days = int(ambiguous_gap_days)
        self.year_pivot = int(year_pivot)

        self._nonnull = self.df.notna().sum()

        # Futures maps
        self._fixed: Dict[_ContractKey, List[str]] = defaultdict(list)     # 2-digit years
        self._amb_cols: Dict[Tuple[str, str, int], List[str]] = defaultdict(list)  # (root, month, last_digit)
        self._amb_blocks: Dict[_ContractKey, List[_Block]] = defaultdict(list)     # resolved ambiguous blocks

        # FX maps
        self._fx_cols: Dict[Tuple[str, str], List[str]] = defaultdict(list)

        # Caches
        self._contract_cache: Dict[_ContractKey, pd.Series] = {}
        self._fx_cache: Dict[Tuple[str, str, Optional[FillPolicy]], pd.Series] = {}

        self._parse_columns()
        self._resolve_ambiguous_columns()

    # -------- Parsing --------

    def _parse_columns(self) -> None:
        for col in self.df.columns:
            col_s = str(col).strip()

            m_fx = _FX_COL_RE.match(col_s)
            if m_fx:
                base = m_fx.group("base").upper()
                quote = m_fx.group("quote").upper()
                self._fx_cols[(base, quote)].append(col_s)
                continue

            m_fut = _FUT_COL_RE.match(col_s)
            if not m_fut:
                continue

            root = m_fut.group("root").upper().strip()
            month = m_fut.group("month").upper()
            year_code = m_fut.group("year").strip()

            if len(year_code) == 2:
                yy = int(year_code)
                year = _expand_year_2digit(yy, pivot=self.year_pivot)
                self._fixed[_ContractKey(root, month, year)].append(col_s)
            else:
                digit = int(year_code)
                self._amb_cols[(root, month, digit)].append(col_s)

    def _best_col(self, cols: List[str]) -> str:
        if len(cols) == 1:
            return cols[0]
        return max(cols, key=lambda c: float(self._nonnull.get(c, 0)))

    def _resolve_ambiguous_columns(self) -> None:
        """
        Split each 1-digit year column into blocks separated by long NaN gaps,
        then map each block to a full-year contract key.
        """
        if not self._amb_cols:
            return

        for (root, month, digit), cols in self._amb_cols.items():
            delivery_month = MONTH_CODE_TO_NUM.get(month)

            for col in cols:
                s = pd.to_numeric(self.df[col], errors="coerce")
                nn = s.dropna()
                if nn.empty:
                    continue

                idx = nn.index.sort_values()
                diffs = idx.to_series().diff().dt.days.fillna(0)
                block_id = (diffs > self.ambiguous_gap_days).cumsum()

                for _, block_idx in idx.to_series().groupby(block_id):
                    start = pd.Timestamp(block_idx.iloc[0])
                    end = pd.Timestamp(block_idx.iloc[-1])
                    nobs = int(len(block_idx))

                    ref_year = int(end.year)
                    if delivery_month is not None and int(end.month) > int(delivery_month):
                        ref_year += 1

                    year = _nearest_year_with_last_digit(digit, ref_year)
                    key = _ContractKey(root, month, year)
                    self._amb_blocks[key].append(_Block(col=col, start=start, end=end, nobs=nobs))

        for key, blocks in list(self._amb_blocks.items()):
            self._amb_blocks[key] = sorted(blocks, key=lambda b: b.nobs, reverse=True)

    # -------- Public discovery helpers --------

    def available_contract_years(self, ticker_root: str, month_code: str) -> List[int]:
        """All contract years resolvable for (root, month)."""
        r = ticker_root.strip().upper()
        m = month_code.strip().upper()
        years = {k.year for k in self._fixed.keys() if k.root == r and k.month == m}
        years |= {k.year for k in self._amb_blocks.keys() if k.root == r and k.month == m}
        return sorted(years)

    def explain_contract(self, ticker_root: str, month_code: str, year: int) -> Dict[str, Any]:
        """Explain how a specific contract is resolved (fixed vs ambiguous block)."""
        key = _ContractKey(ticker_root.strip().upper(), month_code.strip().upper(), int(year))
        if key in self._fixed:
            cols = self._fixed[key]
            return {"type": "fixed", "key": key, "cols": cols, "chosen": self._best_col(cols)}

        if key in self._amb_blocks:
            b = self._amb_blocks[key][0]
            return {"type": "ambiguous", "key": key, "col": b.col, "start": b.start, "end": b.end, "nobs": b.nobs}

        return {"type": "missing", "key": key}

    # -------- Contract series access --------

    def _contract_series(self, root: str, month: str, year: int) -> Optional[pd.Series]:
        """Price series for a specific contract, aligned to df.index."""
        key = _ContractKey(root, month, int(year))
        if key in self._contract_cache:
            return self._contract_cache[key]

        cols = self._fixed.get(key)
        if cols:
            col = self._best_col(cols)
            s = pd.to_numeric(self.df[col], errors="coerce")
            self._contract_cache[key] = s
            return s

        blocks = self._amb_blocks.get(key)
        if not blocks:
            return None

        b = blocks[0]
        s = pd.to_numeric(self.df[b.col], errors="coerce")
        s = s.where((s.index >= b.start) & (s.index <= b.end))
        self._contract_cache[key] = s
        return s

    # -------- FX graph --------

    def fx_rate(self, from_ccy: str, to_ccy: str, *, fill: Optional[FillPolicy] = None) -> pd.Series:
        """
        Series converting 1 unit of from_ccy into to_ccy.
        Supports direct, inverse, and multi-step cross rates.
        """
        from_ccy = from_ccy.upper().strip()
        to_ccy = to_ccy.upper().strip()
        if from_ccy == to_ccy:
            return pd.Series(1.0, index=self.df.index)

        cache_key = (from_ccy, to_ccy, fill)
        if cache_key in self._fx_cache:
            return self._fx_cache[cache_key]

        graph: Dict[str, List[Tuple[str, pd.Series]]] = defaultdict(list)
        for (base, quote), cols in self._fx_cols.items():
            col = self._best_col(cols)
            rate = pd.to_numeric(self.df[col], errors="coerce")
            graph[base].append((quote, rate))
            inv = 1.0 / rate.replace({0.0: np.nan})
            graph[quote].append((base, inv))

        ones = pd.Series(1.0, index=self.df.index)
        visited = {from_ccy}
        q = deque([(from_ccy, ones)])
        found = None

        while q and found is None:
            ccy, acc = q.popleft()
            for nxt, edge in graph.get(ccy, []):
                if nxt in visited:
                    continue
                new_acc = acc * edge
                if nxt == to_ccy:
                    found = new_acc
                    break
                visited.add(nxt)
                q.append((nxt, new_acc))

        if found is None:
            available = sorted({c for pair in self._fx_cols.keys() for c in pair})
            raise KeyError(
                f"No FX conversion path from {from_ccy} to {to_ccy}. "
                f"Available currencies in df: {available}"
            )

        if fill is not None and fill.method == "ffill" and fill.apply_to_fx:
            found = self._fill_business_days(found, limit=fill.limit)

        self._fx_cache[cache_key] = found
        return found

    # -------- Filling --------

    @staticmethod
    def _fill_business_days(s: pd.Series, *, limit: int) -> pd.Series:
        """
        Forward-fill only across Mon-Fri rows (weekends remain NaN),
        treating Monday-after-weekend as "next business day".
        Do not fill trailing tail NaNs after the latest real observation.
        """
        if s.empty:
            return s
        is_bd = s.index.dayofweek < 5
        s_bd_raw = s.loc[is_bd]
        s_bd = s_bd_raw.ffill(limit=int(limit))
        last_real = s_bd_raw.last_valid_index()
        if last_real is not None:
            tail_mask = s_bd.index > pd.Timestamp(last_real)
            if bool(tail_mask.any()):
                s_bd.loc[tail_mask] = s_bd_raw.loc[tail_mask]
        out = s.copy()
        out.loc[s_bd.index] = s_bd
        return out

    # -------- Build core --------

    def _infer_leg_currency(self, leg: LegSpec) -> Optional[str]:
        if leg.currency:
            return leg.currency
        if leg.fx_ticker:
            m = _FX_COL_RE.match(leg.fx_ticker.strip())
            if m:
                return m.group("base").upper()
        return None

    def _leg_fx_to_output(self, leg: LegSpec, output_ccy: str, *, fill: Optional[FillPolicy]) -> pd.Series:
        leg_ccy = self._infer_leg_currency(leg)
        if leg_ccy is None:
            return pd.Series(1.0, index=self.df.index)
        return self.fx_rate(leg_ccy, output_ccy, fill=fill)

    def _anchor_years(self, spec: StrategySpec) -> List[int]:
        sets: List[set[int]] = []
        for leg in spec.legs:
            years = set(self.available_contract_years(leg.ticker_root, leg.month_code))
            anchors = {y - int(leg.year_shift) for y in years}
            sets.append(anchors)
        return sorted(set.intersection(*sets)) if sets else []

    def build_with_report(
        self,
        spec: StrategySpec,
        *,
        anchor_years: Optional[Iterable[int]] = None,
        fill: Optional[FillPolicy] = None,
        multiindex_cols: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Build curves and a status report by anchor year."""
        fill = fill if (fill is not None and fill.method is not None) else None
        out_ccy = spec.output_currency.upper().strip()

        anchors = list(anchor_years) if anchor_years is not None else self._anchor_years(spec)
        anchors = sorted(set(int(a) for a in anchors))

        curves: Dict[int, pd.Series] = {}
        report_rows: List[Dict[str, Any]] = []

        for a in anchors:
            leg_series: Dict[str, pd.Series] = {}
            status = "ok"
            reason = ""

            for leg in spec.legs:
                y = int(a) + int(leg.year_shift)
                s = self._contract_series(leg.ticker_root, leg.month_code, y)
                if s is None:
                    status = "skipped"
                    reason = f"missing_contract({leg.ticker_root}{leg.month_code}{y})"
                    break

                if int(s.notna().sum()) < int(spec.min_obs):
                    status = "skipped"
                    reason = f"too_few_obs({leg.alias}, nobs={int(s.notna().sum())})"
                    break

                s = s * float(leg.uom_mul)

                if fill is not None and fill.method == "ffill":
                    s = self._fill_business_days(s, limit=fill.limit)

                try:
                    fx = self._leg_fx_to_output(leg, out_ccy, fill=fill)
                except KeyError as e:
                    status = "skipped"
                    reason = f"missing_fx({leg.alias}): {e}"
                    break

                s = s * fx
                s = s * float(leg.multiplier)

                leg_series[leg.alias] = s

            if status != "ok" or not leg_series:
                report_rows.append({"anchor_year": int(a), "status": status, "reason": reason})
                continue

            try:
                if spec.expression:
                    legs_df = pd.concat(leg_series, axis=1)
                    val = legs_df.eval(spec.expression, engine="python")
                    val = pd.to_numeric(val, errors="coerce")
                else:
                    val = None
                    for s in leg_series.values():
                        val = s if val is None else (val + s)
                    val = pd.to_numeric(val, errors="coerce")
            except Exception as e:
                report_rows.append({"anchor_year": int(a), "status": "skipped", "reason": f"expression_error: {e}"})
                continue

            curves[int(a)] = val.rename(int(a))
            report_rows.append({"anchor_year": int(a), "status": "ok", "reason": ""})

        if curves:
            res = pd.concat(curves, axis=1).sort_index(axis=1)
            res.columns.name = "anchor_year"
            if multiindex_cols:
                res.columns = pd.MultiIndex.from_product([[spec.name], res.columns], names=["strategy", "anchor_year"])
        else:
            res = pd.DataFrame(index=self.df.index)
            if multiindex_cols:
                res.columns = pd.MultiIndex.from_tuples([], names=["strategy", "anchor_year"])
            else:
                res.columns = pd.Index([], name="anchor_year")

        report = pd.DataFrame(report_rows).sort_values("anchor_year").reset_index(drop=True)
        return res, report

    def build(
        self,
        spec: StrategySpec,
        *,
        anchor_years: Optional[Iterable[int]] = None,
        fill: Optional[FillPolicy] = None,
        multiindex_cols: bool = False,
    ) -> pd.DataFrame:
        """Build curves only."""
        curves, _ = self.build_with_report(
            spec,
            anchor_years=anchor_years,
            fill=fill,
            multiindex_cols=multiindex_cols,
        )
        return curves

    def build_legs(self, spec: StrategySpec, *, anchor_year: int, fill: Optional[FillPolicy] = None) -> pd.DataFrame:
        """Diagnostic: per-leg transformed series for one anchor_year."""
        fill = fill if (fill is not None and fill.method is not None) else None
        out_ccy = spec.output_currency.upper().strip()
        a = int(anchor_year)

        cols: Dict[str, pd.Series] = {}
        for leg in spec.legs:
            y = a + int(leg.year_shift)
            s = self._contract_series(leg.ticker_root, leg.month_code, y)
            if s is None:
                raise KeyError(f"Missing contract series for leg={leg.alias} at contract_year={y}")

            s = s * float(leg.uom_mul)
            if fill is not None and fill.method == "ffill":
                s = self._fill_business_days(s, limit=fill.limit)

            fx = self._leg_fx_to_output(leg, out_ccy, fill=fill)
            s = s * fx
            s = s * float(leg.multiplier)
            cols[leg.alias] = s

        return pd.concat(cols, axis=1)

    def build_leg_contract_values(
        self,
        spec: StrategySpec,
        *,
        anchor_year: int,
        fill: Optional[FillPolicy] = None,
    ) -> pd.DataFrame:
        """
        Per-leg raw futures contract values for one anchor year.
        Column labels include leg alias, resolved contract code, and source ticker.
        """
        fill = fill if (fill is not None and fill.method is not None) else None
        a = int(anchor_year)

        cols: Dict[str, pd.Series] = {}
        for leg in spec.legs:
            y = a + int(leg.year_shift)
            s = self._contract_series(leg.ticker_root, leg.month_code, y)
            if s is None:
                raise KeyError(f"Missing contract series for leg={leg.alias} at contract_year={y}")

            if fill is not None and fill.method == "ffill":
                s = self._fill_business_days(s, limit=fill.limit)

            contract_code = f"{leg.ticker_root}{leg.month_code}{y}"
            resolved = self.explain_contract(leg.ticker_root, leg.month_code, y)
            source = ""
            if resolved.get("type") == "fixed":
                source = str(resolved.get("chosen", ""))
            elif resolved.get("type") == "ambiguous":
                source = str(resolved.get("col", ""))

            if source:
                label = f"{leg.alias} ({contract_code}) [{source}]"
            else:
                label = f"{leg.alias} ({contract_code})"
            cols[label] = pd.to_numeric(s, errors="coerce")

        return pd.concat(cols, axis=1)

    def availability(self, curves: pd.DataFrame) -> pd.DataFrame:
        """First/last valid observation per curve, plus observation count."""
        df = curves
        if isinstance(df.columns, pd.MultiIndex) and df.columns.nlevels == 2:
            rows = []
            for strat, ay in df.columns:
                s = pd.to_numeric(df[(strat, ay)], errors="coerce")
                rows.append(
                    dict(strategy=strat, anchor_year=int(ay),
                         first_valid=s.first_valid_index(), last_valid=s.last_valid_index(),
                         nobs=int(s.notna().sum()))
                )
            return pd.DataFrame(rows).sort_values(["strategy", "anchor_year"])
        else:
            rows = []
            for ay in df.columns:
                s = pd.to_numeric(df[ay], errors="coerce")
                rows.append(
                    dict(anchor_year=int(ay),
                         first_valid=s.first_valid_index(), last_valid=s.last_valid_index(),
                         nobs=int(s.notna().sum()))
                )
            return pd.DataFrame(rows).sort_values(["anchor_year"])

    def to_long(self, curves: pd.DataFrame, *, strategy_name: Optional[str] = None) -> pd.DataFrame:
        """Wide curves -> tidy long frame: date, strategy, anchor_year, value."""
        if isinstance(curves.columns, pd.MultiIndex) and curves.columns.nlevels == 2:
            tmp = curves.stack(level=[0, 1]).rename("value").reset_index()
            tmp.columns = ["date", "strategy", "anchor_year", "value"]
            return tmp
        tmp = curves.stack().rename("value").reset_index()
        tmp.columns = ["date", "anchor_year", "value"]
        tmp.insert(1, "strategy", strategy_name)
        return tmp

    # =========================
    # Overlay views
    # =========================

    def overlay_seasonal(
        self,
        curves: pd.DataFrame,
        *,
        axis: str = "month_day",       # "month_day" or "day_of_year"
        window: Union[str, int, None] = "1Y",
        anchor: str = "last_valid",    # "last_valid" or "first_valid"
        drop_leap_day: bool = True,
    ) -> pd.DataFrame:
        """Build a seasonal overlay matrix on month/day or day-of-year axis."""
        if isinstance(curves.columns, pd.MultiIndex) and curves.columns.nlevels == 2:
            combos = curves.columns.get_level_values(0).unique()
            if len(combos) != 1:
                raise ValueError("overlay_seasonal expects curves for one strategy at a time.")
            df = curves[combos[0]]
        else:
            df = curves

        if axis not in {"month_day", "day_of_year"}:
            raise ValueError("axis must be 'month_day' or 'day_of_year'")

        def _wdays(w):
            if w is None:
                return None
            if isinstance(w, int):
                return int(w)
            w = str(w).strip().upper()
            if w.endswith("Y"):
                yrs = int(w[:-1] or "1")
                return 365 * yrs
            if w.endswith("D"):
                return int(w[:-1])
            raise ValueError(f"Unsupported window={window!r}")

        wdays = _wdays(window)

        out_cols = {}
        for ay in df.columns:
            s = pd.to_numeric(df[ay], errors="coerce")
            if s.notna().sum() == 0:
                continue

            if anchor == "last_valid":
                end = s.last_valid_index()
                if end is None:
                    continue
                start = (end - pd.Timedelta(days=wdays - 1)) if wdays is not None else None
                s2 = s.loc[start:end] if start is not None else s.loc[:end]
            elif anchor == "first_valid":
                start = s.first_valid_index()
                if start is None:
                    continue
                end = (start + pd.Timedelta(days=wdays - 1)) if wdays is not None else None
                s2 = s.loc[start:end] if end is not None else s.loc[start:]
            else:
                raise ValueError("anchor must be 'last_valid' or 'first_valid'")

            s2 = s2.dropna()
            if s2.empty:
                continue

            if axis == "month_day":
                idx = s2.index.strftime("%m-%d")
                if drop_leap_day:
                    mask = idx != "02-29"
                    idx = idx[mask]
                    vals = s2.values[mask]
                else:
                    vals = s2.values
                out = pd.Series(vals, index=pd.Index(idx, name="month_day"))
            else:
                idx = s2.index.dayofyear
                if drop_leap_day:
                    mask = s2.index.strftime("%m-%d") != "02-29"
                    idx = idx[mask]
                    vals = s2.values[mask]
                else:
                    vals = s2.values
                out = pd.Series(vals, index=pd.Index(idx, name="day_of_year"))

            out = out[~out.index.duplicated(keep="last")]
            out_cols[int(ay)] = out

        return pd.concat(out_cols, axis=1).sort_index() if out_cols else pd.DataFrame()

    def overlay_lifecycle_window(
        self,
        curves: pd.DataFrame,
        *,
        window_months: Tuple[int, int] = (-9, 6),
        asof: Optional[Union[str, pd.Timestamp]] = None,
        reference: Union[str, int] = "front",
        active_buffer_days: int = 10,
        axis: str = "t",
        alignment: str = "first_valid",  # "first_valid" | "asof_aligned"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Align curves on a lifecycle axis around an as-of date.

        alignment:
        - "first_valid": absolute trading-day count since each curve first valid.
        - "asof_aligned": align each curve on its own comparable ASOF date
          relative to the reference anchor year.
        """
        if isinstance(curves.columns, pd.MultiIndex) and curves.columns.nlevels == 2:
            combos = curves.columns.get_level_values(0).unique()
            if len(combos) != 1:
                raise ValueError("overlay_lifecycle_window expects curves for one strategy at a time.")
            df = curves[combos[0]]
        else:
            df = curves

        align_mode = str(alignment).lower().strip()
        if align_mode not in {"first_valid", "asof_aligned"}:
            raise ValueError("alignment must be 'first_valid' or 'asof_aligned'")

        start_m, end_m = int(window_months[0]), int(window_months[1])
        if start_m > end_m:
            raise ValueError("window_months must satisfy start_m <= end_m")

        # asof defaults to last date in dataset
        if asof is None:
            asof_ts = df.index.max()
        else:
            asof_ts = pd.Timestamp(asof)
            idx = df.index[df.index <= asof_ts]
            if idx.empty:
                raise ValueError("asof is before the start of curves index.")
            asof_ts = idx.max()

        # Identify active curves near asof
        last_valid: Dict[int, Optional[pd.Timestamp]] = {}
        for ay in df.columns:
            s = pd.to_numeric(df[ay], errors="coerce").loc[:asof_ts]
            last_valid[int(ay)] = s.last_valid_index()

        active = [
            ay for ay, lv in last_valid.items()
            if lv is not None and (asof_ts - lv).days <= int(active_buffer_days)
        ]

        # Choose reference anchor year
        if isinstance(reference, int):
            ref_ay = int(reference)
        else:
            ref_mode = str(reference).lower().strip()
            if active:
                ref_ay = min(active) if ref_mode == "front" else max(active)
            else:
                ref_ay = max(
                    last_valid.items(),
                    key=lambda kv: (kv[1] if kv[1] is not None else pd.Timestamp.min),
                )[0]

        ref_s = pd.to_numeric(df[ref_ay], errors="coerce").loc[:asof_ts].dropna()
        if ref_s.empty:
            raise ValueError(f"Reference curve {ref_ay} has no data up to asof={asof_ts.date()}.")

        ref_dates = ref_s.index
        t_asof = len(ref_dates) - 1

        # Translate month window into dates
        window_start_date = asof_ts + pd.DateOffset(months=start_m)
        window_end_date = asof_ts + pd.DateOffset(months=end_m)

        # Convert window_start_date to a t_start by looking at reference trading dates
        if window_start_date <= ref_dates.min():
            t_start = 0
        else:
            # first t whose date >= window_start_date
            t_start = int(np.searchsorted(ref_dates.values, np.datetime64(window_start_date), side="left"))
            t_start = min(max(t_start, 0), t_asof)

        # Convert end window to extra business days beyond asof (approximation for future)
        extra_bdays = 0
        if window_end_date > asof_ts:
            future_bdays = pd.bdate_range(asof_ts, window_end_date)
            extra_bdays = max(len(future_bdays) - 1, 0)

        t_end = t_asof + extra_bdays

        # Build calendar for t in [t_start..t_end]
        t_index = np.arange(t_start, t_end + 1)
        dates: List[pd.Timestamp] = []
        for t in t_index:
            if t <= t_asof:
                dates.append(pd.Timestamp(ref_dates[t]))
            else:
                # future business days after asof
                k = t - t_asof  # 1..extra_bdays
                d = pd.bdate_range(asof_ts, window_end_date)[k]
                dates.append(pd.Timestamp(d))

        offsets = [int(d.year) - int(ref_ay) for d in dates]
        labels = [f"{d:%d-%m} Year {off:+d}" for d, off in zip(dates, offsets)]
        calendar = pd.DataFrame(
            {
                "t": t_index,
                "date": dates,
                "label": labels,
                "year_offset": offsets,
                "is_asof": [t == t_asof for t in t_index],
                "is_future": [t > t_asof for t in t_index],
            }
        )

        # Build aligned matrix for the chosen t range.
        mat: Dict[int, np.ndarray] = {}
        L = len(t_index)

        for ay in df.columns:
            # Each curve only uses data available up to the selected ASOF date.
            s = pd.to_numeric(df[ay], errors="coerce").loc[:asof_ts].dropna()
            vals = np.full(L, np.nan, dtype=float)
            if not s.empty:
                if align_mode == "first_valid":
                    # Absolute lifecycle mode:
                    # t=0 is each curve's first valid point.
                    src_start = max(int(t_start), 0)
                    src_end = min(int(t_end), len(s) - 1)
                    if src_start <= src_end:
                        dst_start = src_start - int(t_start)
                        dst_end = dst_start + (src_end - src_start)
                        vals[dst_start : dst_end + 1] = s.iloc[src_start : src_end + 1].to_numpy(dtype=float)
                else:
                    # ASOF-aligned mode:
                    # Map each seasonal-axis date into the curve's anchor year,
                    # then sample the latest available value on/before that date.
                    # This keeps the alignment calendar-true and avoids forcing
                    # future anchor years to terminate at the current ASOF point.
                    target_dates = pd.DatetimeIndex(
                        [
                            pd.Timestamp(d) + pd.DateOffset(years=int(ay) - int(ref_ay))
                            for d in dates
                        ]
                    )

                    src_idx = s.index
                    src_vals = s.to_numpy(dtype=float)
                    src_min = pd.Timestamp(src_idx.min())
                    src_max = pd.Timestamp(src_idx.max())

                    valid_target = (
                        (target_dates >= src_min)
                        & (target_dates <= src_max)
                        & (target_dates <= pd.Timestamp(asof_ts))
                    )
                    if bool(np.any(valid_target)):
                        pos = np.searchsorted(src_idx.values, target_dates.values, side="right") - 1
                        ok = (pos >= 0) & valid_target
                        if bool(np.any(ok)):
                            vals[ok] = src_vals[pos[ok]]
            mat[int(ay)] = vals

        matrix = pd.DataFrame(mat, index=pd.Index(t_index, name="t")).sort_index(axis=1)

        if axis == "alias":
            matrix = matrix.copy()
            matrix.index = pd.Index(calendar["label"].values, name="alias_date")
        elif axis != "t":
            raise ValueError("axis must be 't' or 'alias'")

        meta = {
            "reference_anchor_year": int(ref_ay),
            "asof": pd.Timestamp(asof_ts),
            "t_asof": int(t_asof),
            "t_start": int(t_start),
            "t_end": int(t_end),
            "window_start_date": pd.Timestamp(window_start_date),
            "window_end_date": pd.Timestamp(window_end_date),
            "alignment": align_mode,
        }

        return matrix, calendar, meta




