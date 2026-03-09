
"""Bloomberg (blpapi) access helpers for futures and FX data pulls."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, List, Optional, Dict, Any, Set, Tuple
import re

import pandas as pd


def _require_blpapi():
    try:
        import blpapi  # type: ignore
        return blpapi
    except Exception as e:
        raise RuntimeError(
            "blpapi is not available. Install Bloomberg Python API and run with Terminal access.\n"
            "If you want to test the app without Bloomberg, use the CSV/Parquet data source."
        ) from e


def _chunks(items: List[str], n: int) -> List[List[str]]:
    if n <= 0:
        return [items]
    return [items[i : i + n] for i in range(0, len(items), n)]


@dataclass(frozen=True, slots=True)
class BloombergConnection:
    host: str = "localhost"
    port: int = 8194
    service_uri: str = "//blp/refdata"


class BloombergClient:
    """Small wrapper around `blpapi.Session` for reference and history requests."""

    def __init__(self, conn: BloombergConnection):
        self.conn = conn
        self._blpapi = _require_blpapi()
        self._session = None
        self._service = None

    def __enter__(self) -> "BloombergClient":
        options = self._blpapi.SessionOptions()
        options.setServerHost(self.conn.host)
        options.setServerPort(self.conn.port)

        self._session = self._blpapi.Session(options)
        if not self._session.start() or not self._session.openService(self.conn.service_uri):
            raise RuntimeError("Bloomberg Session/Service failed to start.")
        self._service = self._session.getService(self.conn.service_uri)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if self._session is not None:
                self._session.stop()
        finally:
            self._session = None
            self._service = None

    # -------------------------
    # Futures universe discovery
    # -------------------------

    @staticmethod
    def _normalize_root(root: str) -> str:
        r = root.upper().strip()
        return f"{r} " if len(r) == 1 else r

    def get_active_chain(self, root: str) -> Set[str]:
        """Return FUT_CHAIN tickers for `<root>1 Comdty`."""
        assert self._service is not None and self._session is not None

        req = self._service.createRequest("ReferenceDataRequest")
        req.append("securities", f"{self._normalize_root(root)}1 Comdty")
        req.append("fields", "FUT_CHAIN")
        self._session.sendRequest(req)

        results: Set[str] = set()
        while True:
            ev = self._session.nextEvent()
            for msg in ev:
                if msg.messageType() == "ReferenceDataResponse":
                    sec_data = msg.getElement("securityData").getValue(0)

                    # Skip securities with errors.
                    if sec_data.hasElement("securityError"):
                        continue

                    fdata = sec_data.getElement("fieldData")
                    if fdata.hasElement("FUT_CHAIN"):
                        chain = fdata.getElement("FUT_CHAIN")
                        for i in range(chain.numValues()):
                            results.add(chain.getValue(i).getElementAsString("Security Description").upper())

            if ev.eventType() == self._blpapi.Event.RESPONSE:
                break

        return results

    def get_historical_generic_tickers(self, root: str, start_date_yyyymmdd: str) -> Set[str]:
        """Return distinct FUT_CUR_GEN_TICKER values for `<root>1 Comdty`."""
        assert self._service is not None and self._session is not None

        req = self._service.createRequest("HistoricalDataRequest")
        req.append("securities", f"{self._normalize_root(root)}1 Comdty")
        req.append("fields", "FUT_CUR_GEN_TICKER")
        req.set("startDate", start_date_yyyymmdd)
        req.set("endDate", datetime.today().strftime("%Y%m%d"))
        self._session.sendRequest(req)

        results: Set[str] = set()
        while True:
            ev = self._session.nextEvent()
            for msg in ev:
                if msg.messageType() == "HistoricalDataResponse":
                    sec_node = msg.getElement("securityData")

                    # Skip securities with errors.
                    if sec_node.hasElement("securityError"):
                        continue

                    f_data = sec_node.getElement("fieldData")
                    for i in range(f_data.numValues()):
                        item = f_data.getValue(i)
                        if item.hasElement("FUT_CUR_GEN_TICKER"):
                            results.add(item.getElementAsString("FUT_CUR_GEN_TICKER").upper())

            if ev.eventType() == self._blpapi.Event.RESPONSE:
                break

        return results

    def get_combined_futures(
        self,
        root: str,
        *,
        month_code: Optional[str] = None,
        generic_history_start_yyyymmdd: Optional[str] = None,
    ) -> List[str]:
        """Return merged futures tickers from active chain and generic history."""
        if not generic_history_start_yyyymmdd:
            # Default to start of year, 10 years back.
            dt = (datetime.today() - timedelta(days=365 * 10)).replace(month=1, day=1)
            generic_history_start_yyyymmdd = dt.strftime("%Y%m%d")

        raw = self.get_active_chain(root) | self.get_historical_generic_tickers(root, generic_history_start_yyyymmdd)
        formatted = {f"{t.split(' COMDTY')[0].strip()} Comdty" for t in raw}

        if month_code:
            mc = month_code.upper().strip()
            # Allow optional space for one-letter roots.
            pattern = re.compile(rf"^{re.escape(root.strip())}\s?{mc}\d+", re.IGNORECASE)
            formatted = {t for t in formatted if pattern.match(t)}

        return sorted(formatted)

    # -------------------------
    # Time series
    # -------------------------

    def get_historical_timeseries(
        self,
        tickers: List[str],
        *,
        field: str = "PX_LAST",
        start_date_yyyymmdd: str = "19000101",
        end_date_yyyymmdd: Optional[str] = None,
        chunk_size: int = 200,
    ) -> pd.DataFrame:
        """Fetch daily history for many tickers and return a wide DataFrame."""
        assert self._service is not None and self._session is not None
        if not tickers:
            return pd.DataFrame()

        if end_date_yyyymmdd is None:
            end_date_yyyymmdd = (datetime.today() - timedelta(days=1)).strftime("%Y%m%d")

        all_rows: List[Dict[str, Any]] = []

        for chunk in _chunks(list(tickers), int(chunk_size)):
            req = self._service.createRequest("HistoricalDataRequest")
            sec_el = req.getElement("securities")
            for t in chunk:
                sec_el.appendValue(t)

            fields_el = req.getElement("fields")
            fields_el.appendValue(field)

            req.set("startDate", start_date_yyyymmdd)
            req.set("endDate", end_date_yyyymmdd)

            self._session.sendRequest(req)

            while True:
                ev = self._session.nextEvent()
                for msg in ev:
                    if msg.messageType() == "HistoricalDataResponse":
                        sec_node = msg.getElement("securityData")
                        ticker = sec_node.getElementAsString("security")

                        if sec_node.hasElement("securityError"):
                            continue

                        field_data = sec_node.getElement("fieldData")
                        for i in range(field_data.numValues()):
                            item = field_data.getValueAsElement(i)
                            if not item.hasElement(field):
                                continue
                            all_rows.append(
                                {
                                    "date": item.getElementAsDatetime("date"),
                                    "ticker": ticker,
                                    "value": item.getElementAsFloat(field),
                                }
                            )
                if ev.eventType() == self._blpapi.Event.RESPONSE:
                    break

        if not all_rows:
            return pd.DataFrame()

        df = pd.DataFrame(all_rows).pivot(index="date", columns="ticker", values="value")
        df.index = pd.to_datetime(df.index)
        return df.sort_index()

    def get_snapshot(
        self,
        tickers: List[str],
        *,
        field: str = "PX_LAST",
        chunk_size: int = 500,
    ) -> pd.Series:
        """Fetch snapshot values via `ReferenceDataRequest`."""
        assert self._service is not None and self._session is not None
        if not tickers:
            return pd.Series(dtype=float)

        out: Dict[str, float] = {}

        for chunk in _chunks(list(tickers), int(chunk_size)):
            req = self._service.createRequest("ReferenceDataRequest")
            sec_el = req.getElement("securities")
            for t in chunk:
                sec_el.appendValue(t)

            fld_el = req.getElement("fields")
            fld_el.appendValue(field)

            self._session.sendRequest(req)

            while True:
                ev = self._session.nextEvent()
                for msg in ev:
                    if msg.messageType() == "ReferenceDataResponse":
                        sec_data = msg.getElement("securityData")
                        for i in range(sec_data.numValues()):
                            sd = sec_data.getValue(i)
                            ticker = sd.getElementAsString("security")

                            if sd.hasElement("securityError"):
                                out[ticker] = float("nan")
                                continue

                            fdata = sd.getElement("fieldData")
                            if fdata.hasElement(field):
                                try:
                                    out[ticker] = float(fdata.getElementAsFloat(field))
                                except Exception:
                                    out[ticker] = float("nan")
                            else:
                                out[ticker] = float("nan")

                if ev.eventType() == self._blpapi.Event.RESPONSE:
                    break

        return pd.Series(out, name=field).sort_index()
