import unittest

import pandas as pd

from bbg import BloombergClient
from engine import FuturesComboEngine, LegSpec, StrategySpec


class TickerRootSupportTests(unittest.TestCase):
    def test_engine_parses_and_builds_three_letter_root(self) -> None:
        idx = pd.bdate_range("2026-03-02", periods=5)
        df = pd.DataFrame({"OSIK26 Comdty": [50.0, 51.0, 52.0, 53.0, 54.0]}, index=idx)

        engine = FuturesComboEngine(df)
        self.assertEqual(engine.available_contract_years("OSI", "K"), [2026])

        spec = StrategySpec(
            name="Oil share (K)",
            legs=[
                LegSpec(
                    alias="OSIK",
                    ticker_root="OSI",
                    month_code="K",
                    multiplier=1.0,
                    year_shift=0,
                    uom_mul=1.0,
                    currency="USD",
                    fx_ticker=None,
                )
            ],
            output_currency="USD",
            min_obs=1,
        )

        curves, report = engine.build_with_report(spec, fill=None, multiindex_cols=False)
        self.assertIn(2026, curves.columns)
        self.assertEqual(float(curves.loc[idx[-1], 2026]), 54.0)
        self.assertEqual(report.shape[0], 1)
        self.assertEqual(int(report.loc[0, "anchor_year"]), 2026)
        self.assertEqual(str(report.loc[0, "status"]), "ok")

    def test_bloomberg_root_normalization_keeps_multi_letter_root(self) -> None:
        self.assertEqual(BloombergClient._normalize_root("S"), "S ")
        self.assertEqual(BloombergClient._normalize_root("OSI"), "OSI")


if __name__ == "__main__":
    unittest.main()
