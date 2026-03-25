import unittest

import pandas as pd

from engine import FuturesComboEngine, LegSpec, StrategySpec


class BuildReportEmptyTests(unittest.TestCase):
    def test_build_with_report_returns_empty_report_columns_when_no_anchor_years(self) -> None:
        idx = pd.to_datetime(["2025-01-02", "2025-01-03"])
        df = pd.DataFrame({"IJK25 Comdty": [500.0, 501.0]}, index=idx)

        engine = FuturesComboEngine(df)
        spec = StrategySpec(
            name="Missing spread leg",
            output_currency="USD",
            min_obs=1,
            value_source="contract",
            legs=[
                LegSpec(alias="A", ticker_root="IJ", month_code="K", multiplier=1),
                LegSpec(alias="B", ticker_root="RS", month_code="K", multiplier=-1),
            ],
        )

        curves, report = engine.build_with_report(spec, multiindex_cols=False)

        self.assertTrue(curves.empty)
        self.assertEqual(list(report.columns), ["anchor_year", "status", "reason"])
        self.assertTrue(report.empty)


if __name__ == "__main__":
    unittest.main()
