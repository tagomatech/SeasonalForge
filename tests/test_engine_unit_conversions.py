import unittest

import pandas as pd

from engine import FuturesComboEngine, LegSpec, StrategySpec


class UnitConversionTests(unittest.TestCase):
    def test_cme_soybean_crush_uses_cme_conversion_factors(self) -> None:
        idx = pd.bdate_range("2026-05-04", periods=5)
        df = pd.DataFrame(
            {
                "S K26 Comdty": [1037.5, 1037.5, 1037.5, 1037.5, 1037.5],
                "SMK26 Comdty": [300.0, 300.0, 300.0, 300.0, 300.0],
                "BOK26 Comdty": [40.0, 40.0, 40.0, 40.0, 40.0],
            },
            index=idx,
        )
        engine = FuturesComboEngine(df)
        spec = StrategySpec(
            name="CME Soybean Crush (SK, SMK, BOK)",
            output_currency="USD",
            min_obs=1,
            expression="SM + BO - S",
            value_source="calculated",
            legs=[
                LegSpec(
                    alias="S",
                    ticker_root="S",
                    month_code="K",
                    multiplier=1.0,
                    year_shift=0,
                    uom_mul=0.01,
                    currency="USD",
                    fx_ticker=None,
                ),
                LegSpec(
                    alias="SM",
                    ticker_root="SM",
                    month_code="K",
                    multiplier=1.0,
                    year_shift=0,
                    uom_mul=0.022,
                    currency="USD",
                    fx_ticker=None,
                ),
                LegSpec(
                    alias="BO",
                    ticker_root="BO",
                    month_code="K",
                    multiplier=1.0,
                    year_shift=0,
                    uom_mul=0.11,
                    currency="USD",
                    fx_ticker=None,
                ),
            ],
        )

        curves, report = engine.build_with_report(spec, fill=None, multiindex_cols=False)

        self.assertEqual(report.shape[0], 1)
        self.assertEqual(str(report.loc[0, "status"]), "ok")
        expected_crush = (300.0 * 0.022) + (40.0 * 0.11) - (1037.5 * 0.01)
        self.assertAlmostEqual(float(curves.loc[idx[-1], 2026]), expected_crush, places=6)

    def test_bmd_palmoil_minus_cme_sbo_converts_to_usd_per_metric_ton(self) -> None:
        idx = pd.bdate_range("2026-05-04", periods=5)
        df = pd.DataFrame(
            {
                "BOK26 Comdty": [50.0, 50.0, 50.0, 50.0, 50.0],
                "KOK26 Comdty": [4000.0, 4000.0, 4000.0, 4000.0, 4000.0],
                "USDMYR Curncy": [4.0, 4.0, 4.0, 4.0, 4.0],
            },
            index=idx,
        )
        engine = FuturesComboEngine(df)
        spec = StrategySpec(
            name="BMD Palmoil (K) - CME SBO (K)",
            output_currency="USD",
            min_obs=1,
            legs=[
                LegSpec(
                    alias="POK",
                    ticker_root="KO",
                    month_code="K",
                    multiplier=1.0,
                    year_shift=0,
                    uom_mul=1.0,
                    currency="MYR",
                    fx_ticker="USDMYR Curncy",
                ),
                LegSpec(
                    alias="BOK",
                    ticker_root="BO",
                    month_code="K",
                    multiplier=-1.0,
                    year_shift=0,
                    uom_mul=22.04622,
                    currency="USD",
                    fx_ticker=None,
                ),
            ],
        )

        curves, report = engine.build_with_report(spec, fill=None, multiindex_cols=False)

        self.assertEqual(report.shape[0], 1)
        self.assertEqual(str(report.loc[0, "status"]), "ok")
        expected_sbo_usd_per_mt = 50.0 * 22.04622
        expected_palm_usd_per_mt = 4000.0 / 4.0
        expected_spread = expected_palm_usd_per_mt - expected_sbo_usd_per_mt
        self.assertAlmostEqual(float(curves.loc[idx[-1], 2026]), expected_spread, places=6)


if __name__ == "__main__":
    unittest.main()
