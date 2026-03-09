import unittest

import numpy as np
import pandas as pd

from engine import FuturesComboEngine, LegSpec, StrategySpec


class LifecycleAlignmentTests(unittest.TestCase):
    @staticmethod
    def _engine_and_curves() -> tuple[FuturesComboEngine, pd.DataFrame]:
        idx = pd.bdate_range("2024-01-02", "2026-03-31")
        n = len(idx)
        base = np.linspace(70.0, 90.0, n)
        delayed = np.full(n, np.nan, dtype=float)
        delayed[120:] = np.linspace(65.0, 85.0, n - 120)

        df = pd.DataFrame(
            {
                "COK26 Comdty": base,
                "COK25 Comdty": delayed,
            },
            index=idx,
        )
        engine = FuturesComboEngine(df)
        spec = StrategySpec(
            name="Brent(K)-toy",
            legs=[LegSpec(alias="leg", ticker_root="CO", month_code="K")],
        )
        curves, _ = engine.build_with_report(spec, fill=None, multiindex_cols=False)
        return engine, curves

    def test_asof_aligned_recovers_shorter_history_in_window(self) -> None:
        engine, curves = self._engine_and_curves()

        matrix_abs, _, meta_abs = engine.overlay_lifecycle_window(
            curves,
            window_months=(-3, 0),
            asof=pd.Timestamp("2026-03-03"),
            reference=2026,
            alignment="first_valid",
        )
        self.assertEqual(meta_abs["alignment"], "first_valid")
        self.assertEqual(int(matrix_abs[2025].notna().sum()), 0)

        matrix_asof, _, meta_asof = engine.overlay_lifecycle_window(
            curves,
            window_months=(-3, 0),
            asof=pd.Timestamp("2026-03-03"),
            reference=2026,
            alignment="asof_aligned",
        )
        self.assertEqual(meta_asof["alignment"], "asof_aligned")
        self.assertGreater(int(matrix_asof[2025].notna().sum()), 0)

        t_asof = int(meta_asof["t_asof"])
        self.assertTrue(np.isfinite(matrix_asof.loc[t_asof, 2025]))
        self.assertTrue(np.isfinite(matrix_asof.loc[t_asof, 2026]))

    def test_asof_aligned_uses_curve_own_asof_date_for_forward_window(self) -> None:
        idx = pd.bdate_range("2024-09-02", "2026-03-31")
        n = len(idx)

        y2025 = np.full(n, np.nan, dtype=float)
        y2026 = np.full(n, np.nan, dtype=float)

        mask_2025 = (idx >= pd.Timestamp("2024-10-15")) & (idx <= pd.Timestamp("2025-05-30"))
        mask_2026 = (idx >= pd.Timestamp("2025-06-02")) & (idx <= pd.Timestamp("2026-03-03"))

        y2025[mask_2025] = np.linspace(40.0, 55.0, int(mask_2025.sum()))
        y2026[mask_2026] = np.linspace(45.0, 52.0, int(mask_2026.sum()))

        df = pd.DataFrame(
            {
                "COK25 Comdty": y2025,
                "COK26 Comdty": y2026,
            },
            index=idx,
        )
        engine = FuturesComboEngine(df)
        spec = StrategySpec(
            name="Brent(K)-forward-window",
            legs=[LegSpec(alias="leg", ticker_root="CO", month_code="K")],
        )
        curves, _ = engine.build_with_report(spec, fill=None, multiindex_cols=False)

        matrix, _, meta = engine.overlay_lifecycle_window(
            curves,
            window_months=(-1, 3),
            asof=pd.Timestamp("2026-03-03"),
            reference=2026,
            alignment="asof_aligned",
        )

        t_asof = int(meta["t_asof"])
        future_rows = [int(t) for t in matrix.index if int(t) > t_asof]
        self.assertGreater(len(future_rows), 0)
        self.assertGreater(int(matrix.loc[future_rows, 2025].notna().sum()), 0)
        self.assertEqual(int(matrix.loc[future_rows, 2026].notna().sum()), 0)

        expected_2025_asof = float(df.loc[pd.Timestamp("2025-03-03"), "COK25 Comdty"])
        self.assertAlmostEqual(float(matrix.loc[t_asof, 2025]), expected_2025_asof)

    def test_invalid_alignment_raises(self) -> None:
        engine, curves = self._engine_and_curves()
        with self.assertRaises(ValueError):
            engine.overlay_lifecycle_window(
                curves,
                window_months=(-3, 0),
                asof=pd.Timestamp("2026-03-03"),
                reference=2026,
                alignment="bad_mode",
            )


if __name__ == "__main__":
    unittest.main()


