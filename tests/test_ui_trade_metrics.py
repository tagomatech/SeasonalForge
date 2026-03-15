import unittest

import pandas as pd

from ui_components import summarize_forward_trade_metrics


class TradeMetricTests(unittest.TestCase):
    def test_forward_metrics_capture_long_bias_and_stop_statistics(self) -> None:
        calendar = pd.DataFrame(
            {
                "t": [0, 1, 2, 3],
                "date": pd.bdate_range("2026-03-02", periods=4),
                "label": ["d0", "d1", "d2", "d3"],
                "is_asof": [False, True, False, False],
            }
        )
        matrix = pd.DataFrame(
            {
                2026: [100.0, 101.0, 103.0, 104.0],
                2025: [90.0, 100.0, 98.0, 106.0],
                2024: [95.0, 100.0, 101.0, 104.0],
                2023: [97.0, 100.0, 99.0, 102.0],
            },
            index=pd.Index([0, 1, 2, 3], name="t"),
        )

        out = summarize_forward_trade_metrics(
            matrix,
            calendar,
            reference_anchor_year=2026,
        )

        self.assertTrue(out["ok"])
        self.assertEqual(out["bias"], "Long")
        self.assertEqual(out["sample_size"], 3)
        self.assertAlmostEqual(out["up_rate"], 100.0)
        self.assertAlmostEqual(out["success_rate"], 100.0)
        self.assertAlmostEqual(out["avg_change"], 4.0)
        self.assertAlmostEqual(out["median_change"], 4.0)
        self.assertAlmostEqual(out["mae_p75"], 1.5)
        self.assertAlmostEqual(out["mae_p90"], 1.8)
        self.assertAlmostEqual(out["reward_risk"], 4.0)

    def test_forward_metrics_require_forward_window(self) -> None:
        calendar = pd.DataFrame(
            {
                "t": [0, 1],
                "date": pd.bdate_range("2026-03-02", periods=2),
                "label": ["d0", "d1"],
                "is_asof": [False, True],
            }
        )
        matrix = pd.DataFrame({2026: [100.0, 101.0], 2025: [100.0, 100.0]}, index=[0, 1])

        out = summarize_forward_trade_metrics(
            matrix,
            calendar,
            reference_anchor_year=2026,
        )

        self.assertFalse(out["ok"])
        self.assertEqual(out["reason"], "No forward window after ASOF")


if __name__ == "__main__":
    unittest.main()
