import unittest

import numpy as np
import pandas as pd

from ui_components import build_lifecycle_heatmap_table, compact_heatmap_table


class HeatmapOrderingTests(unittest.TestCase):
    def test_reference_year_is_sorted_first_then_past_then_future(self) -> None:
        table = pd.DataFrame(
            {
                "M-1": [1.0, 2.0, 3.0, 4.0],
                "M+0": [5.0, 6.0, 7.0, 8.0],
            },
            index=[2028, 2027, 2026, 2025],
        )

        out, _, _ = compact_heatmap_table(
            table,
            drop_empty_rows=False,
            drop_empty_cols=False,
            reference_anchor_year=2026,
        )
        self.assertEqual(list(int(i) for i in out.index), [2026, 2025, 2027, 2028])

    def test_default_order_stays_descending_without_reference(self) -> None:
        table = pd.DataFrame(
            {
                "M-1": [1.0, 2.0, 3.0, 4.0],
                "M+0": [5.0, 6.0, 7.0, 8.0],
            },
            index=[2028, 2027, 2026, 2025],
        )

        out, _, _ = compact_heatmap_table(
            table,
            drop_empty_rows=False,
            drop_empty_cols=False,
        )
        self.assertEqual(list(int(i) for i in out.index), [2028, 2027, 2026, 2025])


    def test_month_heatmap_carries_first_forward_bucket_from_last_post_asof_value(self) -> None:
        dates = pd.bdate_range("2026-03-02", "2026-04-15")
        t_index = pd.Index(range(len(dates)), name="t")
        calendar = pd.DataFrame(
            {
                "t": t_index,
                "date": dates,
            }
        )

        y2025 = np.full(len(dates), np.nan, dtype=float)
        y2015 = np.linspace(80.0, 100.0, len(dates))
        y2026 = np.full(len(dates), np.nan, dtype=float)

        mask_2025 = dates <= pd.Timestamp("2026-03-31")
        mask_2026 = dates <= pd.Timestamp("2026-03-06")
        y2025[mask_2025] = np.linspace(50.0, 70.0, int(mask_2025.sum()))
        y2026[mask_2026] = np.linspace(60.0, 62.0, int(mask_2026.sum()))

        matrix = pd.DataFrame(
            {
                2015: y2015,
                2025: y2025,
                2026: y2026,
            },
            index=t_index,
        )

        table = build_lifecycle_heatmap_table(
            matrix,
            calendar,
            years=[2015, 2025, 2026],
            asof=pd.Timestamp("2026-03-06"),
            bucket_mode="month",
            agg_mode="mean",
        )

        expected_2025_m1 = float(y2025[np.where(mask_2025)[0][-1]])
        expected_2015_m1 = float(np.mean(y2015[dates.month == 4]))
        self.assertAlmostEqual(float(table.loc[2025, "M+1"]), expected_2025_m1)
        self.assertAlmostEqual(float(table.loc[2015, "M+1"]), expected_2015_m1)
        self.assertTrue(pd.isna(table.loc[2026, "M+1"]))

if __name__ == "__main__":
    unittest.main()
