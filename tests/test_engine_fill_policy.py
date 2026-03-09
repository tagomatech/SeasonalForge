import unittest

import numpy as np
import pandas as pd

from engine import FuturesComboEngine


class FillPolicyTests(unittest.TestCase):
    def test_trailing_tail_is_not_forward_filled(self) -> None:
        idx = pd.to_datetime(["2026-02-27", "2026-03-02", "2026-03-03"])
        s = pd.Series([100.0, np.nan, np.nan], index=idx)
        out = FuturesComboEngine._fill_business_days(s, limit=1)
        self.assertTrue(np.isnan(out.loc[pd.Timestamp("2026-03-02")]))
        self.assertTrue(np.isnan(out.loc[pd.Timestamp("2026-03-03")]))

    def test_interior_gap_is_filled_within_limit(self) -> None:
        idx = pd.to_datetime(["2026-02-27", "2026-03-02", "2026-03-03"])
        s = pd.Series([100.0, np.nan, 101.0], index=idx)
        out = FuturesComboEngine._fill_business_days(s, limit=1)
        self.assertEqual(float(out.loc[pd.Timestamp("2026-03-02")]), 100.0)
        self.assertEqual(float(out.loc[pd.Timestamp("2026-03-03")]), 101.0)

    def test_interior_gap_respects_limit(self) -> None:
        idx = pd.to_datetime(["2026-02-27", "2026-03-02", "2026-03-03", "2026-03-04"])
        s = pd.Series([100.0, np.nan, np.nan, 103.0], index=idx)
        out = FuturesComboEngine._fill_business_days(s, limit=1)
        self.assertEqual(float(out.loc[pd.Timestamp("2026-03-02")]), 100.0)
        self.assertTrue(np.isnan(out.loc[pd.Timestamp("2026-03-03")]))
        self.assertEqual(float(out.loc[pd.Timestamp("2026-03-04")]), 103.0)


if __name__ == "__main__":
    unittest.main()
