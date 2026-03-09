import unittest

import pandas as pd

from data_provider import (
    BloombergPullConfig,
    _cap_history_to_end,
    _next_business_day_yyyymmdd,
    _previous_business_day_yyyymmdd,
    _resolve_incremental_start_yyyymmdd,
    _resolve_hist_end_yyyymmdd,
)


class DateLogicTests(unittest.TestCase):
    def test_previous_business_day_moves_back_over_weekend(self) -> None:
        self.assertEqual(_previous_business_day_yyyymmdd(now=pd.Timestamp("2026-03-03")), "20260302")
        self.assertEqual(_previous_business_day_yyyymmdd(now=pd.Timestamp("2026-03-02")), "20260227")

    def test_next_business_day_skips_weekend(self) -> None:
        self.assertEqual(_next_business_day_yyyymmdd(pd.Timestamp("2026-02-27")), "20260302")
        self.assertEqual(_next_business_day_yyyymmdd(pd.Timestamp("2026-03-02")), "20260303")

    def test_hist_end_without_snapshot_defaults_to_previous_business_day(self) -> None:
        cfg = BloombergPullConfig(end_mode="yesterday", include_today_snapshot=False)
        self.assertEqual(_resolve_hist_end_yyyymmdd(cfg, now=pd.Timestamp("2026-03-03")), "20260302")

    def test_hist_end_today_mode_without_snapshot_uses_today(self) -> None:
        cfg = BloombergPullConfig(end_mode="today", include_today_snapshot=False)
        self.assertEqual(_resolve_hist_end_yyyymmdd(cfg, now=pd.Timestamp("2026-03-03")), "20260303")

    def test_hist_end_today_mode_with_snapshot_caps_to_previous_business_day(self) -> None:
        cfg = BloombergPullConfig(end_mode="today", include_today_snapshot=True)
        self.assertEqual(_resolve_hist_end_yyyymmdd(cfg, now=pd.Timestamp("2026-03-03")), "20260302")

    def test_incremental_start_rewinds_for_backfill(self) -> None:
        out = _resolve_incremental_start_yyyymmdd(
            last_dt=pd.Timestamp("2026-03-03"),
            hist_end_yyyymmdd="20260303",
            start_date_yyyymmdd="20100101",
            backfill_business_days=2,
        )
        self.assertEqual(out, "20260227")

    def test_incremental_start_catches_up_when_cache_is_behind(self) -> None:
        out = _resolve_incremental_start_yyyymmdd(
            last_dt=pd.Timestamp("2026-02-20"),
            hist_end_yyyymmdd="20260303",
            start_date_yyyymmdd="20100101",
            backfill_business_days=2,
        )
        self.assertEqual(out, "20260223")

    def test_incremental_start_respects_history_start_floor(self) -> None:
        out = _resolve_incremental_start_yyyymmdd(
            last_dt=None,
            hist_end_yyyymmdd="20260303",
            start_date_yyyymmdd="20260302",
            backfill_business_days=5,
        )
        self.assertEqual(out, "20260302")

    def test_incremental_start_none_when_start_after_end(self) -> None:
        out = _resolve_incremental_start_yyyymmdd(
            last_dt=None,
            hist_end_yyyymmdd="20260303",
            start_date_yyyymmdd="20260304",
            backfill_business_days=5,
        )
        self.assertIsNone(out)

    def test_cap_history_to_end_trims_rows_after_hist_end(self) -> None:
        idx = pd.to_datetime(["2026-03-02", "2026-03-03", "2026-03-04"])
        df = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=idx)
        capped, was_trimmed = _cap_history_to_end(df, "20260303")
        assert capped is not None
        self.assertTrue(was_trimmed)
        self.assertEqual(list(capped.index), [pd.Timestamp("2026-03-02"), pd.Timestamp("2026-03-03")])

    def test_cap_history_to_end_noop_when_in_range(self) -> None:
        idx = pd.to_datetime(["2026-03-02", "2026-03-03"])
        df = pd.DataFrame({"A": [1.0, 2.0]}, index=idx)
        capped, was_trimmed = _cap_history_to_end(df, "20260303")
        assert capped is not None
        self.assertFalse(was_trimmed)
        self.assertEqual(list(capped.index), list(df.index))


if __name__ == "__main__":
    unittest.main()
