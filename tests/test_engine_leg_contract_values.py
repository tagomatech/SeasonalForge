import unittest

import numpy as np
import pandas as pd

from engine import FillPolicy, FuturesComboEngine, LegSpec, StrategySpec


class LegContractValuesTests(unittest.TestCase):
    def test_build_leg_contract_values_labels_and_values(self) -> None:
        idx = pd.date_range("2026-03-02", periods=5, freq="B")
        df = pd.DataFrame(
            {
                "CLH26 Comdty": [70.0, 71.0, np.nan, np.nan, 74.0],
                "CLM26 Comdty": [72.0, 73.0, 74.0, 75.0, 76.0],
            },
            index=idx,
        )
        engine = FuturesComboEngine(df)
        spec = StrategySpec(
            name="x",
            legs=[
                LegSpec(alias="leg_a", ticker_root="CL", month_code="H", year_shift=0),
                LegSpec(alias="leg_b", ticker_root="CL", month_code="M", year_shift=0),
            ],
        )

        out = engine.build_leg_contract_values(
            spec,
            anchor_year=2026,
            fill=FillPolicy(method="ffill", limit=1, apply_to_fx=True),
        )

        self.assertEqual(out.shape, (5, 2))
        self.assertIn("leg_a (CLH2026) [CLH26 Comdty]", out.columns)
        self.assertIn("leg_b (CLM2026) [CLM26 Comdty]", out.columns)
        self.assertEqual(float(out.loc[idx[2], "leg_a (CLH2026) [CLH26 Comdty]"]), 71.0)
        self.assertTrue(np.isnan(out.loc[idx[3], "leg_a (CLH2026) [CLH26 Comdty]"]))


if __name__ == "__main__":
    unittest.main()
