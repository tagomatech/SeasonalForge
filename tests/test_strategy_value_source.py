import tempfile
import unittest
from pathlib import Path

import yaml

from app import load_strategies_yaml
from engine import LegSpec, StrategySpec


class StrategyValueSourceTests(unittest.TestCase):
    def test_strategy_spec_default_is_contract(self) -> None:
        spec = StrategySpec(
            name="x",
            legs=[LegSpec(alias="leg", ticker_root="CO", month_code="K")],
        )
        self.assertEqual(spec.value_source, "contract")

    def test_strategy_spec_normalizes_value_source(self) -> None:
        spec = StrategySpec(
            name="x",
            legs=[LegSpec(alias="leg", ticker_root="CO", month_code="K")],
            value_source="Calculated",
        )
        self.assertEqual(spec.value_source, "calculated")

    def test_strategy_spec_rejects_invalid_value_source(self) -> None:
        with self.assertRaises(ValueError):
            StrategySpec(
                name="x",
                legs=[LegSpec(alias="leg", ticker_root="CO", month_code="K")],
                value_source="synthetic",
            )

    def test_yaml_loader_infers_value_source_when_missing(self) -> None:
        payload = {
            "strategies": [
                {
                    "name": "Direct",
                    "category": "Examples",
                    "output_currency": "USD",
                    "min_obs": 1,
                    "legs": [
                        {
                            "alias": "A",
                            "ticker_root": "CO",
                            "month_code": "K",
                            "multiplier": 1,
                            "year_shift": 0,
                            "uom_mul": 1.0,
                            "currency": "USD",
                            "fx_ticker": None,
                        }
                    ],
                },
                {
                    "name": "Calc",
                    "category": "Examples",
                    "output_currency": "USD",
                    "min_obs": 1,
                    "expression": "A / (A + B)",
                    "legs": [
                        {
                            "alias": "A",
                            "ticker_root": "CO",
                            "month_code": "K",
                            "multiplier": 1,
                            "year_shift": 0,
                            "uom_mul": 1.0,
                            "currency": "USD",
                            "fx_ticker": None,
                        },
                        {
                            "alias": "B",
                            "ticker_root": "CL",
                            "month_code": "K",
                            "multiplier": 1,
                            "year_shift": 0,
                            "uom_mul": 1.0,
                            "currency": "USD",
                            "fx_ticker": None,
                        },
                    ],
                },
            ]
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "strategies.yaml"
            path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
            specs, _ = load_strategies_yaml(str(path), reload_token=12345)

        self.assertEqual(specs["Direct"].value_source, "contract")
        self.assertEqual(specs["Calc"].value_source, "calculated")


if __name__ == "__main__":
    unittest.main()
