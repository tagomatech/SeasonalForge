import unittest

from app import filter_strategy_names, strategy_structure_type
from engine import LegSpec, StrategySpec


class StrategyBrowserFilterTests(unittest.TestCase):
    @staticmethod
    def _sample_specs() -> tuple[dict[str, StrategySpec], dict[str, str]]:
        specs = {
            "BO Flat (K)": StrategySpec(
                name="BO Flat (K)",
                legs=[LegSpec(alias="BOK", ticker_root="BO", month_code="K")],
                output_currency="USD",
            ),
            "BO (M) - BO (Z)": StrategySpec(
                name="BO (M) - BO (Z)",
                legs=[
                    LegSpec(alias="BOM", ticker_root="BO", month_code="M", multiplier=1),
                    LegSpec(alias="BOZ", ticker_root="BO", month_code="Z", multiplier=-1),
                ],
                output_currency="USD",
            ),
            "3-leg basket": StrategySpec(
                name="3-leg basket",
                legs=[
                    LegSpec(alias="BOK", ticker_root="BO", month_code="K", multiplier=1),
                    LegSpec(alias="SMK", ticker_root="SM", month_code="K", multiplier=-1),
                    LegSpec(alias="COK", ticker_root="CO", month_code="K", multiplier=1),
                ],
                output_currency="USD",
            ),
            "Oil share calc": StrategySpec(
                name="Oil share calc",
                legs=[
                    LegSpec(alias="BO", ticker_root="BO", month_code="N", multiplier=1),
                    LegSpec(alias="SM", ticker_root="SM", month_code="N", multiplier=1),
                ],
                expression="100 * BO / (BO + SM)",
                value_source="calculated",
                output_currency="USD",
            ),
            "Brent (Q)": StrategySpec(
                name="Brent (Q)",
                legs=[LegSpec(alias="COQ", ticker_root="CO", month_code="Q")],
                output_currency="USD",
            ),
        }
        categories = {
            "BO Flat (K)": "Oilseeds",
            "BO (M) - BO (Z)": "Oilseeds",
            "3-leg basket": "Biofuels",
            "Oil share calc": "Oilseeds",
            "Brent (Q)": "Energy",
        }
        return specs, categories

    def test_structure_type_classification(self) -> None:
        specs, _ = self._sample_specs()
        self.assertEqual(strategy_structure_type(specs["BO Flat (K)"]), "Flat price")
        self.assertEqual(strategy_structure_type(specs["BO (M) - BO (Z)"]), "Spread")
        self.assertEqual(strategy_structure_type(specs["3-leg basket"]), "Multi-leg")
        self.assertEqual(strategy_structure_type(specs["Oil share calc"]), "Calculated")

    def test_filter_by_structure(self) -> None:
        specs, categories = self._sample_specs()
        out = filter_strategy_names(
            specs,
            categories,
            category_filter="All",
            search_query="",
            structure_filters={"Spread"},
            commodity_filters=set(),
            month_filters=set(),
            commodity_match_all=False,
        )
        self.assertEqual(out, ["BO (M) - BO (Z)"])

    def test_filter_by_commodity_any(self) -> None:
        specs, categories = self._sample_specs()
        out = filter_strategy_names(
            specs,
            categories,
            category_filter="All",
            search_query="",
            structure_filters=set(),
            commodity_filters={"CO"},
            month_filters=set(),
            commodity_match_all=False,
        )
        self.assertEqual(out, ["3-leg basket", "Brent (Q)"])

    def test_filter_by_commodity_all(self) -> None:
        specs, categories = self._sample_specs()
        out = filter_strategy_names(
            specs,
            categories,
            category_filter="All",
            search_query="",
            structure_filters=set(),
            commodity_filters={"BO", "SM"},
            month_filters=set(),
            commodity_match_all=True,
        )
        self.assertEqual(out, ["3-leg basket", "Oil share calc"])

    def test_filter_by_month_code(self) -> None:
        specs, categories = self._sample_specs()
        out = filter_strategy_names(
            specs,
            categories,
            category_filter="All",
            search_query="",
            structure_filters=set(),
            commodity_filters=set(),
            month_filters={"Q"},
            commodity_match_all=False,
        )
        self.assertEqual(out, ["Brent (Q)"])


if __name__ == "__main__":
    unittest.main()

