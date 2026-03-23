import unittest

import pandas as pd

from src.decision_engine import apply_decision_engine, compute_expected_retention_value, recommend_action


class DecisionEngineTests(unittest.TestCase):
    def test_priority_outreach_rule(self):
        row = pd.Series(
            {
                "monthly_gpv": 90000.0,
                "chargeback_rate": 0.010,
                "inactivity_days": 18,
                "product_adoption_count": 4,
                "churn_probability": 0.82,
            }
        )
        self.assertEqual(recommend_action(row), "priority_outreach")

    def test_offer_incentive_rule(self):
        row = pd.Series(
            {
                "monthly_gpv": 28000.0,
                "chargeback_rate": 0.008,
                "inactivity_days": 15,
                "product_adoption_count": 3,
                "churn_probability": 0.67,
            }
        )
        self.assertEqual(recommend_action(row), "offer_incentive")

    def test_product_education_rule(self):
        row = pd.Series(
            {
                "monthly_gpv": 7000.0,
                "chargeback_rate": 0.005,
                "inactivity_days": 20,
                "product_adoption_count": 1,
                "churn_probability": 0.48,
            }
        )
        self.assertEqual(recommend_action(row), "product_education")

    def test_monitor_only_rule(self):
        row = pd.Series(
            {
                "monthly_gpv": 5000.0,
                "chargeback_rate": 0.004,
                "inactivity_days": 8,
                "product_adoption_count": 5,
                "churn_probability": 0.19,
            }
        )
        self.assertEqual(recommend_action(row), "monitor_only")

    def test_expected_retention_value_accounts_for_incentive_cost(self):
        row = pd.Series(
            {
                "monthly_gpv": 40000.0,
                "churn_probability": 0.70,
                "recommended_action": "offer_incentive",
            }
        )
        expected_value = compute_expected_retention_value(row)
        self.assertGreater(expected_value, 0.0)
        self.assertLess(expected_value, 40000.0 * 0.012 * 6 * 0.70 * 0.28)

    def test_priority_ranks_are_unique_and_sorted(self):
        df = pd.DataFrame(
            [
                {
                    "merchant_id": "M001",
                    "segment": "mid_market",
                    "monthly_gpv": 85000.0,
                    "chargeback_rate": 0.012,
                    "inactivity_days": 25,
                    "product_adoption_count": 4,
                    "churn_probability": 0.81,
                },
                {
                    "merchant_id": "M002",
                    "segment": "smb",
                    "monthly_gpv": 26000.0,
                    "chargeback_rate": 0.007,
                    "inactivity_days": 14,
                    "product_adoption_count": 3,
                    "churn_probability": 0.66,
                },
                {
                    "merchant_id": "M003",
                    "segment": "micro",
                    "monthly_gpv": 6000.0,
                    "chargeback_rate": 0.004,
                    "inactivity_days": 18,
                    "product_adoption_count": 1,
                    "churn_probability": 0.45,
                },
            ]
        )

        ranked = apply_decision_engine(df)
        self.assertEqual(ranked["priority_rank"].tolist(), [1, 2, 3])
        self.assertEqual(ranked["priority_rank"].nunique(), len(ranked))
        self.assertTrue(
            ranked["expected_retention_value"].tolist()
            == sorted(ranked["expected_retention_value"].tolist(), reverse=True)
        )


if __name__ == "__main__":
    unittest.main()
