"""Runs the real, vendored evaluator against the real Agent on a tiny fixture.

Proves the whole loop (ledger -> schedule -> BM25 -> evaluator scoring) works
through the actual grading code path, without needing the real 50k-row
catalog (see data/README.md for why that file isn't available here).
"""
from __future__ import annotations

import unittest
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate
from starter.agent import Agent

FIXTURES = Path(__file__).parent / "fixtures"

SAMPLES = [
    {
        "sample_id": "fixture_0001",
        "scenario_type": "buying",
        "ground_truth": {"parent_asin": "T0001"},
        "user_profile": {
            "purchase_frequency": "3-4 prior purchases",
            "average_prior_rating": 4.5,
            "rating_style": "usually positive",
            "preference_tags": ["durability"],
            "summary": "fixture session",
        },
    },
]


class TestEvaluatorSmoke(unittest.TestCase):
    def test_agent_runs_end_to_end_through_the_real_evaluator(self) -> None:
        catalog_ids, categories, products = catalog_index(FIXTURES / "catalog.jsonl")
        agent = Agent(FIXTURES / "catalog.jsonl")
        result = evaluate(agent, SAMPLES, catalog_ids, categories, products)
        self.assertEqual(result["sample_count"], 1)
        self.assertTrue(result["sessions"][0]["hit"], "expected the target to surface within 10 turns")


if __name__ == "__main__":
    unittest.main()
