"""Differential tests: exact solver vs. brute-force reference."""

from __future__ import annotations

import json
import random
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from anonymizer import anonymize  # noqa: E402
from generator import make_job  # noqa: E402
from reference import solve_reference  # noqa: E402


class DifferentialTests(unittest.TestCase):
    def assert_matches_reference(self, job: dict) -> None:
        expected = solve_reference(job)
        result = anonymize(json.dumps(job))
        self.assertEqual(result["objectives"]["suppressed_records"], expected.suppressed)
        self.assertEqual(result["objectives"]["retained_information_loss"],
                         expected.retained_loss)
        self.assertEqual(
            tuple(tuple(row["nodes"]) for row in result["generalization"]),
            expected.cuts,
        )
        self.assertEqual(
            tuple(result["published"]),
            expected.published,
        )
        self.assertEqual(tuple(result["suppressed_ids"]), expected.suppressed_ids)

    def test_random_jobs(self) -> None:
        rng = random.Random(2025)
        for case in range(160):
            job = make_job(rng)
            with self.subTest(case=case, job=job):
                self.assert_matches_reference(job)

    def test_small_hierarchies_many_records(self) -> None:
        # Deeper trees and more records; small enough for exhaustive reference.
        rng = random.Random(90210)
        for case in range(40):
            job = make_job(rng, max_nodes=9, max_records=9)
            with self.subTest(case=case):
                self.assert_matches_reference(job)


if __name__ == "__main__":
    unittest.main()
