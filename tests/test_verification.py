"""Independent verification must catch corrupted results."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anonymizer import run_bytes  # noqa: E402
from anonymizer.output import verification_errors  # noqa: E402

DOCUMENT = {
    "records": [
        {"id": "a", "quasi": {"x": "l1"}, "sensitive": "s1"},
        {"id": "b", "quasi": {"x": "l1"}, "sensitive": "s2"},
        {"id": "c", "quasi": {"x": "l2"}, "sensitive": "s3"},
        {"id": "d", "quasi": {"x": "l2"}, "sensitive": "s4"},
    ],
    "hierarchies": [
        {"attribute": "x", "root": "R",
         "nodes": [{"id": "R", "parent": None},
                   {"id": "l1", "parent": "R"},
                   {"id": "l2", "parent": "R"}]},
    ],
    "k": 2,
    "l": 2,
}


class VerificationMutationTests(unittest.TestCase):
    def setUp(self):
        dataset, solution, document = run_bytes(json.dumps(DOCUMENT).encode())
        self.dataset = dataset
        self.solution = solution
        self.document = document
        self.assertEqual(verification_errors(dataset, document, solution), [])

    def mutate(self, function):
        document = copy.deepcopy(self.document)
        function(document)
        return verification_errors(self.dataset, document, self.solution)

    def test_detect_k_violation(self):
        def corrupt(document):
            document["published"] = document["published"][:1]
            document["suppressed_ids"] = ["b", "c", "d"]
        problems = self.mutate(corrupt)
        self.assertTrue(any("fewer than k" in problem for problem in problems), problems)

    def test_detect_l_violation(self):
        def corrupt(document):
            document["published"][0]["sensitive"] = "s2"
            document["published"][1]["sensitive"] = "s2"
        problems = self.mutate(corrupt)
        self.assertTrue(any("fewer than l" in problem for problem in problems), problems)

    def test_detect_wrong_ancestor(self):
        def corrupt(document):
            document["published"][0]["quasi"]["x"] = "l2"
            document["audit"]["mapping"][0]["generalized"]["x"] = "l2"
        problems = self.mutate(corrupt)
        self.assertTrue(any("not an ancestor" in problem for problem in problems), problems)

    def test_detect_missing_record(self):
        def corrupt(document):
            document["published"] = document["published"][1:]
            document["audit"]["mapping"] = document["audit"]["mapping"][1:]
        problems = self.mutate(corrupt)
        self.assertTrue(problems)

    def test_detect_duplicate_published_id(self):
        def corrupt(document):
            document["published"][1]["id"] = document["published"][0]["id"]
        problems = self.mutate(corrupt)
        self.assertTrue(problems)

    def test_detect_invalid_cut(self):
        def corrupt(document):
            document["generalization"][0]["nodes"] = ["l1"]
        problems = self.mutate(corrupt)
        self.assertTrue(any("not a valid global cut" in problem for problem in problems), problems)

    def test_detect_unsorted_cut(self):
        def corrupt(document):
            document["generalization"][0]["nodes"] = ["R", "l1", "l2"]
            document["generalization"][0]["nodes"].sort(reverse=True)
        problems = self.mutate(corrupt)
        self.assertTrue(any("not ASCII-sorted" in problem for problem in problems), problems)

    def test_detect_status_mismatch(self):
        def corrupt(document):
            document["audit"]["mapping"][0]["status"] = "suppressed"
        problems = self.mutate(corrupt)
        self.assertTrue(any("status mismatch" in problem for problem in problems), problems)


if __name__ == "__main__":
    unittest.main()
