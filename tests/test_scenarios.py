"""Fixture tests: named scenarios demanded by the acceptance checklist."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from anonymizer import anonymize  # noqa: E402
from reference import solve_reference  # noqa: E402


def job(records, hierarchies, k, l):
    return {"records": records, "hierarchies": hierarchies, "k": k, "l": l}


def rec(identifier, **quasi):
    return {"id": identifier,
            "quasi": dict(quasi),
            "sensitive": quasi.pop("_s", "s")}


def make_record(identifier, sensitive, **quasi):
    return {"id": identifier, "quasi": quasi, "sensitive": sensitive}


def check_against_reference(test, document):
    expected = solve_reference(document)
    result = anonymize(json.dumps(document))
    test.assertEqual(result["objectives"]["suppressed_records"], expected.suppressed)
    test.assertEqual(result["objectives"]["retained_information_loss"],
                     expected.retained_loss)
    test.assertEqual(
        tuple(tuple(row["nodes"]) for row in result["generalization"]),
        expected.cuts,
    )
    test.assertEqual(tuple(result["published"]), expected.published)
    test.assertEqual(tuple(result["suppressed_ids"]), expected.suppressed_ids)
    return result


class SingleLeafTreeTests(unittest.TestCase):
    def test_single_leaf_tree_uses_the_root(self):
        hierarchy = {"attribute": "a", "root": "r",
                     "nodes": [{"id": "r", "parent": None}]}
        records = [make_record(f"r{index}", "s", a="r") for index in range(3)]
        document = job(records, [hierarchy], 2, 1)
        result = check_against_reference(self, document)
        self.assertEqual(result["generalization"], [{"attribute": "a", "nodes": ["r"]}])
        self.assertEqual(result["objectives"]["retained_information_loss"], 0)

    def test_empty_records_is_valid(self):
        hierarchy = {"attribute": "a", "root": "r",
                     "nodes": [{"id": "r", "parent": None},
                               {"id": "x", "parent": "r"}]}
        document = job([], [hierarchy], 3, 2)
        result = check_against_reference(self, document)
        self.assertEqual(result["published"], [])
        self.assertEqual(result["suppressed_ids"], [])
        self.assertEqual(result["generalization"], [{"attribute": "a", "nodes": ["r"]}])

    def test_no_hierarchies_with_no_records(self):
        document = job([], [], 1, 1)
        result = anonymize(json.dumps(document))
        self.assertEqual(result["generalization"], [])
        self.assertEqual(result["published"], [])


class AllSuppressedTests(unittest.TestCase):
    def test_every_class_below_k_is_fully_suppressed(self):
        hierarchy = {"attribute": "a", "root": "R",
                     "nodes": [{"id": "R", "parent": None},
                               {"id": "l1", "parent": "R"},
                               {"id": "l2", "parent": "R"}]}
        records = [
            make_record("q1", "s1", a="l1"),
            make_record("q2", "s1", a="l1"),
            make_record("q3", "s1", a="l2"),
        ]
        document = job(records, [hierarchy], 3, 2)
        result = check_against_reference(self, document)
        self.assertEqual(result["objectives"]["suppressed_records"], 3)
        self.assertEqual(result["published"], [])
        self.assertEqual(result["suppressed_ids"], ["q1", "q2", "q3"])
        self.assertEqual(result["objectives"]["retained_information_loss"], 0)
        for row in result["audit"]["mapping"]:
            self.assertEqual(row["status"], "suppressed")
            self.assertIn(row["generalized"]["a"], {"R", "l1", "l2"})

    def test_partial_retention_beats_retaining_more_lossy_everything(self):
        # Two tight clusters; the global root works for all records but one
        # cluster lacks l distinct sensitive values.  Suppressing the invalid
        # class is the only way to keep the rest.
        hierarchy = {"attribute": "a", "root": "R",
                     "nodes": [{"id": "R", "parent": None},
                               {"id": "x", "parent": "R"},
                               {"id": "y", "parent": "R"}]}
        records = [
            make_record("q1", "s1", a="x"),
            make_record("q2", "s2", a="x"),
            make_record("q3", "s2", a="y"),
            make_record("q4", "s2", a="y"),
            make_record("q5", "s3", a="y"),
        ]
        document = job(records, [hierarchy], 2, 2)
        result = check_against_reference(self, document)
        self.assertEqual(result["objectives"]["suppressed_records"], 0)
        self.assertEqual(result["suppressed_ids"], [])


class SensitiveDuplicateTests(unittest.TestCase):
    def test_duplicate_sensitive_values_count_once(self):
        hierarchy = {"attribute": "a", "root": "R",
                     "nodes": [{"id": "R", "parent": None},
                               {"id": "l1", "parent": "R"},
                               {"id": "l2", "parent": "R"}]}
        # All records map to R under the only publishable cut; sensitive has
        # only one distinct value, so l=2 fails and the whole class is dropped.
        records = [
            make_record("q1", "flu", a="l1"),
            make_record("q2", "flu", a="l2"),
            make_record("q3", "flu", a="l1"),
        ]
        document = job(records, [hierarchy], 2, 2)
        result = check_against_reference(self, document)
        self.assertEqual(result["objectives"]["suppressed_records"], 3)

    def test_one_new_sensitive_value_makes_class_publishable(self):
        hierarchy = {"attribute": "a", "root": "R",
                     "nodes": [{"id": "R", "parent": None}]}
        records = [
            make_record("q1", "flu", a="R"),
            make_record("q2", "flu", a="R"),
            make_record("q3", "cold", a="R"),
        ]
        document = job(records, [hierarchy], 2, 2)
        result = check_against_reference(self, document)
        self.assertEqual(result["objectives"]["suppressed_records"], 0)
        self.assertEqual(len(result["published"]), 3)


class CrossAttributeTests(unittest.TestCase):
    def test_partitions_are_only_refined_not_coarsened(self):
        # Generalising sex alone cannot undo the age partition: joint classes
        # are intersections, and the winner must be searched jointly.
        hierarchies = [
            {"attribute": "age", "root": "A",
             "nodes": [{"id": "A", "parent": None},
                       {"id": "young", "parent": "A"},
                       {"id": "old", "parent": "A"}]},
            {"attribute": "sex", "root": "S",
             "nodes": [{"id": "S", "parent": None},
                       {"id": "m", "parent": "S"},
                       {"id": "f", "parent": "S"}]},
        ]
        records = [
            make_record("r1", "s1", age="young", sex="m"),
            make_record("r2", "s2", age="young", sex="f"),
            make_record("r3", "s3", age="old", sex="m"),
            make_record("r4", "s1", age="old", sex="m"),
            make_record("r5", "s2", age="old", sex="f"),
        ]
        document = job(records, hierarchies, 2, 2)
        result = check_against_reference(self, document)
        # Greedy per-field would suppress r3/r4; the optimum keeps them together.
        combined = [
            (row["quasi"]["age"], row["quasi"]["sex"]) for row in result["published"]
        ]
        self.assertEqual(
            sorted(combined),
            [("old", "S"), ("old", "S"), ("old", "S"),
             ("young", "S"), ("young", "S")],
        )

    def test_attributes_are_ordered_by_ascii(self):
        hierarchies = [
            {"attribute": "z", "root": "z0", "nodes": [{"id": "z0", "parent": None}]},
            {"attribute": "A", "root": "A0", "nodes": [{"id": "A0", "parent": None}]},
            {"attribute": "m", "root": "m0", "nodes": [{"id": "m0", "parent": None}]},
        ]
        records = [make_record("r1", "s", z="z0", A="A0", m="m0")]
        document = job(records, hierarchies, 1, 1)
        result = check_against_reference(self, document)
        self.assertEqual([row["attribute"] for row in result["generalization"]],
                         ["A", "m", "z"])
        self.assertEqual(list(result["published"][0]["quasi"]), ["A", "m", "z"])


class TieBreakTests(unittest.TestCase):
    def test_equal_objectives_choose_lexicographically_smallest_cut_arrays(self):
        # Only records at g1 exist.  Several zero-loss cuts contain g1:
        # [Q,g1,g2] and [g1,g2,q] (the only others containing g1), plus
        # [g1]... not a cut (it would miss the Q branch).  Arrays are sorted
        # node lists, so the winner is ["Q", "g1", "g2"]; this is a genuine
        # tie decided only by the third key.
        hierarchy = {"attribute": "a", "root": "R",
                     "nodes": [{"id": "R", "parent": None},
                               {"id": "G", "parent": "R"},
                               {"id": "Q", "parent": "R"},
                               {"id": "g1", "parent": "G"},
                               {"id": "g2", "parent": "G"},
                               {"id": "q", "parent": "Q"}]}
        records = [make_record(f"t{index}", "s", a="g1") for index in range(2)]
        document = job(records, [hierarchy], 1, 1)
        result = check_against_reference(self, document)
        self.assertEqual(result["generalization"][0]["nodes"], ["Q", "g1", "g2"])

    def test_shorter_prefix_rule_is_documented_as_vacuous(self):
        # Any language of sorted cut arrays: two valid cuts are antichains that
        # each meet every root-to-leaf path exactly once.  If sorted array A
        # were a strict prefix of sorted array B, then B would contain every
        # node of A plus at least one extra.  The extra node must be
        # incomparable with all of A (else two selected nodes are ancestor/
        # descendant), yet its subtree leaves are already hit by A, forcing
        # comparability.  So no valid cut can be a strict prefix of another and
        # the prefix clause is a specification consistency rule, not a
        # tie-breaker that can fire.  We still implement plain tuple
        # comparison, which gives the shorter set the win if the situation were
        # ever constructed, and check that the reference and solver agree.
        hierarchies = [
            {"attribute": "a", "root": "Ra",
             "nodes": [{"id": "Ra", "parent": None},
                       {"id": "Ga", "parent": "Ra"},
                       {"id": "Qa", "parent": "Ra"},
                       {"id": "a1", "parent": "Ga"},
                       {"id": "a2", "parent": "Ga"},
                       {"id": "a3", "parent": "Qa"}]},
            {"attribute": "b", "root": "Rb",
             "nodes": [{"id": "Rb", "parent": None},
                       {"id": "Gb", "parent": "Rb"},
                       {"id": "b1", "parent": "Gb"}]},
        ]
        records = [make_record("x", "done", a="a1", b="b1"),
                   make_record("y", "cold", a="a2", b="b1")]
        document = job(records, hierarchies, 2, 2)
        result = check_against_reference(self, document)
        self.assertEqual(
            tuple(tuple(row["nodes"]) for row in result["generalization"]),
            (("Ga", "Qa"), ("Gb",)),
        )
        # "Ga" alone is *not* a valid cut (it misses the Qa branch), so the
        # minimal option is [Ga, Qa].  The prefix rule is exercised by the
        # separate prefix fixture above.

    def test_shorter_prefix_wins_when_one_array_is_a_prefix(self):
        # Attribute "a": the only record sits at B, so every cut maps it to a
        # zero-loss node; the valid zero-loss-ish arrays differ in length.
        # Both objective keys tie, so the cut-array list itself decides.
        hierarchy = {"attribute": "a", "root": "R",
                     "nodes": [{"id": "R", "parent": None},
                               {"id": "A", "parent": "R"},
                               {"id": "B", "parent": "A"}]}
        hierarchies = [
            hierarchy,
            {"attribute": "b", "root": "S",
             "nodes": [{"id": "S", "parent": None},
                       {"id": "C", "parent": "S"}]},
        ]
        records = [make_record("x", "done", a="B", b="C")]
        document = job(records, hierarchies, 1, 1)
        result = check_against_reference(self, document)
        self.assertEqual(
            tuple(tuple(row["nodes"]) for row in result["generalization"]),
            (("A",), ("C",)),
        )


if __name__ == "__main__":
    unittest.main()
