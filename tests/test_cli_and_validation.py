"""Tests for the command line, error paths and the atomic replacement rule."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from anonymizer import anonymize  # noqa: E402
from anonymizer.errors import ValidationError  # noqa: E402

VALID = {
    "records": [
        {"id": "r0", "quasi": {"a": "n0"}, "sensitive": "s1"},
        {"id": "r1", "quasi": {"a": "n1"}, "sensitive": "s2"},
        {"id": "r2", "quasi": {"a": "n1"}, "sensitive": "s3"},
    ],
    "hierarchies": [
        {"attribute": "a", "root": "R",
         "nodes": [{"id": "R", "parent": None},
                   {"id": "P", "parent": "R"},
                   {"id": "n0", "parent": "P"},
                   {"id": "n1", "parent": "P"},
                   {"id": "n2", "parent": "P"}]},
    ],
    "k": 2,
    "l": 2,
}


class ErrorPathTests(unittest.TestCase):
    def check(self, payload: bytes, code: str, path: str) -> None:
        with self.assertRaises(ValidationError) as caught:
            anonymize(payload)
        self.assertEqual(caught.exception.code, code)
        self.assertEqual(caught.exception.path, path)

    def test_missing_top_level_field(self):
        document = dict(VALID)
        del document["l"]
        self.check(json.dumps(document).encode(), "missing_field", "$.l")

    def test_boolean_k_is_not_integer(self):
        document = dict(VALID, k=True)
        self.check(json.dumps(document).encode(), "invalid_type", "$.k")

    def test_l_greater_than_k(self):
        document = dict(VALID, l=4)
        self.check(json.dumps(document).encode(), "l_exceeds_k", "$.l")

    def test_unknown_quasi_value(self):
        document = json.loads(json.dumps(VALID))
        document["records"][0]["quasi"]["a"] = "nope"
        self.check(json.dumps(document).encode(), "unknown_node",
                   "$.records[0].quasi.a")

    def test_non_leaf_quasi_value(self):
        document = json.loads(json.dumps(VALID))
        document["records"][0]["quasi"]["a"] = "P"
        self.check(json.dumps(document).encode(), "not_a_leaf",
                   "$.records[0].quasi.a")

    def test_tree_cycle(self):
        document = json.loads(json.dumps(VALID))
        document["hierarchies"][0]["nodes"][1]["parent"] = "n1"
        document["hierarchies"][0]["nodes"][3]["parent"] = "P"
        self.check(json.dumps(document).encode(), "tree_cycle",
                   "$.hierarchies[0].nodes[1].parent")

    def test_second_root(self):
        document = json.loads(json.dumps(VALID))
        document["hierarchies"][0]["nodes"].append({"id": "gg", "parent": None})
        self.check(json.dumps(document).encode(), "multiple_roots",
                   "$.hierarchies[0].nodes[5].parent")

    def test_unknown_parent(self):
        document = json.loads(json.dumps(VALID))
        document["hierarchies"][0]["nodes"][3]["parent"] = "ghost"
        self.check(json.dumps(document).encode(), "unknown_parent",
                   "$.hierarchies[0].nodes[3].parent")

    def test_empty_sensitive(self):
        document = json.loads(json.dumps(VALID))
        document["records"][0]["sensitive"] = ""
        self.check(json.dumps(document).encode(), "empty_sensitive",
                   "$.records[0].sensitive")

    def test_non_ascii_id(self):
        document = json.loads(json.dumps(VALID))
        document["records"][0]["id"] = "ré"
        self.check(json.dumps(document).encode(), "invalid_identifier",
                   "$.records[0].id")

    def test_duplicate_json_key(self):
        self.check(b'{"records":[],"hierarchies":[],"k":1,"l":1,"k":2}',
                   "duplicate_key", "$.k")

    def test_invalid_utf8(self):
        self.check(b'{"records":[],\xff}', "invalid_utf8", "$")

    def test_invalid_json(self):
        self.check(b'{"records":', "invalid_json", "$")


class CliTests(unittest.TestCase):
    def run_cli(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-I", str(ROOT / "run.py"), *args],
            capture_output=True,
            text=True,
            timeout=60,
        )

    def test_success_writes_stdout(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "job.json"
            path.write_text(json.dumps(VALID), encoding="utf-8")
            completed = self.run_cli(str(path))
        self.assertEqual(completed.returncode, 0, completed.stderr)
        document = json.loads(completed.stdout)
        self.assertEqual(document["input"]["k"], 2)

    def test_failure_keeps_previous_result(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            path = directory_path / "job.json"
            output = directory_path / "out.json"
            path.write_text(json.dumps(VALID), encoding="utf-8")
            output.write_text('{"sentinel": true}', encoding="utf-8")
            bad = dict(VALID, k=0)
            (directory_path / "bad.json").write_text(json.dumps(bad), encoding="utf-8")
            completed = self.run_cli(str(directory_path / "bad.json"),
                                     "--result", str(output))
            self.assertEqual(completed.returncode, 2)
            self.assertEqual(output.read_text(encoding="utf-8"), '{"sentinel": true}')
            failure = json.loads(completed.stderr)
            self.assertTrue(failure["rejected"])
            self.assertEqual(failure["error"]["path"], "$.k")

    def test_stdin(self):
        completed = subprocess.run(
            [sys.executable, "-I", str(ROOT / "run.py"), "-"],
            input=json.dumps(VALID), capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        # k=2,l=2: the publishable classes are {r0,r1,r2}? no -- r0 alone
        # under n0 is below k, so the valid global cut maps all three to P.
        document = json.loads(completed.stdout)
        self.assertEqual(len(document["published"]), 3)
        self.assertEqual(document["suppressed_ids"], [])


if __name__ == "__main__":
    unittest.main()
