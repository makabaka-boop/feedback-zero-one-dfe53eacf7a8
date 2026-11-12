#!/usr/bin/env python3
"""Acceptance runner: executes every documented check in one pass.

    python3 -I scripts/acceptance.py

It runs the unittest suite, compares a batch of random jobs against the
brute-force reference, re-derives published classes from a real result and
verifies the k/l conditions, and exercises the CLI's atomic-replacement rule.
Exit code 0 means everything passed.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from anonymizer import anonymize  # noqa: E402
from anonymizer.output import verification_errors  # noqa: E402
from generator import make_job  # noqa: E402
from reference import solve_reference  # noqa: E402


def section(title: str) -> None:
    print(f"\n== {title} ==")


def run_unittest() -> bool:
    completed = subprocess.run(
        [sys.executable, "-I", "-m", "unittest", "discover", "-s",
         os.path.join(ROOT, "tests")],
        capture_output=True, text=True, cwd=ROOT,
    )
    tail = completed.stderr.strip().splitlines()[-3:]
    print("\n".join(tail))
    return completed.returncode == 0


def differential(count: int = 150, seed: int = 2718) -> bool:
    rng = random.Random(seed)
    start = time.perf_counter()
    for case in range(count):
        job = make_job(rng, max_nodes=10, max_records=10)
        expected = solve_reference(job)
        result = anonymize(json.dumps(job))
        ok = (
            result["objectives"]["suppressed_records"] == expected.suppressed
            and result["objectives"]["retained_information_loss"]
            == expected.retained_loss
            and tuple(tuple(row["nodes"]) for row in result["generalization"])
            == expected.cuts
            and tuple(result["published"]) == expected.published
            and tuple(result["suppressed_ids"]) == expected.suppressed_ids
        )
        if not ok:
            print(f"mismatch at case {case}")
            return False
    print(f"{count} random jobs matched the independent brute-force reference "
          f"in {time.perf_counter() - start:.2f}s")
    return True


def privacy_recheck() -> bool:
    from anonymizer import run_bytes

    document = {
        "records": [
            {"id": "a", "quasi": {"x": "n1", "y": "m1"}, "sensitive": "s1"},
            {"id": "b", "quasi": {"x": "n1", "y": "m2"}, "sensitive": "s2"},
            {"id": "c", "quasi": {"x": "n2", "y": "m1"}, "sensitive": "s1"},
            {"id": "d", "quasi": {"x": "n2", "y": "m1"}, "sensitive": "s1"},
            {"id": "e", "quasi": {"x": "n2", "y": "m2"}, "sensitive": "s3"},
            {"id": "f", "quasi": {"x": "n2", "y": "m2"}, "sensitive": "s4"},
        ],
        "hierarchies": [
            {"attribute": "x", "root": "R",
             "nodes": [{"id": "R", "parent": None},
                       {"id": "P", "parent": "R"}, {"id": "Q", "parent": "R"},
                       {"id": "n1", "parent": "P"}, {"id": "n2", "parent": "P"}]},
            {"attribute": "y", "root": "S",
             "nodes": [{"id": "S", "parent": None},
                       {"id": "m1", "parent": "S"}, {"id": "m2", "parent": "S"}]},
        ],
        "k": 2,
        "l": 2,
    }
    dataset, solution, result = run_bytes(json.dumps(document).encode())
    # Re-derive classes from scratch, ignoring the solver's own bookkeeping.
    classes: dict[tuple, list] = {}
    for row in result["published"]:
        key = tuple(row["quasi"][attribute] for attribute in dataset.attributes)
        classes.setdefault(key, []).append(row)
    for key, rows in classes.items():
        if len(rows) < dataset.k:
            print(f"class {key} violates k")
            return False
        if len({row["sensitive"] for row in rows}) < dataset.l:
            print(f"class {key} violates l")
            return False
    problems = verification_errors(dataset, result, solution)
    if problems:
        print("\n".join(problems))
        return False
    print(f"recomputed {len(classes)} published class(es); all satisfy k={dataset.k}, "
          f"l={dataset.l}; objective "
          f"suppressed={result['objectives']['suppressed_records']}, "
          f"loss={result['objectives']['retained_information_loss']}")
    return True


def atomic_replacement() -> bool:
    valid = {
        "records": [], "hierarchies": [], "k": 1, "l": 1,
    }
    with tempfile.TemporaryDirectory() as directory:
        job = os.path.join(directory, "job.json")
        good = os.path.join(directory, "good.json")
        bad = os.path.join(directory, "bad.json")
        output = os.path.join(directory, "out.json")
        with open(job, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(valid))
        with open(bad, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"records": [], "hierarchies": [], "k": True, "l": 1}))
        with open(good, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(valid))
        if subprocess.run([sys.executable, "-I", os.path.join(ROOT, "run.py"),
                           good, "--result", output]).returncode != 0:
            print("initial valid run failed")
            return False
        with open(output, "rb") as handle:
            before = handle.read()
        completed = subprocess.run([sys.executable, "-I", os.path.join(ROOT, "run.py"),
                                    bad, "--result", output], capture_output=True, text=True)
        with open(output, "rb") as handle:
            after = handle.read()
    if before != after:
        print("rejected run overwrote the previous result")
        return False
    failure = json.loads(completed.stderr)
    print(f"rejected job kept the previous result; error path {failure['error']['path']}")
    return completed.returncode == 2


def main() -> int:
    checks = [
        ("unittest suite", run_unittest()),
        ("differential vs brute force", differential()),
        ("privacy recheck of published classes", privacy_recheck()),
        ("atomic replacement on rejection", atomic_replacement()),
    ]
    section("results")
    ok = True
    for name, passed in checks:
        print(f"{'PASS' if passed else 'FAIL'}  {name}")
        ok = ok and passed
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
