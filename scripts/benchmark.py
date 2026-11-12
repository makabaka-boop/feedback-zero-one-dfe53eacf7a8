#!/usr/bin/env python3
"""Small benchmark / smoke harness.

Usage::

    python3 scripts/benchmark.py            # built-in cases
    python3 -I scripts/benchmark.py

It constructs several job sizes and prints objective values and wall-clock
time.  It is a development aid, not part of the deliverable API.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anonymizer import anonymize  # noqa: E402


def job(records, attributes, k, l):
    hierarchies = []
    leaves = {}
    for attribute, shape in attributes.items():
        nodes = [{"id": f"{attribute}root", "parent": None}]
        leaf_ids = []
        for group, children in enumerate(shape):
            group_id = f"{attribute}g{group}"
            nodes.append({"id": group_id, "parent": f"{attribute}root"})
            for index in range(children):
                leaf = f"{attribute}{group}{index}"
                nodes.append({"id": leaf, "parent": group_id})
                leaf_ids.append(leaf)
        leaves[attribute] = leaf_ids
        hierarchies.append(
            {"attribute": attribute, "root": f"{attribute}root", "nodes": nodes}
        )
    rng = random.Random(42)
    payload = []
    for index in range(records):
        payload.append(
            {
                "id": f"r{index:03d}",
                "quasi": {attribute: rng.choice(leaves[attribute])
                          for attribute in attributes},
                "sensitive": f"s{rng.randrange(6)}",
            }
        )
    return {"records": payload, "hierarchies": hierarchies, "k": k, "l": l}


CASES = [
    ("tiny", dict(records=4, attributes={"a": [2, 2]}, k=2, l=2)),
    ("two-attribute", dict(records=40, attributes={"a": [3, 3, 3], "b": [3, 3, 3]}, k=4, l=3)),
    ("three-attribute", dict(records=60, attributes={"a": [3, 3, 3], "b": [3, 3, 3], "c": [3, 3, 3]}, k=5, l=3)),
    ("wide", dict(records=100, attributes={"a": [6, 6, 6], "b": [6, 6, 6]}, k=6, l=4)),
]


def main() -> int:
    for name, kwargs in CASES:
        payload = job(**kwargs)
        start = time.perf_counter()
        document = anonymize(json.dumps(payload).encode())
        elapsed = time.perf_counter() - start
        print(
            f"{name:16s} records={kwargs['records']:3d} "
            f"suppressed={document['objectives']['suppressed_records']:3d} "
            f"loss={document['objectives']['retained_information_loss']:6d} "
            f"time={elapsed:.3f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
