"""Construction and independent verification of the result document."""

from __future__ import annotations

import json
from typing import Any

from .model import Dataset


def build_document(dataset: Dataset, solution) -> dict:
    """Assemble the JSON-serialisable result document."""

    index_by_id = {record.id: index for index, record in enumerate(dataset.records)}

    def record_loss(record_id: str) -> int:
        index = index_by_id[record_id]
        return sum(
            dataset.trees[attribute].leaf_count[
                solution.selected_per_record[attr_index][index]
            ] - 1
            for attr_index, attribute in enumerate(dataset.attributes)
        )

    document = {
        "input": {
            "k": dataset.k,
            "l": dataset.l,
            "record_count": len(dataset.records),
            "attributes": list(dataset.attributes),
        },
        "generalization": [
            {"attribute": attribute, "nodes": list(nodes)}
            for attribute, nodes in zip(dataset.attributes, solution.cuts)
        ],
        "objectives": {
            "suppressed_records": solution.suppressed_count,
            "retained_information_loss": solution.retained_loss,
        },
        "published": [
            {"id": row["id"], "quasi": row["quasi"], "sensitive": row["sensitive"]}
            for row in solution.published
        ],
        "suppressed_ids": list(solution.suppressed_ids),
        "audit": {
            "mapping": [
                {
                    "id": row["id"],
                    "status": row["status"],
                    "original_leaf": row["original"],
                    "generalized": row["generalized"],
                    "information_loss": record_loss(row["id"]),
                }
                for row in solution.audit
            ]
        },
    }
    return document


def _cut_covers_every_path(dataset: Dataset, attribute: str, nodes: list[str]) -> bool:
    tree = dataset.trees[attribute]
    selected = set(nodes)
    if len(selected) != len(nodes):
        return False
    stack = [tree.root]
    while stack:
        node = stack.pop()
        if node in selected:
            continue
        children = tree.children[node]
        if not children:
            return False
        stack.extend(children)
    return True


def _ancestor_or_self(dataset: Dataset, attribute: str, leaf: str, node: str) -> bool:
    tree = dataset.trees[attribute]
    cursor = leaf
    steps = 0
    while cursor is not None:
        if cursor == node:
            return True
        steps += 1
        if steps > len(tree.node_ids):
            return False
        cursor = tree.parent[cursor]
    return False


def verification_errors(dataset: Dataset, document: Any, solution) -> list[str]:
    """Recompute every invariant from scratch; return a list of problems.

    This deliberately does *not* trust the search code: it re-reads the
    emitted generalisation, re-derives classes from the published rows, and
    re-checks the k/l conditions and audit consistency.
    """

    errors: list[str] = []
    try:
        published_rows = document["published"]
        suppressed_ids = document["suppressed_ids"]
        generalisation = document["generalization"]
        objectives = document["objectives"]
        audit_mapping = document["audit"]["mapping"]
    except (KeyError, TypeError) as exc:
        return [f"missing or malformed result field: {exc!r}"]

    if [row["attribute"] for row in generalisation] != list(dataset.attributes):
        errors.append("generalization attributes are not in ASCII order")

    cuts = [list(row["nodes"]) for row in generalisation]
    for attribute, nodes in zip(dataset.attributes, cuts):
        if nodes != sorted(nodes):
            errors.append(f"cut nodes for {attribute!r} are not ASCII-sorted")
        if not _cut_covers_every_path(dataset, attribute, nodes):
            errors.append(f"cut for {attribute!r} is not a valid global cut")

    original_by_id = {record.id: record for record in dataset.records}
    published_ids = [row["id"] for row in published_rows]
    if len(published_ids) != len(set(published_ids)):
        errors.append("duplicate published id")
    if not set(published_ids).isdisjoint(suppressed_ids):
        errors.append("id appears both published and suppressed")
    if len(published_ids) + len(suppressed_ids) != len(dataset.records):
        errors.append("published + suppressed counts do not cover all records")
    if suppressed_ids != sorted(suppressed_ids, key=lambda value: value.encode("ascii")):
        errors.append("suppressed ids are not sorted")
    if published_ids != sorted(published_ids, key=lambda value: value.encode("ascii")):
        errors.append("published rows are not sorted by id")
    unknown = [value for value in published_ids + suppressed_ids
               if value not in original_by_id]
    if unknown:
        errors.append(f"unknown ids in output: {unknown[:3]}")

    # Every published value must be the selected ancestor of the original leaf.
    for row in published_rows:
        record = original_by_id[row["id"]]
        for attr_index, attribute in enumerate(dataset.attributes):
            if row["quasi"][attribute] not in cuts[attr_index]:
                errors.append(
                    f"published value for {row['id']!r}/{attribute!r} is not on the cut"
                )
                continue
            if not _ancestor_or_self(
                dataset, attribute, record.quasi[attribute],
                row["quasi"][attribute],
            ):
                errors.append(
                    f"published value for {row['id']!r}/{attribute!r} is not "
                    "an ancestor of the original leaf"
                )
        if row["sensitive"] != record.sensitive:
            errors.append(f"sensitive value changed for {row['id']!r}")

    # Reconstruct classes from the published quasi tuples.
    classes: dict[tuple[str, ...], list[dict]] = {}
    for row in published_rows:
        key = tuple(row["quasi"][attribute] for attribute in dataset.attributes)
        classes.setdefault(key, []).append(row)
    for key, members in classes.items():
        sensitivities = {row["sensitive"] for row in members}
        if len(members) < dataset.k:
            errors.append(f"published class {key!r} has fewer than k records")
        if len(sensitivities) < dataset.l:
            errors.append(f"published class {key!r} has fewer than l sensitive values")

    if objectives.get("suppressed_records") != len(suppressed_ids):
        errors.append("suppressed_records objective does not match suppressed ids")
    if objectives.get("suppressed_records") != len(solution.suppressed_ids):
        errors.append("suppressed_records objective does not match the solution")
    if len(suppressed_ids) != len(solution.suppressed_ids):
        errors.append("suppressed id count does not match the solution")

    # Audit must cover every record exactly, with the same outcome and a
    # recorded ancestor for every attribute.
    audit_ids = [row["id"] for row in audit_mapping]
    if audit_ids != sorted(original_by_id, key=lambda value: value.encode("ascii")):
        errors.append("audit mapping does not cover every record exactly once")
    published_set = set(published_ids)
    for row in audit_mapping:
        record = original_by_id.get(row["id"])
        if record is None:
            errors.append(f"audit references unknown id {row['id']!r}")
            continue
        expected_status = "published" if row["id"] in published_set else "suppressed"
        if row.get("status") != expected_status:
            errors.append(f"audit status mismatch for {row['id']!r}")
        if row.get("original_leaf") != {
            attribute: record.quasi[attribute] for attribute in dataset.attributes
        }:
            errors.append(f"audit original_leaf mismatch for {row['id']!r}")
        for attr_index, attribute in enumerate(dataset.attributes):
            mapped = row.get("generalized", {}).get(attribute)
            if mapped not in cuts[attr_index]:
                errors.append(f"audit mapped node for {row['id']!r}/{attribute!r} is not on the cut")
            elif not _ancestor_or_self(dataset, attribute,
                                       record.quasi[attribute], mapped):
                errors.append(f"audit mapped node for {row['id']!r}/{attribute!r} is not an ancestor")

    return errors


def encode_document(document: dict) -> bytes:
    return (json.dumps(
        document,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=False,
    ) + "\n").encode("utf-8")
