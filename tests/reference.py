"""Brute-force reference solver used only by the test-suite.

It shares nothing with :mod:`anonymizer.search`: every global cut is generated
directly from the tree, the full Cartesian product is scored naively, and the
key construction is written independently from the specification text.

The reference key is
``(suppressed, retained loss, cut-array list)`` where the cut-array list is
compared lexicographically (shorter prefix first).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

Record = dict
Tree = dict


def cuts_of(tree: Tree) -> list[tuple[str, ...]]:
    """All global cuts of ``tree`` as sorted node arrays, generated directly."""

    root = tree["root"]
    parent = {node["id"]: node.get("parent") for node in tree["nodes"]}
    children: dict[str, list[str]] = {node["id"]: [] for node in tree["nodes"]}
    for node in tree["nodes"]:
        if node.get("parent") is not None:
            children[node["parent"]].append(node["id"])

    # Every cut is a set that hits every root-leaf path exactly once.  Enumerate
    # all node subsets of manageable size and filter, which is deliberately a
    # different algorithm from the bottom-up enumeration under test.
    node_ids = [node["id"] for node in tree["nodes"]]

    def is_cut(candidate: frozenset) -> bool:
        stack = [root]
        while stack:
            node = stack.pop()
            if node in candidate:
                continue
            if not children[node]:
                return False
            stack.extend(children[node])
        # Each root-to-leaf path must hit exactly one selected node, so the
        # selected nodes form an antichain: no node may be an ancestor of
        # another (incomparable siblings in different branches are fine).
        for left, right in itertools.combinations(candidate, 2):
            if _ancestor(parent, left, right) or _ancestor(parent, right, left):
                return False
        return True

    result = []
    for size in range(1, len(node_ids) + 1):
        for subset in itertools.combinations(node_ids, size):
            candle = frozenset(subset)
            if is_cut(candle):
                result.append(tuple(sorted(candle)))
    return result


def _ancestor(parent: dict, maybe_ancestor: str, node: str) -> bool:
    cursor = parent[node]
    while cursor is not None:
        if cursor == maybe_ancestor:
            return True
        cursor = parent[cursor]
    return False


def _selected(parent: dict, leaf: str, cut: frozenset) -> str:
    cursor = leaf
    while cursor not in cut:
        cursor = parent[cursor]
    return cursor


def leaf_counts(tree: Tree) -> dict[str, int]:
    parent = {node["id"]: node.get("parent") for node in tree["nodes"]}
    children: dict[str, list[str]] = {node["id"]: [] for node in tree["nodes"]}
    for node in tree["nodes"]:
        if node.get("parent") is not None:
            children[node["parent"]].append(node["id"])
    counts: dict[str, int] = {}

    def walk(node: str) -> int:
        if not children[node]:
            counts[node] = 1
        else:
            counts[node] = sum(walk(child) for child in children[node])
        return counts[node]

    walk(tree["root"])
    return counts


@dataclass(frozen=True)
class ReferenceResult:
    suppressed: int
    retained_loss: int
    cuts: tuple[tuple[str, ...], ...]
    published: tuple[dict, ...]
    suppressed_ids: tuple[str, ...]


def solve_reference(job: dict) -> ReferenceResult:
    records = job["records"]
    trees = {hierarchy["attribute"]: hierarchy for hierarchy in job["hierarchies"]}
    attributes = sorted(trees)
    k = job["k"]
    l = job["l"]
    count = len(records)

    per_attribute_cuts = [
        (attribute, cuts_of(trees[attribute]))
        for attribute in attributes
    ]
    parents = {
        attribute: {node["id"]: node.get("parent") for node in trees[attribute]["nodes"]}
        for attribute in attributes
    }
    counts = {
        attribute: leaf_counts(trees[attribute]) for attribute in attributes
    }

    best = None
    for combination in itertools.product(*[cuts for _, cuts in per_attribute_cuts]):
        cut_sets = [frozenset(cut) for cut in combination]
        values = []
        losses = []
        for record in records:
            row_values = []
            row_loss = 0
            for index, attribute in enumerate(attributes):
                selected = _selected(parents[attribute], record["quasi"][attribute],
                                     cut_sets[index])
                row_values.append(selected)
                row_loss += counts[attribute][selected] - 1
            values.append(tuple(row_values))
            losses.append(row_loss)

        classes: dict[tuple[str, ...], list[int]] = {}
        for index, key in enumerate(values):
            classes.setdefault(key, []).append(index)
        published_indices = []
        suppressed_indices = []
        for key, members in classes.items():
            sensitives = {records[index]["sensitive"] for index in members}
            if len(members) >= k and len(sensitives) >= l:
                published_indices.extend(members)
            else:
                suppressed_indices.extend(members)
        retained_loss = sum(losses[index] for index in published_indices)
        key = (len(suppressed_indices), retained_loss, tuple(combination))
        if best is None or key < best[0]:
            best = (key, values, losses, published_indices, suppressed_indices)

    assert best is not None
    key, values, losses, published_indices, suppressed_indices = best
    published = tuple(
        sorted(
            (
                {
                    "id": records[index]["id"],
                    "quasi": {
                        attribute: values[index][attr_index]
                        for attr_index, attribute in enumerate(attributes)
                    },
                    "sensitive": records[index]["sensitive"],
                }
                for index in published_indices
            ),
            key=lambda row: row["id"].encode("ascii"),
        )
    )
    return ReferenceResult(
        suppressed=key[0],
        retained_loss=key[1],
        cuts=key[2],
        published=published,
        suppressed_ids=tuple(
            sorted(
                (records[index]["id"] for index in suppressed_indices),
                key=lambda value: value.encode("ascii"),
            )
        ),
    )
