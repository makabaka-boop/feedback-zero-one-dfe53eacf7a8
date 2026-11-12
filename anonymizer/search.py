"""Exact search for an optimal, l-diverse k-anonymization.

Algorithm outline
-----------------
* For every attribute hierarchy we enumerate every valid generalization cut
  (one node on every root-to-leaf path).  A cut is kept as its ASCII-sorted
  node array, the selected node for every input record, the per-record
  information loss, and the induced partition of records.
* Cuts inducing the same per-attribute partition and the same per-record loss
  are interchangeable for the privacy/utility objectives; only equality of
  generalised values matters.  For such a group only the lexicographically
  smallest node array is retained, which is exactly the tie-break rule at that
  attribute position.
* The per-attribute candidate sets are joined exhaustively with an exact
  dynamic program over refined record partitions.  A DP state is the current
  partition plus the cumulative per-record information loss.  This is
  sufficient because later attributes only refine partitions and add loss
  independently per record.  For each state the lexicographically smallest
  cut-array prefix is kept.
* Every final class is checked against k and l; a failing class is suppressed
  in full (never partially).  Solutions are compared by
  ``(suppressed record count, retained information loss, tie-break key)``.
  Pruning by loss alone is never used.

The tie-break key is the list of per-attribute sorted cut arrays in ASCII
attribute order, compared lexicographically with shorter-prefix-wins; Python
tuple comparison has precisely those semantics.

Because the sorted node array of a cut is a *set* representation, a child
subtree's individually-minimal array is not necessarily the best choice once
several sibling arrays are merged and re-sorted (e.g. arrays ``["a","z"]`` and
``["b"]`` sort to ``["a","b","z"]``, which beats ``["b","m"]``).  The
enumeration therefore never discards a cut at an intermediate node for the
tie-break; it collapses only at the root, after the full set of root cuts is
known.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .model import Dataset, Tree


@dataclass(frozen=True)
class Candidate:
    """One generalisation cut for one attribute."""

    nodes: tuple[str, ...]
    # Selected node id for every record (input order).
    profile: tuple[str, ...]
    # Information loss for every record: descendants(selected)-1.
    losses: tuple[int, ...]
    # Canonical partition of records by their selected node.
    partition: tuple[int, ...]


@dataclass(frozen=True)
class Solution:
    cuts: tuple[tuple[str, ...], ...]  # per attribute, ASCII order
    selected_per_record: tuple[tuple[str, ...], ...]  # attr, record -> node
    suppressed_count: int
    retained_loss: int
    suppressed_ids: tuple[str, ...]
    published: tuple[dict, ...]
    audit: tuple[dict, ...]


def _canonical_partition(labels: Sequence[object]) -> tuple[int, ...]:
    mapping: dict[object, int] = {}
    out: list[int] = []
    for label in labels:
        index = mapping.get(label)
        if index is None:
            index = len(mapping)
            mapping[label] = index
        out.append(index)
    return tuple(out)


def _attribute_candidates(
    tree: Tree, leaf_of: Sequence[str], record_count: int
) -> list[Candidate]:
    """Enumerate every valid cut, then collapse equivalent ones.

    A profile is a tuple of ``(record index, selected node)`` pairs sorted by
    record index, restricted to the subtree under construction.  Child subtrees
    cover disjoint record sets, so merging is a sorted concatenation and the
    node arrays merge by sorted union.
    """

    records_for_leaf: dict[str, list[int]] = {}
    for index, leaf in enumerate(leaf_of):
        records_for_leaf.setdefault(leaf, []).append(index)

    order: list[str] = []
    stack = [tree.root]
    while stack:
        node = stack.pop()
        order.append(node)
        stack.extend(tree.children[node])

    descendants: dict[str, tuple[int, ...]] = {}
    # node -> profile -> smallest ASCII-sorted node array realising that
    # profile inside the subtree.
    raw: dict[str, dict[tuple[tuple[int, str], ...], tuple[str, ...]]] = {}
    for node in reversed(order):
        children = tree.children[node]
        if children:
            record_set: set[int] = set()
            for child in children:
                record_set.update(descendants[child])
            descendants[node] = tuple(sorted(record_set))
        else:
            descendants[node] = tuple(records_for_leaf.get(node, ()))

        own_profile = tuple((index, node) for index in descendants[node])
        candidates: dict[tuple[tuple[int, str], ...], tuple[str, ...]] = {
            own_profile: (node,)
        }

        if children:
            merged: dict[tuple[tuple[int, str], ...], tuple[str, ...]] = {(): ()}
            for child in children:
                combinations: dict[
                    tuple[tuple[int, str], ...], tuple[str, ...]
                ] = {}
                for left_profile, left_nodes in merged.items():
                    for right_profile, right_nodes in raw[child].items():
                        combined = tuple(sorted(left_profile + right_profile))
                        combined_nodes = tuple(sorted(left_nodes + right_nodes))
                        previous = combinations.get(combined)
                        if previous is None or combined_nodes < previous:
                            combinations[combined] = combined_nodes
                merged = combinations
            for merged_profile, merged_nodes in merged.items():
                previous = candidates.get(merged_profile)
                if previous is None or merged_nodes < previous:
                    candidates[merged_profile] = merged_nodes
        raw[node] = candidates

    # Collapse cuts that induce the same partition with the same losses,
    # keeping the smallest sorted-node array (the tie-break at this position).
    compressed: dict[tuple[tuple[int, ...], tuple[int, ...]], Candidate] = {}
    label_cache: dict[tuple[str, ...], tuple[tuple[int, ...], tuple[int, ...]]] = {}
    for profile, nodes in raw[tree.root].items():
        values = tuple(value for _, value in profile)
        if len(values) != record_count:
            raise AssertionError("root profile does not cover every record")
        cached = label_cache.get(values)
        if cached is None:
            losses = tuple(tree.leaf_count[value] - 1 for value in values)
            partition = _canonical_partition(values)
            cached = (partition, losses)
            label_cache[values] = cached
        key = cached
        previous = compressed.get(key)
        if previous is None or nodes < previous.nodes:
            compressed[key] = Candidate(
                nodes=nodes,
                profile=values,
                losses=key[1],
                partition=key[0],
            )
    result = list(compressed.values())
    result.sort(key=lambda candidate: candidate.partition + candidate.losses)
    return result


def _join_partition(
    left: tuple[int, ...], right: tuple[int, ...]
) -> tuple[int, ...]:
    """Refine ``left`` by ``right`` and return the canonical joint partition."""

    return _canonical_partition(tuple(zip(left, right)))


def _prune_losses(
    entries: dict[tuple[int, ...], _Entry]
) -> dict[tuple[int, ...], _Entry]:
    """Drop entries provably no better than another entry of the same state.

    Within a state the current partition is fixed, so a later attribute can
    only refine it further: any continuation refines the same partition in the
    same way for both entries, producing the same classes and therefore the
    same suppressed set.  Consequently, for a fixed continuation:

    * if every per-record loss of ``K`` is <= that of ``B``, then ``K`` has
      weakly lower retained loss;
    * if ``K``'s cut-array prefix is lexicographically <= ``B``'s (both prefixes
      have the same length and receive the same suffix), ``K`` wins any
      tie-break.

    Hence ``B`` can be removed whenever such a ``K`` exists.  Equal loss
    vectors collapse to the lexicographically smallest prefix at insertion
    time, so only strictly better loss vectors matter here.
    """

    if len(entries) <= 1:
        return entries
    kept: list[tuple[tuple[int, ...], _Entry]] = []
    for losses, entry in entries.items():
        dominated = False
        for kept_losses, kept_entry in kept:
            if kept_entry.prefix > entry.prefix:
                continue
            if all(a <= b for a, b in zip(kept_losses, losses)):
                dominated = True
                break
        if not dominated:
            kept.append((losses, entry))
    return dict(kept)


@dataclass(frozen=True)
class _Entry:
    losses: tuple[int, ...]  # cumulative per-record loss
    prefix: tuple[tuple[str, ...], ...]  # cut arrays for attributes processed


def _class_conditions(
    partition: tuple[int, ...],
    records,
    k: int,
    l: int,
) -> tuple[int, set[int], dict[int, bool], dict[int, list[int]]]:
    """Classify classes of ``partition`` against k and l.

    Returns ``(doomed count, doomed record indices, per-class validity,
    per-class member lists)``.

    At an intermediate DP state the partition only gets refined further, so a
    class that already has fewer than ``k`` records or fewer than ``l``
    distinct sensitive values can never become publishable: any final
    sub-class is a subset of it.  Records in such classes are *certainly*
    suppressed in every completion, which gives a valid lower bound for
    branch-and-bound.  On a final partition the same computation is exact.
    """

    classes: dict[int, list[int]] = {}
    for index, class_id in enumerate(partition):
        classes.setdefault(class_id, []).append(index)
    valid: dict[int, bool] = {}
    for class_id, members in classes.items():
        valid[class_id] = (
            len(members) >= k
            and len({records[index].sensitive for index in members}) >= l
        )
    doomed = {
        index
        for class_id, members in classes.items()
        if not valid[class_id]
        for index in members
    }
    return len(doomed), doomed, valid, classes


def _evaluate_partition(
    dataset: Dataset,
    partition: tuple[int, ...],
    entry: _Entry,
) -> tuple[tuple, list[int]]:
    """Score a complete partition and entry against the objective ordering."""

    records = dataset.records
    record_count = len(records)
    doomed_count, doomed, _, _ = _class_conditions(
        partition, records, dataset.k, dataset.l
    )
    suppressed_indices = [
        index for index in range(record_count) if index in doomed
    ]
    retained_loss = sum(
        entry.losses[index]
        for index in range(record_count)
        if index not in doomed
    )
    key = (doomed_count, retained_loss, entry.prefix)
    return key, suppressed_indices


def _root_candidates(
    attribute_candidates: list[list[Candidate]],
) -> list[Candidate]:
    """The single-node (root) cut of each attribute, in attribute order."""

    roots: list[Candidate] = []
    for candidates in attribute_candidates:
        for candidate in candidates:
            if len(candidate.nodes) == 1:
                roots.append(candidate)
                break
        else:  # pragma: no cover - the root cut always exists
            raise AssertionError("missing root cut candidate")
    return roots


def _solve(dataset: Dataset) -> Solution:
    records = dataset.records
    count = len(records)
    attributes = dataset.attributes

    attribute_candidates: list[list[Candidate]] = []
    for attribute in attributes:
        tree = dataset.trees[attribute]
        leaf_of = [record.quasi[attribute] for record in records]
        attribute_candidates.append(_attribute_candidates(tree, leaf_of, count))

    # Seed the incumbent with each attribute's root cut.  A root cut is always
    # a valid global cut, and jointly they put all records into one class.
    best_key: tuple | None = None
    best_solution: Solution | None = None
    if count:
        roots = _root_candidates(attribute_candidates)
        root_losses = tuple(
            sum(candidate.losses[index] for candidate in roots)
            for index in range(count)
        )
        root_entry = _Entry(root_losses, tuple(c.nodes for c in roots))
        key, suppressed_indices = _evaluate_partition(
            dataset, _canonical_partition([()] * count), root_entry
        )
        best_key = key
        best_solution = _build_solution(dataset, suppressed_indices, root_entry)
    else:
        # No records: each attribute's compressed candidate is already the
        # lexicographically smallest cut, and no other objective depends on
        # the cut choice.
        roots = [candidates[0] for candidates in attribute_candidates]
        entry = _Entry((), tuple(candidate.nodes for candidate in roots))
        key = (0, 0, entry.prefix)
        best_key = key
        best_solution = _build_solution(dataset, [], entry)

    if not count:
        assert best_solution is not None
        return best_solution

    # state partition -> cumulative-loss tuple -> best prefix
    zero_losses = (0,) * count
    states: dict[tuple[int, ...], dict[tuple[int, ...], _Entry]] = {
        (0,) * count: {zero_losses: _Entry(zero_losses, ())}
    }
    incumbent_suppressed, incumbent_loss = best_key[0], best_key[1]

    # Attribute-level prefilter: joining partitions only splits classes, so a
    # class of a single-attribute candidate partition can never grow.  If it
    # already fails k or l, every completion keeps it failing.  A candidate
    # that dooms more records than the incumbent's suppressed count can
    # therefore never improve the primary objective.
    doomed_by_partition: dict[tuple[int, ...], int] = {}

    def candidate_doomed(candidate: Candidate) -> int:
        cached = doomed_by_partition.get(candidate.partition)
        if cached is None:
            cached = len(_class_conditions(
                candidate.partition, records, dataset.k, dataset.l
            )[1])
            doomed_by_partition[candidate.partition] = cached
        return cached

    attribute_candidates = [
        [c for c in candidates if candidate_doomed(c) <= incumbent_suppressed]
        for candidates in attribute_candidates
    ]
    for candidates in attribute_candidates:
        if not candidates:  # pragma: no cover - the root cut always survives
            raise AssertionError("attribute prefilter removed every candidate")

    for candidates in attribute_candidates:
        next_states: dict[tuple[int, ...], dict[tuple[int, ...], _Entry]] = {}
        doomed_cache: dict[tuple[int, ...], set[int]] = {}
        for state_partition, loss_entries in states.items():
            doomed = doomed_cache.get(state_partition)
            if doomed is None:
                doomed = _class_conditions(
                    state_partition, records, dataset.k, dataset.l
                )[1]
                doomed_cache[state_partition] = doomed
            if len(doomed) > incumbent_suppressed:
                # Every completion only refines this partition, so these
                # records stay suppressed: the primary objective cannot win.
                continue
            for candidate in candidates:
                new_partition = _join_partition(state_partition, candidate.partition)
                new_doomed = doomed_cache.get(new_partition)
                if new_doomed is None:
                    new_doomed = _class_conditions(
                        new_partition, records, dataset.k, dataset.l
                    )[1]
                    doomed_cache[new_partition] = new_doomed
                if len(new_doomed) > incumbent_suppressed:
                    continue
                bucket = next_states.setdefault(new_partition, {})
                for old_losses, entry in loss_entries.items():
                    if (entry.prefix
                            and len(new_doomed) == incumbent_suppressed):
                        # Lower bound on retained loss: accumulated loss of
                        # records that can still be published is final for
                        # them up to this stage.
                        lower = sum(
                            old_losses[index]
                            for index in range(count)
                            if index not in new_doomed
                        )
                        if lower > incumbent_loss:
                            continue
                    new_losses = tuple(
                        old + extra
                        for old, extra in zip(old_losses, candidate.losses)
                    )
                    new_prefix = entry.prefix + (candidate.nodes,)
                    previous = bucket.get(new_losses)
                    if previous is None or new_prefix < previous.prefix:
                        bucket[new_losses] = _Entry(new_losses, new_prefix)
        states = {
            partition: _prune_losses(entries)
            for partition, entries in next_states.items()
        }

    for partition, loss_entries in states.items():
        for losses, entry in loss_entries.items():
            key, suppressed_indices = _evaluate_partition(
                dataset, partition, entry
            )
            if key < best_key:
                best_key = key
                best_solution = _build_solution(
                    dataset, suppressed_indices, entry
                )

    assert best_solution is not None
    return best_solution


def _build_solution(
    dataset: Dataset,
    suppressed_indices: list[int],
    entry: _Entry,
) -> Solution:
    cuts = entry.prefix
    # A cut plus the original leaf determines the selected ancestor uniquely,
    # so per-record selected nodes need not be stored with the cut.
    selected_per_record: list[tuple[str, ...]] = []
    for attribute, nodes in zip(dataset.attributes, cuts):
        tree = dataset.trees[attribute]
        selected_set = set(nodes)
        per_record: list[str] = []
        for record in dataset.records:
            cursor = record.quasi[attribute]
            steps = 0
            while cursor not in selected_set:
                parent = tree.parent[cursor]
                steps += 1
                if parent is None or steps > len(tree.node_ids):
                    raise AssertionError("cut does not cover a record leaf")
                cursor = parent
            per_record.append(cursor)
        selected_per_record.append(tuple(per_record))

    suppressed_set = set(suppressed_indices)
    published = []
    suppressed_ids = []
    audit = []
    for index, record in enumerate(dataset.records):
        selected = {
            attribute: selected_per_record[attr_index][index]
            for attr_index, attribute in enumerate(dataset.attributes)
        }
        original = {
            attribute: record.quasi[attribute]
            for attribute in dataset.attributes
        }
        if index in suppressed_set:
            suppressed_ids.append(record.id)
        else:
            published.append(
                {
                    "id": record.id,
                    "quasi": selected,
                    "sensitive": record.sensitive,
                }
            )
        audit.append(
            {
                "id": record.id,
                "status": "suppressed" if index in suppressed_set else "published",
                "original": original,
                "generalized": selected,
            }
        )

    published.sort(key=lambda row: row["id"].encode("ascii"))
    suppressed_ids.sort(key=lambda value: value.encode("ascii"))
    audit.sort(key=lambda row: row["id"].encode("ascii"))

    return Solution(
        cuts=cuts,
        selected_per_record=tuple(selected_per_record),
        suppressed_count=len(suppressed_indices),
        retained_loss=sum(
            entry.losses[index] for index in range(len(dataset.records))
            if index not in suppressed_set
        ),
        suppressed_ids=tuple(suppressed_ids),
        published=tuple(published),
        audit=tuple(audit),
    )


def solve(dataset: Dataset) -> Solution:
    """Return the unique optimum under the specified objective ordering."""

    return _solve(dataset)
