"""Strict input parsing and validation for anonymization jobs.

JSON shape (all levels reject unknown keys)::

    {
      "records": [
        {"id": "...", "quasi": {"Attr": "leaf-node-id", ...}, "sensitive": "..."}
      ],
      "hierarchies": [
        {"attribute": "...", "root": "...",
         "nodes": [{"id": "...", "parent": null | "..."}, ...]}
      ],
      "k": 3,
      "l": 2
    }

Every record's ``quasi`` object has exactly one entry per hierarchy, and its
value names a leaf of that hierarchy.  Node identifiers are unique inside an
attribute (attributes are isolated namespaces) but may repeat across
attributes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .errors import InputJSONError, fail

TOP_FIELDS = frozenset(("records", "hierarchies", "k", "l"))
RECORD_FIELDS = frozenset(("id", "quasi", "sensitive"))
HIERARCHY_FIELDS = frozenset(("attribute", "root", "nodes"))
NODE_FIELDS = frozenset(("id", "parent"))


class _JsonObject(dict):
    """A dict that remembers keys which occurred more than once."""

    duplicate_keys: tuple[str, ...]

    def __init__(self, pairs: list[tuple[str, Any]]) -> None:
        super().__init__(pairs)
        seen: set[str] = set()
        duplicates: list[str] = []
        for key, _ in pairs:
            if key in seen:
                duplicates.append(key)
            seen.add(key)
        self.duplicate_keys = tuple(duplicates)


def _reject_constant(token: str) -> Any:
    raise ValueError(f"non-finite JSON number {token!r} is not allowed")


def _scan_duplicates(value: Any, path: tuple[Any, ...]) -> None:
    if isinstance(value, _JsonObject):
        if value.duplicate_keys:
            fail(path + (value.duplicate_keys[0],), "duplicate_key",
                 "object key occurs more than once")
        for key, child in value.items():
            _scan_duplicates(child, path + (key,))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _scan_duplicates(child, path + (index,))


def _require_type(value: Any, expected: type, path: Iterable[Any], code: str,
                  description: str) -> None:
    if expected is int:
        # bool is a subclass of int in Python but is not an integer here.
        if type(value) is not int:
            fail(path, code, f"expected {description}")
    elif expected is str:
        if type(value) is not str:
            fail(path, code, f"expected {description}")
    elif expected is list:
        if type(value) is not list:
            fail(path, code, f"expected {description}")
    elif not isinstance(value, expected):
        # Parsed objects are dict subclasses (duplicate-key aware), so plain
        # isinstance is the right check for objects.
        fail(path, code, f"expected {description}")


def _is_ascii_nonempty(value: str) -> bool:
    return bool(value) and all(ord(ch) < 128 for ch in value)


def _has_lone_surrogate(value: str) -> bool:
    try:
        value.encode("utf-8", "strict")
    except UnicodeEncodeError:
        return True
    return False


def _check_ascii_identifier(value: str, path: Iterable[Any], what: str) -> None:
    if not _is_ascii_nonempty(value):
        fail(path, "invalid_identifier",
             f"{what} must be a non-empty ASCII string")


def _check_object_fields(
    obj: Mapping[str, Any],
    required: frozenset[str],
    path: tuple[Any, ...],
) -> None:
    for key in obj:
        if key not in required:
            fail(path + (key,), "unknown_field", f"unknown field {key!r}")
    for key in required:
        if key not in obj:
            fail(path + (key,), "missing_field", f"required field {key!r} is missing")


def decode_bytes(data: bytes) -> Any:
    """Decode UTF-8 bytes and parse JSON without silently losing information."""

    try:
        text = data.decode("utf-8", "strict")
    except UnicodeDecodeError as exc:
        raise InputJSONError(
            "invalid_utf8", (), "input is not valid UTF-8",
            detail=f"byte offset {exc.start}",
        ) from None
    try:
        parsed = json.loads(
            text,
            object_pairs_hook=_JsonObject,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise InputJSONError(
            "invalid_json", (), "input is not valid JSON",
            detail=f"line {exc.lineno}, column {exc.colno}: {exc.msg}",
        ) from None
    except (ValueError, RecursionError) as exc:
        raise InputJSONError(
            "invalid_json", (), "input is not valid JSON",
            detail=str(exc),
        ) from None
    _scan_duplicates(parsed, ())
    return parsed


@dataclass(frozen=True)
class Record:
    id: str
    quasi: Mapping[str, str]
    sensitive: str


@dataclass
class Tree:
    attribute: str
    root: str
    node_ids: tuple[str, ...]
    parent: Mapping[str, str | None]
    children: Mapping[str, tuple[str, ...]]
    leaves: tuple[str, ...]
    leaf_count: Mapping[str, int]


@dataclass
class Dataset:
    records: tuple[Record, ...]
    trees: Mapping[str, Tree]
    attributes: tuple[str, ...]
    k: int
    l: int


def _build_tree(
    hierarchy: _JsonObject,
    path: tuple[Any, ...],
) -> Tree:
    _require_type(hierarchy, dict, path, "invalid_type", "an object")
    _check_object_fields(hierarchy, HIERARCHY_FIELDS, path)
    attribute = hierarchy["attribute"]
    _require_type(attribute, str, path + ("attribute",), "invalid_type", "a string")
    _check_ascii_identifier(attribute, path + ("attribute",), "attribute")
    root = hierarchy["root"]
    _require_type(root, str, path + ("root",), "invalid_type", "a string")
    _check_ascii_identifier(root, path + ("root",), "root node id")

    nodes = hierarchy["nodes"]
    _require_type(nodes, list, path + ("nodes",), "invalid_type", "an array")
    if not nodes:
        fail(path + ("nodes",), "empty_tree", "a hierarchy must contain at least one node")

    node_ids: list[str] = []
    parent: dict[str, str | None] = {}
    for index, node in enumerate(nodes):
        node_path = path + ("nodes", index)
        _require_type(node, dict, node_path, "invalid_type", "an object")
        for key in node:
            if key not in NODE_FIELDS:
                fail(node_path + (key,), "unknown_field", f"unknown field {key!r}")
        if "id" not in node:
            fail(node_path + ("id",), "missing_field", "required field 'id' is missing")
        node_id = node["id"]
        _require_type(node_id, str, node_path + ("id",), "invalid_type", "a string")
        _check_ascii_identifier(node_id, node_path + ("id",), "node id")
        if node_id in parent:
            fail(node_path + ("id",), "duplicate_node_id",
                 f"node id {node_id!r} occurs more than once in this attribute")
        node_ids.append(node_id)

        if "parent" in node:
            raw_parent = node["parent"]
            if raw_parent is None:
                parent[node_id] = None
            else:
                _require_type(raw_parent, str, node_path + ("parent",),
                              "invalid_type", "a string or null")
                _check_ascii_identifier(raw_parent, node_path + ("parent",),
                                        "parent node id")
                parent[node_id] = raw_parent
        else:
            # Only the declared root may omit its parent entry.
            if node_id != root:
                fail(node_path, "missing_parent",
                     "a non-root node must have a parent")
            parent[node_id] = None

    if root not in parent:
        fail(path + ("root",), "unknown_root",
             f"root {root!r} does not name a node in this attribute")
    root_index = node_ids.index(root)
    if parent[root] is not None:
        fail(path + ("nodes", root_index, "parent"), "root_has_parent",
             "the declared root must not have a parent")

    for index, node_id in enumerate(node_ids):
        node_parent = parent[node_id]
        if node_id == root:
            continue
        if node_parent is None:
            fail(path + ("nodes", index, "parent"), "multiple_roots",
                 "only the declared root may have no parent")
        if node_parent not in parent:
            fail(path + ("nodes", index, "parent"), "unknown_parent",
                 f"parent {node_parent!r} does not name a node in this attribute")
        if node_parent == node_id:
            fail(path + ("nodes", index, "parent"), "self_parent",
                 "a node may not be its own parent")

    children: dict[str, tuple[str, ...]] = {node_id: () for node_id in node_ids}
    child_lists: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
    for index, node_id in enumerate(node_ids):
        node_parent = parent[node_id]
        if node_parent is not None:
            child_lists[node_parent].append(node_id)
    for node_id in node_ids:
        children[node_id] = tuple(child_lists[node_id])

    # Cycle detection over the single-parent graph (before reachability, so a
    # detached cycle is reported clearly rather than as a generic disconnect).
    completed: set[str] = set()
    for start in node_ids:
        chain: list[str] = []
        chain_seen: set[str] = set()
        cursor: str | None = start
        while cursor is not None and cursor not in completed:
            if cursor in chain_seen:
                fail(path + ("nodes", node_ids.index(cursor), "parent"),
                     "tree_cycle", "parent links contain a cycle")
            chain_seen.add(cursor)
            chain.append(cursor)
            cursor = parent[cursor]
        completed.update(chain)

    # Reachability from the declared root; with one parent each this is also
    # the connectivity/one-root requirement.
    visited: set[str] = set()
    stack = [root]
    while stack:
        node_id = stack.pop()
        if node_id in visited:
            fail(path + ("nodes", node_ids.index(node_id) if node_id in node_ids else 0),
                 "tree_cycle", "child links contain a cycle")
        visited.add(node_id)
        stack.extend(children[node_id])
    if len(visited) != len(node_ids):
        missing = next(node_id for node_id in node_ids if node_id not in visited)
        fail(path + ("nodes", node_ids.index(missing)),
             "tree_disconnected",
             f"node {missing!r} is not reachable from the declared root")

    leaves = tuple(node_id for node_id in node_ids if not children[node_id])

    # Post-order leaf counts, without recursion.
    counts: dict[str, int] = {}
    post_order: list[str] = []
    stack = [root]
    while stack:
        node_id = stack.pop()
        post_order.append(node_id)
        stack.extend(children[node_id])
    for node_id in reversed(post_order):
        if not children[node_id]:
            counts[node_id] = 1
        else:
            counts[node_id] = sum(counts[child] for child in children[node_id])

    return Tree(
        attribute=attribute,
        root=root,
        node_ids=tuple(node_ids),
        parent=dict(parent),
        children=children,
        leaves=leaves,
        leaf_count=counts,
    )


def _validate_int(value: Any, path: tuple[Any, ...], name: str, minimum: int) -> int:
    _require_type(value, int, path, "invalid_type", "an integer (booleans are not integers)")
    if value < minimum:
        fail(path, "out_of_range", f"{name} must be a positive integer")
    return value


def parse_document(document: Any) -> Dataset:
    """Validate an already-JSON-parsed document and build the input model."""

    _require_type(document, dict, (), "invalid_type", "a JSON object")
    _check_object_fields(document, TOP_FIELDS, ())

    hierarchies = document["hierarchies"]
    _require_type(hierarchies, list, ("hierarchies",), "invalid_type", "an array")
    trees: dict[str, Tree] = {}
    for index, hierarchy in enumerate(hierarchies):
        path = ("hierarchies", index)
        tree = _build_tree(hierarchy, path)
        if tree.attribute in trees:
            fail(path + ("attribute",), "duplicate_attribute",
                 f"attribute {tree.attribute!r} has more than one hierarchy")
        trees[tree.attribute] = tree

    # ASCII order is the required deterministic attribute order.
    attributes = tuple(sorted(trees, key=lambda value: value.encode("ascii")))

    k = _validate_int(document["k"], ("k",), "k", 1)
    l = _validate_int(document["l"], ("l",), "l", 1)
    if l > k:
        fail(("l",), "l_exceeds_k", "l must be less than or equal to k")

    records = document["records"]
    _require_type(records, list, ("records",), "invalid_type", "an array")
    parsed_records: list[Record] = []
    seen_ids: set[str] = set()
    for index, raw_record in enumerate(records):
        race_path = ("records", index)
        _require_type(raw_record, dict, race_path, "invalid_type", "an object")
        _check_object_fields(raw_record, RECORD_FIELDS, race_path)

        record_id = raw_record["id"]
        _require_type(record_id, str, race_path + ("id",), "invalid_type", "a string")
        _check_ascii_identifier(record_id, race_path + ("id",), "record id")
        if record_id in seen_ids:
            fail(race_path + ("id",), "duplicate_record_id",
                 f"record id {record_id!r} occurs more than once")
        seen_ids.add(record_id)

        quasi = raw_record["quasi"]
        _require_type(quasi, dict, race_path + ("quasi",), "invalid_type", "an object")
        for key in quasi:
            if key not in trees:
                fail(race_path + ("quasi", key), "unknown_quasi_attribute",
                     f"no hierarchy exists for quasi attribute {key!r}")
        for attribute in attributes:
            if attribute not in quasi:
                fail(race_path + ("quasi", attribute), "missing_quasi_attribute",
                     f"quasi is missing attribute {attribute!r}")
            value = quasi[attribute]
            _require_type(value, str, race_path + ("quasi", attribute),
                          "invalid_type", "a string node id")
            tree = trees[attribute]
            if value not in tree.parent:
                fail(race_path + ("quasi", attribute), "unknown_node",
                     f"value {value!r} is not a node of attribute {attribute!r}")
            if value not in tree.leaves:
                fail(race_path + ("quasi", attribute), "not_a_leaf",
                     f"value {value!r} is not a leaf of attribute {attribute!r}")
        # Preserve a canonical attribute ordering in the model; only ASCII
        # identifiers are accepted above, so plain sorting is ASCII sorting.
        canonical_quasi = {attribute: quasi[attribute] for attribute in attributes}

        sensitive = raw_record["sensitive"]
        _require_type(sensitive, str, race_path + ("sensitive",),
                      "invalid_type", "a string")
        if sensitive == "":
            fail(race_path + ("sensitive",), "empty_sensitive",
                 "sensitive value must be a non-empty string")
        if _has_lone_surrogate(sensitive):
            fail(race_path + ("sensitive",), "invalid_unicode",
                 "sensitive value contains an unpaired surrogate")

        parsed_records.append(Record(record_id, canonical_quasi, sensitive))

    return Dataset(
        records=tuple(parsed_records),
        trees=trees,
        attributes=attributes,
        k=k,
        l=l,
    )


def load_bytes(data: bytes) -> Dataset:
    """Full UTF-8/JSON/field/tree/reference validation."""

    return parse_document(decode_bytes(data))


def load_text(text: str) -> Dataset:
    return load_bytes(text.encode("utf-8"))
