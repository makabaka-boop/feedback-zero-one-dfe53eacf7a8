"""Random small-job generator for differential testing."""

from __future__ import annotations

import random


def random_tree(rng: random.Random, attribute: str, max_nodes: int) -> dict:
    """Build a random rooted tree with ASCII node ids unique per attribute."""

    node_count = rng.randint(1, max_nodes)
    ids = [f"{attribute}{index}" for index in range(node_count)]
    nodes = [{"id": ids[0], "parent": None}]
    for index in range(1, node_count):
        parent = ids[rng.randrange(0, index)]
        nodes.append({"id": ids[index], "parent": parent})
    return {"attribute": attribute, "root": ids[0], "nodes": nodes}


def make_job(rng: random.Random, *, max_nodes: int = 7, max_records: int = 7,
             max_attributes: int = 3) -> dict:
    attribute_names = {
        1: ["a"],
        2: ["a", "b"],
        3: ["a", "b", "c"],
    }[rng.randint(1, max_attributes)]
    hierarchies = [random_tree(rng, name, max_nodes) for name in attribute_names]

    trees = {hierarchy["attribute"]: hierarchy for hierarchy in hierarchies}
    leaves = {}
    for attribute, tree in trees.items():
        parents = {node["id"]: node["parent"] for node in tree["nodes"]}
        children = {node["id"]: 0 for node in tree["nodes"]}
        for node in tree["nodes"]:
            if node["parent"] is not None:
                children[node["parent"]] += 1
        leaves[attribute] = [node_id for node_id in parents if children[node_id] == 0]

    k = rng.randint(1, max(1, max_records // 2))
    l = rng.randint(1, k)
    record_count = rng.randint(0, max_records)
    sensitive_pool = ["s1", "s2", "s3"][: rng.randint(1, 3)]
    records = []
    order = list(range(record_count))
    rng.shuffle(order)
    for index in order:
        records.append(
            {
                "id": f"r{index}",
                "quasi": {
                    attribute: rng.choice(leaves[attribute])
                    for attribute in attribute_names
                },
                "sensitive": rng.choice(sensitive_pool),
            }
        )
    return {
        "records": records,
        "hierarchies": hierarchies,
        "k": k,
        "l": l,
    }
