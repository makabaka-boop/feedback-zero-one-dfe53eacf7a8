# Exact k-anonymous, l-diverse anonymisation job

A self-contained Python 3.11 implementation of a global-recoding
k-anonymity / l-diversity backend. Implementation lives in `anonymizer/`.
No third-party dependency, no generic ILP/SAT/optimisation solver, no
anonymisation library: the cut enumeration, the exact search over attributes,
the objective comparison and the tie-break are all implemented here.

## What the job does

Input is UTF-8 JSON with:

| field | meaning |
| --- | --- |
| `records` | array of `{"id", "quasi", "sensitive"}` |
| `hierarchies` | one per attribute: `"attribute"`, `"root"`, `"nodes"` |
| `k` | positive integer (`bool` is rejected) |
| `l` | positive integer with `l <= k` |

Each record has a unique, non-empty, **ASCII** `id`; `quasi` has exactly one
entry per hierarchy attribute; each value is a **leaf** of that attribute's
tree. `sensitive` is a non-empty string (arbitrary Unicode).

Each hierarchy node has a unique (per attribute) ASCII id, one root, and one
parent per non-root node. Per attribute the tree must be connected, acyclic,
single-rooted. Attributes and node-id namespaces are isolated: the same node id
may be used in two attributes.

Every attribute gets one **global cut**: a set of nodes meeting every
root-to-leaf path exactly once. All records' leaf values are mapped to the
selected ancestor on that path. Records with an equal full generalised quasi
tuple form a class; a class is published only if it (a) has at least `k`
records and (b) has at least `l` distinct original `sensitive` values. A class
failing either condition is suppressed **in its entirety** — never partially.

### Objective ordering (exact, global)

1. minimise the number of suppressed records;
2. then minimise the total information loss of the retained records, where one
   record's loss is `sum over attributes of (descendant leaf count of its
   mapped node - 1)`;
3. then the tie-break: attributes sorted ASCII-wise, each attribute's selected
   node array sorted ASCII-wise, the list of these arrays compared
   lexicographically (shorter prefix wins; Python tuple comparison matches the
   specification exactly).

The solver searches the whole space with an exact dynamic program over record
partitions; no greedy field-by-field choice is made, so records that can be
kept jointly are never lost to per-field greediness. Candidate cuts are
collapsed only when they induce the **same record partition with the same per
record loss**, in which case they are interchangeable for privacy and loss;
for such a group the smallest ASCII-sorted node array is retained, which is
exactly that attribute position's tie-break. Exhaustive differential tests
against an independent brute-force reference (`tests/reference.py`) confirm
equal optimum, equal objectives, equal cut arrays, equal published rows.

## Output

The result document contains:

- `input` — echoed `k`, `l`, record count and ASCII-ordered attributes;
- `generalization` — per attribute, the ASCII-sorted cut node array;
- `objectives` — `suppressed_records` and `retained_information_loss`;
- `published` — rows sorted by `id`, each `{id, quasi, sensitive}` with the
  original sensitive value preserved;
- `suppressed_ids` — sorted by `id`;
- `audit.mapping` — every record (sorted by `id`) with `status`,
  `original_leaf`, `generalized` and its `information_loss`, i.e. the original
  leaf → selected ancestor mapping per attribute.

After building the result the program runs an **independent verification pass**
(re-reads the generalisation, recomputes classes, re-checks k/l, cut validity,
ancestry, sorting, completeness and audit) before emitting anything. A
verification failure is an internal error, never a published result.

## Rejection semantics

Any illegal field, tree shape, or reference rejects the whole job; nothing is
partially accepted. Every rejection carries a JSON path and an RFC 6901 pointer,
e.g.:

```json
{"ok": false, "rejected": true,
 "error": {"code": "not_a_leaf",
           "path": "$.records[0].quasi.a",
           "json_pointer": "/records/0/quasi/a",
           "message": "value 'P' is not a leaf of attribute 'a'"}}
```

Validation also rejects duplicate JSON keys, invalid UTF-8, invalid JSON,
non-finite numbers, booleans used as integers, non-ASCII identifiers, empty
sensitive values, duplicate attributes/node ids/record ids, missing or extra
unknown fields, unknown parents, self-parents, cycles, multiple roots, and
quasi values that are not leaves.

Only a successful job replaces the previous output. With `--result PATH` the
new result is written to a temporary file and `os.replace`d over the target, so
a rejected run leaves the previous file byte-for-byte intact. It is an error to
treat an old result as a new input: unknown top-level fields are rejected.

## Running

```bash
# library
python3 - <<'PY'
import json
from anonymizer import anonymize, dump_result
data = open("examples/job.json", "rb").read()
result = anonymize(data)          # dict; raises ValidationError on bad input
print(dump_result(result))
PY

# CLI (works from any cwd; -I is safe because run.py inserts its own dir)
python3 -I run.py examples/job.json --result /tmp/result.json --summary
python3 -I run.py - < examples/job.json      # stdin; result on stdout
python3 -I -m anonymizer examples/job.json   # equivalent, needs the package on sys.path

# tests
python3 -I -m unittest discover -s tests -v
```

The CLI exits 0 on success, 2 on input rejection (JSON failure body on
stderr), 3 on an internal/verification failure.

## Verification and acceptance

`python3 -I -m unittest discover -s tests -v` runs:

- `tests/test_differential.py` — 200 random small jobs compared field by field
  with the independent brute-force reference (`tests/reference.py`, which
  enumerates all cuts by subset filtering, a deliberately different algorithm);
- `tests/test_scenarios.py` — required scenarios: cross-attribute joint
  optimum (a greedy per-field run loses records), duplicate sensitive values
  (distinct-count semantics), full suppression, empty records, single-leaf
  trees, ASCII attribute ordering and genuine tie-break cases;
- `tests/test_cli_and_validation.py` — every rejection class, JSON-path
  reporting, CLI output, and the "failure keeps the previous result" rule;
- `tests/test_verification.py` — the independent verifier catches corrupted
  k/l/ancestry/audit results;
- the run_bytes entry automatically re-derives published classes and re-checks
  the privacy conditions on every successful job (this is the "recompute
  published classes and validate privacy" acceptance item, not a test-only
  helper).

`scripts/benchmark.py` prints objective values and timings for built-in sizes.

### Known computational bound

The search is exact, so its worst case is exponential in the number of
attributes and record partitions (NP-hard in general). Hierarchies with very
large balanced branching and dozens of distinct record patterns can be slow
(e.g. two 32-leaf binary hierarchies with 32 mutually distinct records took
about 20 s here). Everyday sizes (a handful of attributes, tens to hundreds of
records over moderate trees) run in milliseconds to low seconds; see
`scripts/benchmark.py`.
