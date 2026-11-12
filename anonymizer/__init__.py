"""Public entry points for the exact anonymization job.

Library use::

    from anonymizer import anonymize
    result = anonymize(payload_bytes)   # dict; raises ValidationError
    text = dump_result(result)          # canonical JSON text

Command line::

    python -m anonymizer INPUT.json [--result out.json] [--summary]
    python -m anonymizer - < input.json

A rejected job raises/returns before any output file is touched; on success
the result file is replaced atomically, so a previously written result
survives every failed run.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from typing import Any

from .errors import AnonymizationError, InputJSONError, ValidationError
from .model import Dataset, load_bytes
from .output import build_document, encode_document, verification_errors
from .search import Solution, solve


class InternalError(AnonymizationError):
    """The solver produced a result that fails independent verification."""


def run_bytes(data: bytes, *, verify: bool = True) -> tuple[Dataset, Solution, dict]:
    dataset = load_bytes(data)
    solution = solve(dataset)
    document = build_document(dataset, solution)
    if verify:
        problems = verification_errors(dataset, document, solution)
        if problems:
            raise InternalError("; ".join(problems))
    return dataset, solution, document


def anonymize(data: bytes | str, *, verify: bool = True) -> dict[str, Any]:
    """Validate, solve and return the result document as a Python dict."""

    if isinstance(data, str):
        data = data.encode("utf-8")
    if not isinstance(data, bytes):
        raise TypeError("anonymize() accepts bytes or str")
    return run_bytes(data, verify=verify)[2]


def dump_result(result: dict[str, Any]) -> str:
    return encode_document(result).decode("utf-8")


def _atomic_write(path: str, payload: bytes) -> None:
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".anonymizer-", dir=directory)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def failure_document(exc: AnonymizationError) -> dict[str, Any]:
    """Machine-readable failure body; every rejection carries a path."""

    if isinstance(exc, ValidationError):
        return {
            "ok": False,
            "rejected": True,
            "error": exc.to_dict(),
        }
    return {
        "ok": False,
        "rejected": True,
        "error": {
            "code": "internal_error",
            "path": "$",
            "json_pointer": "",
            "message": str(exc),
        },
    }


def _read_bytes(path: str) -> bytes:
    try:
        with open(path, "rb") as handle:
            return handle.read()
    except OSError as exc:
        raise InputJSONError(
            "unreadable_input", (), f"cannot read input file {path!r}",
            detail=str(exc),
        ) from None


def run_cli(args: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m anonymizer",
        description="Exact k-anonymous / l-diverse global-recoding solver.",
    )
    parser.add_argument("input", nargs="?", default="-",
                        help="UTF-8 JSON job file (default '-' = stdin)")
    parser.add_argument("--result", metavar="PATH",
                        help="atomically write the result document here")
    parser.add_argument("--summary", action="store_true",
                        help="print a one-line human summary to stderr")
    parser.add_argument("--no-verify", action="store_true",
                        help="skip the independent post-solve verification pass")
    options = parser.parse_args(args)

    try:
        raw = sys.stdin.buffer.read() if options.input == "-" else _read_bytes(options.input)
    except AnonymizationError as exc:
        json.dump(failure_document(exc), sys.stderr, ensure_ascii=False)
        sys.stderr.write("\n")
        return 2

    assert isinstance(raw, bytes)
    try:
        dataset, solution, document = run_bytes(raw, verify=not options.no_verify)
    except ValidationError as exc:
        json.dump(failure_document(exc), sys.stderr, ensure_ascii=False)
        sys.stderr.write("\n")
        return 2
    except AnonymizationError as exc:
        json.dump(failure_document(exc), sys.stderr, ensure_ascii=False)
        sys.stderr.write("\n")
        return 3

    payload = encode_document(document)
    if options.result is not None:
        _atomic_write(options.result, payload)
    else:
        sys.stdout.buffer.write(payload)
        sys.stdout.buffer.flush()
    if options.summary:
        print(
            f"records={len(dataset.records)} "
            f"suppressed={solution.suppressed_count} "
            f"retained_loss={solution.retained_loss}",
            file=sys.stderr,
        )
    return 0
