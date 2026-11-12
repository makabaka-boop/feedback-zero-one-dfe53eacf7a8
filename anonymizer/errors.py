"""Error types and JSON path helpers.

All user-facing failures carry a JSONPath-style ``path`` (for humans) and an
RFC 6901 ``json_pointer`` (for machines).  Validation never partially applies
an input: the first invalid field/tree/reference rejects the whole job.
"""

from __future__ import annotations

import json
from typing import Iterable, Sequence, Union

PathPart = Union[str, int]


def _human_path(parts: Sequence[PathPart]) -> str:
    out = "$"
    for part in parts:
        if isinstance(part, int):
            out += f"[{part}]"
        elif part.isidentifier():
            out += f".{part}"
        else:
            out += "[" + json.dumps(part, ensure_ascii=True) + "]"
    return out


def _json_pointer(parts: Sequence[PathPart]) -> str:
    encoded = []
    for part in parts:
        token = str(part).replace("~", "~0").replace("/", "~1")
        encoded.append(token)
    return "" if not encoded else "/" + "/".join(encoded)


class AnonymizationError(Exception):
    """Base class for expected, non-bug failures."""


class ValidationError(AnonymizationError):
    """The submitted JSON document is not valid for this job."""

    def __init__(
        self,
        code: str,
        parts: Iterable[PathPart],
        message: str,
        *,
        detail: str | None = None,
    ) -> None:
        self.code = code
        self.parts = tuple(parts)
        self.path = _human_path(self.parts)
        self.json_pointer = _json_pointer(self.parts)
        self.message = message
        self.detail = detail
        super().__init__(f"{code} at {self.path}: {message}")

    def to_dict(self) -> dict:
        result = {
            "code": self.code,
            "path": self.path,
            "json_pointer": self.json_pointer,
            "message": self.message,
        }
        if self.detail is not None:
            result["detail"] = self.detail
        return result


class InputJSONError(ValidationError):
    """Raised when the byte stream is not valid UTF-8 or valid JSON."""


def fail(
    parts: Iterable[PathPart],
    code: str,
    message: str,
    *,
    detail: str | None = None,
) -> None:
    """Raise a :class:`ValidationError` rooted at ``parts``."""

    raise ValidationError(code, parts, message, detail=detail)
