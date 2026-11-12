#!/usr/bin/env python3
"""Convenience entry point: ``python3 run.py INPUT.json [options]``.

Unlike ``python3 -m anonymizer`` this works under isolated mode (``-I``) and
from any working directory, because it adds its own directory to ``sys.path``
explicitly instead of relying on the interpreter's script-directory rule.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from anonymizer import run_cli  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_cli(sys.argv[1:]))
