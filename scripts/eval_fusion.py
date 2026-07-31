#!/usr/bin/env python3
"""Compatibility wrapper for the canonical Node evaluator."""

from __future__ import annotations

import subprocess
import sys

result = subprocess.run(
    ["node", "scripts/eval_fusion.mjs"],
    check=False,
)
sys.exit(result.returncode)
