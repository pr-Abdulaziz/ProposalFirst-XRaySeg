#!/usr/bin/env python3
"""Compatibility guard for the former ambiguous experiment launcher."""

raise SystemExit(
    "Choose an explicit dataset runtime: "
    "scripts/run_pixray_experiments.py or "
    "scripts/run_pidray30_experiments.py"
)
