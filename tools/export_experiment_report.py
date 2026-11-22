"""Export a saved MMDetection metric dictionary to the project Excel report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiment_reporting import write_experiment_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path, help="JSON object returned by runner.test()")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/reports/xray_experiment_results.xlsx"),
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    if not isinstance(metrics, dict):
        raise TypeError("metrics must be a JSON object.")
    report, raw = write_experiment_report(
        args.output,
        metrics,
        config=args.config,
        checkpoint=args.checkpoint,
        seed=args.seed,
        dataset=args.dataset,
        split=args.split,
        notes=args.notes,
    )
    print(f"Excel report: {report}")
    print(f"Raw metrics: {raw}")


if __name__ == "__main__":
    main()
