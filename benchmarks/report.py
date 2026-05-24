from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate markdown benchmark report from JSON results.")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to benchmark JSON. Defaults to newest benchmarks/results/cord_*.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output markdown path. If omitted, report is printed to stdout only.",
    )
    return parser.parse_args()


def _latest_result_file() -> Path:
    results_dir = Path("benchmarks") / "results"
    candidates = sorted(results_dir.glob("cord_*.json"))
    if not candidates:
        raise FileNotFoundError("No benchmark result files found in benchmarks/results.")
    return candidates[-1]


def _load_result(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Benchmark result not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


def _usd(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"${value:.4f}"


def _build_markdown(result: dict[str, Any], source_path: Path) -> str:
    metrics = result.get("metrics", {})
    merchant_metrics = metrics.get("merchant_name", {})
    date_metrics = metrics.get("date", {})
    total_metrics = metrics.get("total", {})
    line_name_metrics = metrics.get("line_item_name", {})
    line_price_metrics = metrics.get("line_item_price", {})
    latency_metrics = metrics.get("latency", {})
    cost_metrics = metrics.get("cost", {})

    calibration_rows = metrics.get("confidence_calibration", [])
    calibration_lines = [
        "| Confidence Level | Receipts | Accuracy |",
        "|---|---:|---:|",
    ]
    for row in calibration_rows:
        level = str(row.get("confidence_level", "unknown")).upper()
        receipts = int(row.get("receipts", 0))
        accuracy = _pct(row.get("accuracy"))
        calibration_lines.append(f"| {level} | {receipts} | {accuracy} |")

    markdown = "\n".join(
        [
            "# Benchmark Results (CORD Dataset)",
            f"Source: `{source_path.as_posix()}`",
            f"- Dataset: `{result.get('dataset', 'unknown')}` (`{result.get('split', 'unknown')}` split)",
            f"- Requested samples: {result.get('requested_samples', 0)}",
            f"- Successful evaluations: {result.get('success_count', 0)}",
            f"- Failures: {result.get('failure_count', 0)}",
            "",
            "## Field Accuracy",
            "| Field | Exact Match | Fuzzy Match |",
            "|---|---:|---:|",
            f"| Merchant name | {_pct(merchant_metrics.get('exact_match'))} | {_pct(merchant_metrics.get('fuzzy_match'))} |",
            f"| Date | {_pct(date_metrics.get('exact_match'))} | - |",
            f"| Total | {_pct(total_metrics.get('exact_match'))} | - |",
            f"| Line item name | - | {_pct(line_name_metrics.get('fuzzy_match'))} |",
            f"| Line item price | {_pct(line_price_metrics.get('exact_match'))} | - |",
            "",
            "## Confidence Calibration",
            *calibration_lines,
            "",
            "## Throughput and Cost",
            f"- Average latency: `{latency_metrics.get('avg_seconds', 'N/A')}` seconds",
            f"- P95 latency: `{latency_metrics.get('p95_seconds', 'N/A')}` seconds",
            f"- Estimated tokens total: `{cost_metrics.get('estimated_tokens_total', 'N/A')}`",
            f"- Estimated cost total: `{_usd(cost_metrics.get('estimated_cost_total_usd'))}`",
            f"- Estimated cost per receipt: `{_usd(cost_metrics.get('estimated_cost_per_receipt_usd'))}`",
        ]
    )
    return markdown


def main() -> None:
    args = _parse_args()
    input_path = args.input or _latest_result_file()
    result = _load_result(input_path)
    markdown = _build_markdown(result, input_path)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
        print(f"Report written to {args.output}")
    print(markdown)


if __name__ == "__main__":
    main()
