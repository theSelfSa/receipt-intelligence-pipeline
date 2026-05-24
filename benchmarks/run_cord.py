from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

DEFAULT_DATASET = "naver-clova-ix/cord-v2"
DEFAULT_COST_PER_TOKEN_USD = 0.000005
TOKENS_PER_CHAR = 0.25
_RAPIDFUZZ = None


@dataclass
class GroundTruthReceipt:
    merchant_name: str | None
    receipt_date: str | None
    total: float | None
    line_item_names: list[str]
    line_item_prices: list[float]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CORD benchmark against the receipt pipeline.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="HuggingFace dataset id.")
    parser.add_argument("--split", default="test", help="Dataset split to evaluate.")
    parser.add_argument("--limit", type=int, default=100, help="Maximum number of samples to process.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output file path. Defaults to benchmarks/results/cord_<split>_<timestamp>.json",
    )
    parser.add_argument(
        "--max-failure-records",
        type=int,
        default=50,
        help="Maximum number of per-sample failures to store in JSON.",
    )
    return parser.parse_args()


async def _run() -> Path:
    args = _parse_args()
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The 'datasets' package is required for CORD benchmarking. Install dependencies from requirements.txt."
        ) from exc
    try:
        from app.config import get_settings
        from app.services.confidence import compute_confidence
        from app.services.extraction import ReceiptExtractionService
        from app.services.ocr import OCRService
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Project dependencies are missing. Install dependencies from requirements.txt."
        ) from exc

    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required to run benchmark extraction.")

    dataset = load_dataset(args.dataset, split=args.split)
    total_available = len(dataset)
    max_samples = min(args.limit, total_available)
    samples = dataset.select(range(max_samples))

    ocr = OCRService(settings=settings)
    extractor = ReceiptExtractionService(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
    )

    merchant_exact_hits = 0
    merchant_fuzzy_hits = 0
    merchant_total = 0

    date_exact_hits = 0
    date_total = 0

    total_exact_hits = 0
    total_total = 0

    line_name_matched = 0
    line_name_total = 0

    line_price_matched = 0
    line_price_total = 0

    latency_values: list[float] = []
    calibration: dict[str, list[float]] = {"high": [], "medium": [], "low": []}

    estimated_tokens_total = 0
    failures: list[dict[str, Any]] = []
    success_count = 0

    for idx, sample in enumerate(samples):
        sample_id = sample.get("id", idx)
        try:
            image = _coerce_image(sample.get("image"))
            gt = _parse_ground_truth(sample.get("ground_truth"))

            started = perf_counter()
            ocr_result = await ocr.extract_text(image)
            extracted = await extractor.extract(ocr_result.raw_text)
            _, confidence_level = compute_confidence(extracted, ocr_result.mean_ocr_confidence)
            latency = perf_counter() - started

            latency_values.append(latency)
            estimated_tokens_total += int(len(ocr_result.raw_text) * TOKENS_PER_CHAR)

            (
                merchant_exact,
                merchant_fuzzy,
                date_exact,
                total_exact,
                line_name_hit_count,
                line_name_count,
                line_price_hit_count,
                line_price_count,
                receipt_accuracy,
            ) = _evaluate_sample(gt, extracted)

            if gt.merchant_name:
                merchant_total += 1
                merchant_exact_hits += int(merchant_exact)
                merchant_fuzzy_hits += int(merchant_fuzzy)
            if gt.receipt_date:
                date_total += 1
                date_exact_hits += int(date_exact)
            if gt.total is not None:
                total_total += 1
                total_exact_hits += int(total_exact)

            line_name_total += line_name_count
            line_name_matched += line_name_hit_count

            line_price_total += line_price_count
            line_price_matched += line_price_hit_count

            calibration[confidence_level.value].append(receipt_accuracy)
            success_count += 1
        except Exception as exc:
            if len(failures) < args.max_failure_records:
                failures.append(
                    {
                        "sample_index": idx,
                        "sample_id": sample_id,
                        "error": str(exc),
                    }
                )

    estimated_cost_total = estimated_tokens_total * DEFAULT_COST_PER_TOKEN_USD
    output_path = args.output or _default_output_path(args.split)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metrics_payload = {
        "merchant_name": {
            "exact_match": _safe_ratio(merchant_exact_hits, merchant_total),
            "fuzzy_match": _safe_ratio(merchant_fuzzy_hits, merchant_total),
        },
        "date": {
            "exact_match": _safe_ratio(date_exact_hits, date_total),
        },
        "total": {
            "exact_match": _safe_ratio(total_exact_hits, total_total),
            "tolerance": 0.01,
        },
        "line_item_name": {
            "fuzzy_match": _safe_ratio(line_name_matched, line_name_total),
            "fuzzy_threshold": 0.85,
        },
        "line_item_price": {
            "exact_match": _safe_ratio(line_price_matched, line_price_total),
            "tolerance": 0.01,
        },
        "latency": {
            "avg_seconds": mean(latency_values) if latency_values else None,
            "p95_seconds": _percentile(latency_values, 95),
        },
        "confidence_calibration": [
            {
                "confidence_level": level,
                "receipts": len(values),
                "accuracy": (sum(values) / len(values)) if values else None,
            }
            for level, values in calibration.items()
        ],
        "cost": {
            "estimated_tokens_total": estimated_tokens_total,
            "estimated_cost_total_usd": estimated_cost_total,
            "estimated_cost_per_receipt_usd": (
                estimated_cost_total / success_count if success_count > 0 else None
            ),
        },
    }

    payload = {
        "dataset": args.dataset,
        "split": args.split,
        "requested_samples": max_samples,
        "processed_samples": success_count + len(failures),
        "success_count": success_count,
        "failure_count": len(failures),
        "metrics": metrics_payload,
        "failures": failures,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Benchmark complete. Results written to {output_path}")
    return output_path


def _coerce_image(image: Any) -> Any:
    if image is not None and hasattr(image, "convert"):
        return image.convert("RGB")
    raise ValueError("Dataset sample is missing a valid PIL image.")


def _parse_ground_truth(raw_ground_truth: Any) -> GroundTruthReceipt:
    if isinstance(raw_ground_truth, str):
        try:
            parsed = json.loads(raw_ground_truth)
        except json.JSONDecodeError:
            parsed = {}
    elif isinstance(raw_ground_truth, dict):
        parsed = raw_ground_truth
    else:
        parsed = {}

    root = parsed.get("gt_parse", parsed) if isinstance(parsed, dict) else {}
    merchant_name = (
        _find_first_text_by_paths(root, [("meta", "company"), ("meta", "store"), ("meta", "merchant")])
        or _find_first_text_by_keys(root, {"company", "store", "merchant", "vendor"})
    )
    receipt_date = (
        _normalize_date(
            _find_first_text_by_paths(root, [("meta", "date"), ("date",), ("transaction_date",)])
            or _find_first_text_by_keys(root, {"date", "transaction_date", "purchase_date"})
        )
    )
    total = (
        _find_first_float_by_paths(root, [("total", "total_price"), ("total",), ("summary", "total")])
        if isinstance(root, dict)
        else None
    )
    if total is None:
        total = _find_first_float_by_keys(root, {"total_price", "total", "amount_total"})

    line_item_names, line_item_prices = _parse_line_items(root)
    return GroundTruthReceipt(
        merchant_name=merchant_name,
        receipt_date=receipt_date,
        total=total,
        line_item_names=line_item_names,
        line_item_prices=line_item_prices,
    )


def _parse_line_items(root: Any) -> tuple[list[str], list[float]]:
    if not isinstance(root, dict):
        return [], []

    candidate_lists: list[Any] = []
    for key in ("menu", "items", "line_items"):
        value = root.get(key)
        if isinstance(value, list):
            candidate_lists.append(value)

    line_item_names: list[str] = []
    line_item_prices: list[float] = []

    for entries in candidate_lists:
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            name = _find_first_text_by_keys(entry, {"nm", "name", "item_name", "menu_nm"})
            if name:
                line_item_names.append(name)
            price = _find_first_float_by_keys(entry, {"price", "unitprice", "amount", "total", "item_price"})
            if price is not None:
                line_item_prices.append(price)

    return line_item_names, line_item_prices


def _evaluate_sample(
    gt: GroundTruthReceipt,
    prediction: Any,
) -> tuple[bool, bool, bool, bool, int, int, int, int, float]:
    merchant_exact = False
    merchant_fuzzy = False
    if gt.merchant_name:
        merchant_exact = _normalized_text(gt.merchant_name) == _normalized_text(prediction.merchant_name)
        merchant_similarity = _fuzzy_similarity(gt.merchant_name, prediction.merchant_name)
        merchant_fuzzy = merchant_similarity >= 0.85

    date_exact = False
    predicted_date = prediction.date.isoformat() if prediction.date else None
    if gt.receipt_date:
        date_exact = gt.receipt_date == predicted_date

    total_exact = False
    if gt.total is not None:
        total_exact = abs(gt.total - float(prediction.total)) <= 0.01

    predicted_names = [item.name for item in prediction.line_items if item.name]
    predicted_prices = [float(item.total_price) for item in prediction.line_items]

    line_name_hits = _count_fuzzy_matches(gt.line_item_names, predicted_names, threshold=0.85)
    line_price_hits = _count_price_matches(gt.line_item_prices, predicted_prices, tolerance=0.01)

    components: list[float] = []
    if gt.merchant_name:
        components.append(1.0 if merchant_fuzzy else 0.0)
    if gt.receipt_date:
        components.append(1.0 if date_exact else 0.0)
    if gt.total is not None:
        components.append(1.0 if total_exact else 0.0)
    if gt.line_item_names:
        components.append(line_name_hits / len(gt.line_item_names))
    if gt.line_item_prices:
        components.append(line_price_hits / len(gt.line_item_prices))
    receipt_accuracy = sum(components) / len(components) if components else 0.0

    return (
        merchant_exact,
        merchant_fuzzy,
        date_exact,
        total_exact,
        line_name_hits,
        len(gt.line_item_names),
        line_price_hits,
        len(gt.line_item_prices),
        receipt_accuracy,
    )


def _count_fuzzy_matches(
    ground_truth_items: list[str],
    predicted_items: list[str],
    *,
    threshold: float,
) -> int:
    if not ground_truth_items or not predicted_items:
        return 0
    used_indices: set[int] = set()
    matches = 0

    for expected in ground_truth_items:
        best_score = -1.0
        best_idx = -1
        for idx, actual in enumerate(predicted_items):
            if idx in used_indices:
                continue
            score = _fuzzy_similarity(expected, actual)
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_score >= threshold and best_idx >= 0:
            matches += 1
            used_indices.add(best_idx)
    return matches


def _count_price_matches(
    ground_truth_prices: list[float],
    predicted_prices: list[float],
    *,
    tolerance: float,
) -> int:
    if not ground_truth_prices or not predicted_prices:
        return 0
    used_indices: set[int] = set()
    matches = 0

    for expected in ground_truth_prices:
        best_idx = -1
        best_diff = None
        for idx, actual in enumerate(predicted_prices):
            if idx in used_indices:
                continue
            diff = abs(float(expected) - float(actual))
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_idx = idx
        if best_idx >= 0 and best_diff is not None and best_diff <= tolerance:
            matches += 1
            used_indices.add(best_idx)
    return matches


def _normalized_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value.strip().lower())


def _fuzzy_similarity(expected: str, actual: str) -> float:
    if not expected or not actual:
        return 0.0
    return float(_rapidfuzz_fuzz().token_sort_ratio(expected, actual)) / 100.0


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _percentile(values: list[float], percentile: int) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = ((len(ordered) - 1) * percentile) / 100
    lower_idx = int(math.floor(position))
    upper_idx = int(math.ceil(position))
    if lower_idx == upper_idx:
        return ordered[lower_idx]
    weight = position - lower_idx
    return (ordered[lower_idx] * (1 - weight)) + (ordered[upper_idx] * weight)


def _extract_text_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        normalized = value.strip()
        return [normalized] if normalized else []
    if isinstance(value, (int, float)):
        return [str(value)]
    if isinstance(value, list):
        output: list[str] = []
        for item in value:
            output.extend(_extract_text_values(item))
        return output
    if isinstance(value, dict):
        output: list[str] = []
        if "text" in value:
            output.extend(_extract_text_values(value.get("text")))
        else:
            for nested in value.values():
                output.extend(_extract_text_values(nested))
        return output
    return []


def _extract_float_values(value: Any) -> list[float]:
    values = _extract_text_values(value)
    parsed_values: list[float] = []
    for item in values:
        parsed = _parse_float(item)
        if parsed is not None:
            parsed_values.append(parsed)
    return parsed_values


def _parse_float(text: str | None) -> float | None:
    if not text:
        return None
    cleaned = text.replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _normalize_date(value: str | None) -> str | None:
    if not value:
        return None
    candidate = value.strip()
    if not candidate:
        return None

    date_patterns = (
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%m/%d/%Y",
        "%m-%d-%Y",
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%y/%m/%d",
        "%m/%d/%y",
    )
    for pattern in date_patterns:
        try:
            return datetime.strptime(candidate, pattern).date().isoformat()
        except ValueError:
            continue

    try:
        return datetime.fromisoformat(candidate).date().isoformat()
    except ValueError:
        return None


def _find_first_text_by_paths(root: Any, paths: list[tuple[str, ...]]) -> str | None:
    for path in paths:
        current = root
        for key in path:
            if not isinstance(current, dict) or key not in current:
                current = None
                break
            current = current.get(key)
        if current is None:
            continue
        for text in _extract_text_values(current):
            if text:
                return text
    return None


def _find_first_float_by_paths(root: Any, paths: list[tuple[str, ...]]) -> float | None:
    for path in paths:
        current = root
        for key in path:
            if not isinstance(current, dict) or key not in current:
                current = None
                break
            current = current.get(key)
        if current is None:
            continue
        for value in _extract_float_values(current):
            return value
    return None


def _find_first_text_by_keys(root: Any, keys: set[str]) -> str | None:
    target = {key.lower() for key in keys}
    if isinstance(root, dict):
        for key, value in root.items():
            if key.lower() in target:
                texts = _extract_text_values(value)
                if texts:
                    return texts[0]
            nested = _find_first_text_by_keys(value, target)
            if nested:
                return nested
    elif isinstance(root, list):
        for item in root:
            nested = _find_first_text_by_keys(item, target)
            if nested:
                return nested
    return None


def _find_first_float_by_keys(root: Any, keys: set[str]) -> float | None:
    target = {key.lower() for key in keys}
    if isinstance(root, dict):
        for key, value in root.items():
            if key.lower() in target:
                numbers = _extract_float_values(value)
                if numbers:
                    return numbers[0]
            nested = _find_first_float_by_keys(value, target)
            if nested is not None:
                return nested
    elif isinstance(root, list):
        for item in root:
            nested = _find_first_float_by_keys(item, target)
            if nested is not None:
                return nested
    return None


def _default_output_path(split: str) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return Path("benchmarks") / "results" / f"cord_{split}_{timestamp}.json"


def _rapidfuzz_fuzz():
    global _RAPIDFUZZ
    if _RAPIDFUZZ is None:
        try:
            from rapidfuzz import fuzz as rapidfuzz_fuzz
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "The 'rapidfuzz' package is required for benchmarking. Install dependencies from requirements.txt."
            ) from exc
        _RAPIDFUZZ = rapidfuzz_fuzz
    return _RAPIDFUZZ


if __name__ == "__main__":
    asyncio.run(_run())
