#!/usr/bin/env python3
"""
Utility script to compute corpus-level text metrics (ROUGE / BLEU / METEOR)
for one or more JSON/JSONL prediction files.

Each input file is expected to contain per-example dictionaries with at least
one of the following prediction/ground-truth keys:
  - predictions:  `most_likely_answer`, `prediction_text`, `pred`, `answer`
  - references:   `reference`, `gold`, `label`, `answers`

Usage examples
--------------
Evaluate a single JSONL file and write metrics next to it:
    python compute_utility.py --input outputs/qasnne_qwen.jsonl --outdir outputs/analysis/text-metrics

Evaluate every JSONL under a directory (recursive) and compute ROUGE only:
    python compute_utility.py --input outputs/*.jsonl --metric rouge

Include the legacy hard-coded study files:
    python compute_utility.py --use-default-files

Outputs
-------
For each processed file we emit a `<stem>_metrics.json` containing the metrics,
and the aggregated results are stored in `<outdir>/<summary_name>`.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import pandas as pd
import evaluate

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_FILES: Sequence[str] = (
    "/SAN/medic/Cholec/thesis/generations/out_template_validation/peft/generation_surgical_out_template_dataset.jsonl",
    "/SAN/medic/Cholec/thesis/generations/out_template_validation/peft/generation_pitlora_out_template_dataset.jsonl",
    "/SAN/medic/Cholec/thesis/generations/in_template_validation/peft/generation_surgical_in_template_dataset.jsonl",
    "/SAN/medic/Cholec/thesis/generations/in_template_validation/peft/generation_pitlora_in_template_dataset.jsonl",
)

AVAILABLE_METRICS = ("rouge", "bleu", "meteor")

LOGGER = logging.getLogger("compute_utility")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def read_json_records(path: Path) -> List[Dict[str, Any]]:
    """Load JSON or JSONL into a list of dictionaries."""
    with path.open("r", encoding="utf-8") as handle:
        text = handle.read().strip()
    if not text:
        return []

    # Try JSONL first.
    lines = [ln for ln in text.splitlines() if ln.strip()]
    try:
        return [json.loads(ln) for ln in lines]
    except json.JSONDecodeError:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            return [obj]
    raise ValueError(f"Unsupported JSON structure in {path}")


def extract_preds_refs(records: Iterable[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    """Extract paired (prediction, reference) strings with common fallbacks."""
    preds: List[str] = []
    refs: List[str] = []
    for row in records:
        pred = (
            row.get("most_likely_answer")
            or row.get("prediction_text")
            or row.get("pred")
            or row.get("answer")
        )
        ref = (
            row.get("reference")
            or row.get("gold")
            or row.get("label")
            or row.get("answers")
        )

        # Unwrap common containers used in QA benchmarks.
        if isinstance(ref, list):
            if ref and isinstance(ref[0], str):
                ref = ref[0]
            elif ref and isinstance(ref[0], dict) and "text" in ref[0]:
                ref = ref[0]["text"]
        if isinstance(ref, dict) and "text" in ref:
            ref = ref["text"]

        if pred is None or ref is None:
            continue

        pred_str = str(pred).strip()
        ref_str = str(ref).strip()
        if pred_str and ref_str:
            preds.append(pred_str)
            refs.append(ref_str)

    return preds, refs


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------
def ensure_cache_dir(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir.parent))
    os.environ["HF_EVALUATE_CACHE"] = str(cache_dir)
    return cache_dir


def compute_metrics(
    preds: Sequence[str],
    refs: Sequence[str],
    metrics: Sequence[str],
    cache_dir: Path,
) -> Dict[str, Any]:
    """Compute the requested metrics using the Hugging Face `evaluate` package."""
    results: Dict[str, Any] = {"count": len(preds)}

    for metric in metrics:
        try:
            metric_module = evaluate.load(metric, cache_dir=str(cache_dir))
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning("Skipping metric '%s' (load failed: %s)", metric, exc)
            continue

        if metric == "rouge":
            metric_out = metric_module.compute(
                predictions=list(preds),
                references=list(refs),
                use_stemmer=True,
            )
            results.update(
                {
                    "rouge1": metric_out.get("rouge1"),
                    "rouge2": metric_out.get("rouge2"),
                    "rougeL": metric_out.get("rougeL"),
                    "rougeLsum": metric_out.get("rougeLsum"),
                }
            )

        elif metric == "bleu":
            metric_out = metric_module.compute(
                predictions=list(preds),
                references=[[r] for r in refs],
            )
            results.update(
                {
                    "bleu": metric_out.get("bleu"),
                    "brevity_penalty": metric_out.get("brevity_penalty"),
                    "precisions": metric_out.get("precisions"),
                    "length_ratio": metric_out.get("length_ratio"),
                }
            )

        elif metric == "meteor":
            metric_out = metric_module.compute(
                predictions=list(preds),
                references=list(refs),
            )
            results["meteor"] = metric_out.get("meteor")

        else:
            metric_out = metric_module.compute(
                predictions=list(preds),
                references=list(refs),
            )
            for key, value in metric_out.items():
                results[f"{metric}_{key}"] = value

    return results


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------
def detect_wildcard(pattern: str) -> bool:
    return any(ch in pattern for ch in "*?[]")


def gather_input_paths(
    inputs: Sequence[str] | None,
    include_defaults: bool,
    recursive: bool,
) -> List[Path]:
    paths: List[Path] = []

    def add_path(p: Path) -> None:
        if p.exists() and p.is_file():
            paths.append(p.resolve())
        else:
            LOGGER.warning("Skipping missing file: %s", p)

    if include_defaults:
        for entry in DEFAULT_FILES:
            add_path(Path(entry).expanduser())

    for entry in inputs or []:
        entry = entry.strip()
        if not entry:
            continue
        expanded = Path(entry).expanduser()

        if detect_wildcard(entry):
            base = expanded.parent if expanded.parent != Path("") else Path(".")
            pattern = expanded.name
            iterator = base.rglob(pattern) if recursive else base.glob(pattern)
            for match in iterator:
                if match.is_file():
                    paths.append(match.resolve())
            continue

        if expanded.is_dir():
            iterator = expanded.rglob("*.jsonl") if recursive else expanded.glob("*.jsonl")
            for match in iterator:
                if match.is_file():
                    paths.append(match.resolve())
            continue

        add_path(expanded)

    # Deduplicate while preserving order.
    seen = set()
    unique_paths = []
    for pth in paths:
        if pth not in seen:
            unique_paths.append(pth)
            seen.add(pth)

    return unique_paths


def write_metrics_json(path: Path, metrics: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute ROUGE/BLEU/METEOR for JSON or JSONL prediction files."
    )
    parser.add_argument(
        "--input",
        "-i",
        action="append",
        help="File path, directory, or glob pattern to include. Repeatable.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse directories (and glob patterns) when collecting inputs.",
    )
    parser.add_argument(
        "--use-default-files",
        action="store_true",
        help="Include the legacy hard-coded study files.",
    )
    parser.add_argument(
        "--metric",
        nargs="+",
        choices=AVAILABLE_METRICS,
        default=list(AVAILABLE_METRICS),
        help="Metrics to compute (default: all).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("outputs/analysis/text-metrics"),
        help="Directory where per-file metrics and the summary CSV will be stored.",
    )
    parser.add_argument(
        "--summary-name",
        default="metrics_summary.csv",
        help="Name of the summary CSV (relative to --outdir).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Writable cache directory for Hugging Face evaluate metrics.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    inputs = gather_input_paths(args.input, include_defaults=args.use_default_files, recursive=args.recursive)
    if not inputs:
        raise SystemExit("No valid input files found. Pass --input or --use-default-files.")

    outdir: Path = args.outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    cache_dir = args.cache_dir or Path.home() / ".cache" / "huggingface" / "metrics"
    cache_dir = ensure_cache_dir(Path(cache_dir).expanduser())

    summary_rows: List[Dict[str, Any]] = []
    metrics_to_compute = list(dict.fromkeys(args.metric))  # preserve order/remove duplicates

    LOGGER.info("Processing %d file(s)", len(inputs))
    for path in inputs:
        LOGGER.info("→ %s", path)
        try:
            records = read_json_records(path)
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.error("  Failed to read %s: %s", path, exc)
            continue

        preds, refs = extract_preds_refs(records)
        if not preds:
            LOGGER.warning("  No usable prediction/reference pairs found; skipping.")
            continue

        metrics = compute_metrics(preds, refs, metrics_to_compute, cache_dir)
        metrics_with_path = {"file": str(path), **metrics}

        stem = path.stem
        metrics_filename = f"{stem}_metrics.json" if not stem.endswith("_metrics") else f"{stem}.json"
        metrics_path = outdir / metrics_filename
        write_metrics_json(metrics_path, metrics_with_path)
        LOGGER.info("  metrics → %s", metrics_path)

        summary_rows.append(
            {
                "file": str(path),
                "count": metrics.get("count"),
                "rouge1": metrics.get("rouge1"),
                "rouge2": metrics.get("rouge2"),
                "rougeL": metrics.get("rougeL"),
                "bleu": metrics.get("bleu"),
                "meteor": metrics.get("meteor"),
            }
        )

    if not summary_rows:
        LOGGER.warning("No metrics produced; exiting without summary CSV.")
        return

    summary_path = outdir / args.summary_name
    if summary_path.exists() and summary_path.is_dir():
        raise RuntimeError(
            f"Summary path {summary_path} is a directory. Choose a different --summary-name."
        )

    df = pd.DataFrame(summary_rows).sort_values("file")
    df.to_csv(summary_path, index=False)
    LOGGER.info("Summary saved to %s", summary_path)


def main() -> None:
    parser = build_parser()
    run(parser.parse_args())


if __name__ == "__main__":
    main()
