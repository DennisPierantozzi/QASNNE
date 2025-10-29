#!/usr/bin/env python3
"""
Risk–coverage and removal-curve analysis utilities for QA-SNNE uncertainty scores.

Given a per-example JSONL file containing predictions, references, and one or
more uncertainty signals, this script:
  • computes quality metrics (ROUGE-L, optional BERTScore)
  • orients each uncertainty so that higher ⇒ lower quality
  • evaluates "remove top-uncertain x%" performance curves
  • estimates risk–coverage curves and PRR summary statistics
  • produces lightweight plots and CSV exports for downstream use

Example
-------
python coverage_curves_analysis.py \\
    --input outputs/qasnne_qwen.jsonl \\
    --method vl_uncertainty \\
    --method snne=SNNE \\
    --quality rougeL bertscore \\
    --out-dir outputs/analysis/coverage
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from bert_score import score as bertscore_score
from matplotlib.colors import to_hex
from rouge_score import rouge_scorer
from scipy.stats import spearmanr

LOGGER = logging.getLogger(__name__)

DEFAULT_REMOVAL = (0, 20, 40, 60, 80)
DEFAULT_PLOT_FORMATS = ("png",)
DPI = 300

QUALITY_LABELS = {
    "rougeL": "ROUGE-L",
    "bertscore": "BERTScore F1",
}

PREFERRED_METHOD_COLORS = {
    "dse": "#8B9DC3",
    "snne": "#d62728",
    "snne_bgeqa": "#17becf",
    "snne_nliqa": "#8c564b",
    "qa_snne": "#bcbd22",
}

_ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


# ---------------------------------------------------------------------------
# Argument parsing & logging
# ---------------------------------------------------------------------------
def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s | %(message)s",
    )


def parse_method_specs(specs: Optional[Iterable[str]]) -> Dict[str, str]:
    """
    Parse --method entries of the form field or field=Label.
    Returns a mapping from JSON field → display label.
    """
    mapping: Dict[str, str] = {}
    if not specs:
        return mapping
    for spec in specs:
        if "=" in spec:
            field, label = spec.split("=", 1)
            field = field.strip()
            label = label.strip()
        else:
            field, label = spec.strip(), spec.strip()
        if not field:
            raise argparse.ArgumentTypeError(f"Invalid method spec '{spec}'")
        mapping[field] = label or field
    return mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Risk/coverage analysis for uncertainty methods.")
    parser.add_argument("--input", type=Path, required=True, help="Per-example JSONL file.")
    parser.add_argument(
        "--method",
        "-m",
        action="append",
        help="Uncertainty field to evaluate (optional label via field=Label). Repeatable.",
    )
    parser.add_argument(
        "--quality",
        nargs="+",
        choices=["rougeL", "bertscore"],
        default=["rougeL", "bertscore"],
        help="Quality metrics to compute for coverage/performance curves.",
    )
    parser.add_argument(
        "--removal",
        type=int,
        nargs="+",
        default=list(DEFAULT_REMOVAL),
        help="Removal percentages for performance curves.",
    )
    parser.add_argument(
        "--plot-formats",
        nargs="+",
        default=list(DEFAULT_PLOT_FORMATS),
        help="Image formats for plots (use 'none' to disable plotting).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/analysis/coverage"),
        help="Directory to store CSVs and plots.",
    )
    parser.add_argument(
        "--bertscore-model",
        default="roberta-large",
        help="BERTScore model (used when 'bertscore' in --quality).",
    )
    parser.add_argument(
        "--bertscore-lang",
        default="en",
        help="Language code for BERTScore.",
    )
    parser.add_argument(
        "--bertscore-batch-size",
        type=int,
        default=64,
        help="Batch size for BERTScore computation.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Computation device for BERTScore.",
    )
    parser.add_argument("--min-samples", type=int, default=20, help="Minimum finite samples per method.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def iter_json_records(path: Path) -> Iterable[dict]:
    decoder = json.JSONDecoder()
    with open(path, "r", encoding="utf-8") as fp:
        content = fp.read().replace("\r\n", "\n").replace("\r", "\n")
    for line_no, raw_line in enumerate(content.split("\n"), 1):
        line = raw_line.strip()
        if not line:
            continue
        idx = 0
        while idx < len(line):
            while idx < len(line) and line[idx].isspace():
                idx += 1
            if idx >= len(line):
                break
            try:
                obj, end_idx = decoder.raw_decode(line, idx)
            except json.JSONDecodeError as exc:
                LOGGER.warning("Skipping malformed JSON at line %d: %s", line_no, exc)
                break
            yield obj
            idx = end_idx


def infer_numeric_methods(records: Iterable[dict]) -> List[str]:
    methods: List[str] = []
    for rec in records:
        for key, val in rec.items():
            if key in {
                "prediction",
                "base_answer",
                "most_likely_answer",
                "answer",
                "reference",
                "ground_truth",
                "gt_answer",
                "question",
                "generated_answers",
                "perturb",
                "semantic",
                "semantic_ids",
                "token_likelihoods",
                "g_scores",
                "w_visual",
                "w_token",
                "w_total",
                "image_path",
                "id",
            }:
                continue
            if isinstance(val, (int, float)):
                methods.append(key)
        if methods:
            break
    return sorted(set(methods))


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------
def compute_rouge(preds: List[str], refs: List[str]) -> np.ndarray:
    scores = []
    for pred, ref in zip(preds, refs):
        pred = (pred or "").strip()
        ref = (ref or "").strip()
        if not pred or not ref:
            scores.append(np.nan)
            continue
        try:
            scores.append(_ROUGE.score(ref, pred)["rougeL"].fmeasure)
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.debug("ROUGE failed for sample (%s)", exc)
            scores.append(np.nan)
    return np.asarray(scores, dtype=float)


def compute_bertscore(
    preds: List[str],
    refs: List[str],
    model: str,
    lang: str,
    batch_size: int,
    device: str,
) -> np.ndarray:
    LOGGER.info("Computing BERTScore with %s on %s", model, device)
    P, R, F = bertscore_score(
        preds,
        refs,
        model_type=model,
        lang=lang,
        batch_size=batch_size,
        rescale_with_baseline=True,
        device=device,
    )
    return F.cpu().numpy()


# ---------------------------------------------------------------------------
# Uncertainty orientation and metrics
# ---------------------------------------------------------------------------
def orient_uncertainty(u: np.ndarray, quality: np.ndarray, tag: str = "") -> Tuple[np.ndarray, bool]:
    mask = np.isfinite(u) & np.isfinite(quality)
    if mask.sum() < 20:
        return u, False
    rho, _ = spearmanr(u[mask], -quality[mask])
    flip = bool(np.isfinite(rho) and rho < 0)
    if tag:
        LOGGER.info("    %s Spearman(u,-q)=%.3f flip=%s", tag, rho, flip)
    return (-u) if flip else u, flip


def performance_after_removal(
    scores: np.ndarray,
    quality: np.ndarray,
    removal_percentages: Sequence[int],
) -> np.ndarray:
    scores = scores.astype(float).copy()
    quality = quality.astype(float)
    n = scores.size
    if n == 0:
        return np.zeros(len(removal_percentages), dtype=float)
    rng = np.random.default_rng(0)
    scores += 1e-12 * rng.standard_normal(n)
    order = np.argsort(-scores)
    q_sorted = quality[order]
    perf = []
    for pct in removal_percentages:
        k = int(round(n * (pct / 100.0)))
        remain = q_sorted[k:]
        perf.append(float(np.nanmean(remain)) if remain.size > 0 else np.nan)
    return np.asarray(perf, dtype=float)


def risk_coverage_curve(scores: np.ndarray, quality: np.ndarray) -> pd.DataFrame:
    mask = np.isfinite(scores) & np.isfinite(quality)
    if mask.sum() == 0:
        return pd.DataFrame(columns=["coverage", "performance"])
    scores = scores[mask]
    quality = quality[mask]
    order = np.argsort(scores)  # most certain first
    quality = quality[order]
    n = quality.size
    kept = np.arange(1, n + 1)
    coverage = kept / n
    mean_quality = np.cumsum(quality) / kept
    risk = 1.0 - mean_quality
    return pd.DataFrame(
        {
            "coverage": coverage,
            "performance": mean_quality,
            "risk": risk,
            "kept": kept,
        }
    )


def pr_curve_from_uncertainty(scores: np.ndarray, quality: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(scores) & np.isfinite(quality)
    scores = scores[mask]
    quality = quality[mask]
    if quality.size == 0:
        return np.array([]), np.array([])
    order = np.argsort(scores)
    quality = quality[order]
    n = quality.size
    coverage = np.arange(1, n + 1, dtype=float) / n
    perf = np.cumsum(quality) / np.arange(1, n + 1, dtype=float)
    return coverage, perf


def prr_from_scores(scores: np.ndarray, quality: np.ndarray) -> Dict[str, float]:
    cov_u, perf_u = pr_curve_from_uncertainty(scores, quality)
    if cov_u.size == 0:
        return {"prr": np.nan, "auc_unc": np.nan, "auc_oracle": np.nan, "auc_rand": np.nan}
    auc_unc = float(np.trapz(perf_u, cov_u))
    mask = np.isfinite(scores) & np.isfinite(quality)
    quality = quality[mask]
    order_oracle = np.argsort(-quality)
    quality_oracle = quality[order_oracle]
    cov_oracle = np.arange(1, quality_oracle.size + 1) / quality_oracle.size
    perf_oracle = np.cumsum(quality_oracle) / np.arange(1, quality_oracle.size + 1)
    auc_oracle = float(np.trapz(perf_oracle, cov_oracle))
    auc_rand = float(np.nanmean(quality))
    denom = auc_oracle - auc_rand
    prr = (auc_unc - auc_rand) / denom if denom > 0 else np.nan
    return {
        "prr": float(prr),
        "auc_unc": auc_unc,
        "auc_oracle": auc_oracle,
        "auc_rand": auc_rand,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def build_color_map(labels: List[str]) -> Dict[str, str]:
    color_map: Dict[str, str] = {}
    used = set()

    for name in labels:
        if name in PREFERRED_METHOD_COLORS and PREFERRED_METHOD_COLORS[name] not in used:
            color_map[name] = PREFERRED_METHOD_COLORS[name]
            used.add(PREFERRED_METHOD_COLORS[name])

    palette = []
    for cmap in ("tab20", "tab20b", "tab20c"):
        palette.extend(plt.get_cmap(cmap).colors)
    palette_hex = []
    seen = set()
    for rgba in palette:
        h = to_hex(rgba, keep_alpha=False)
        if h not in seen:
            palette_hex.append(h)
            seen.add(h)

    idx = 0
    for name in labels:
        if name in color_map:
            continue
        while idx < len(palette_hex) and palette_hex[idx] in used:
            idx += 1
        if idx >= len(palette_hex):
            palette_hex.extend(
                to_hex(plt.get_cmap("hsv")(x), keep_alpha=False) for x in np.linspace(0, 1, len(labels), endpoint=False)
            )
        color_map[name] = palette_hex[idx]
        used.add(palette_hex[idx])
        idx += 1
    return color_map


def plot_performance(
    quality_label: str,
    removal_percentages: Sequence[int],
    results: Dict[str, np.ndarray],
    color_map: Dict[str, str],
    out_path: Path,
    formats: Sequence[str],
):
    plt.figure(figsize=(8, 5))
    for label, perf in results.items():
        plt.plot(
            removal_percentages,
            perf,
            marker="o",
            linewidth=2,
            markersize=6,
            label=label,
            color=color_map[label],
        )
    plt.xlabel("Removal of highest-uncertainty samples (%)")
    plt.ylabel(f"Mean {quality_label}")
    plt.title(f"Performance after removal ({quality_label})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    base = out_path.with_suffix("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    saved = []
    for fmt in formats:
        target = base.with_suffix(f".{fmt}")
        plt.savefig(target, dpi=DPI)
        saved.append(target)
    plt.close()
    for target in saved:
        LOGGER.info("Saved plot → %s", target)


def plot_risk_coverage(
    quality_label: str,
    curves: Dict[str, pd.DataFrame],
    color_map: Dict[str, str],
    out_path: Path,
    formats: Sequence[str],
):
    plt.figure(figsize=(8, 5))
    for label, df in curves.items():
        if df.empty:
            continue
        plt.plot(
            df["coverage"],
            df["risk"],
            marker="o",
            linewidth=2,
            markersize=5,
            label=label,
            color=color_map[label],
        )
    plt.xlabel("Coverage (fraction kept)")
    plt.ylabel(f"Risk = 1 − mean {quality_label}")
    plt.title(f"Risk–coverage curve ({quality_label})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    base = out_path.with_suffix("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    saved = []
    for fmt in formats:
        target = base.with_suffix(f".{fmt}")
        plt.savefig(target, dpi=DPI)
        saved.append(target)
    plt.close()
    for target in saved:
        LOGGER.info("Saved plot → %s", target)


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    records = list(iter_json_records(args.input))
    if not records:
        raise RuntimeError(f"No rows found in {args.input}")

    method_map = parse_method_specs(args.method)
    if not method_map:
        inferred = infer_numeric_methods(records)
        if not inferred:
            raise RuntimeError("Unable to auto-detect uncertainty fields; please pass --method.")
        method_map = {m: m for m in inferred}
        LOGGER.info("Auto-detected methods: %s", ", ".join(method_map.keys()))

    preds, refs = [], []
    field_values: Dict[str, List[float]] = {field: [] for field in method_map}
    for rec in records:
        pred = (
            rec.get("base_answer")
            or rec.get("most_likely_answer")
            or rec.get("prediction")
            or rec.get("answer")
            or ""
        )
        ref = rec.get("reference") or rec.get("ground_truth") or rec.get("gt_answer") or ""
        preds.append(str(pred))
        refs.append(str(ref))
        for field in method_map:
            val = rec.get(field, np.nan)
            field_values[field].append(float(val) if isinstance(val, (int, float)) else np.nan)

    quality_arrays: Dict[str, np.ndarray] = {}
    if "rougeL" in args.quality:
        quality_arrays["rougeL"] = compute_rouge(preds, refs)
    if "bertscore" in args.quality:
        device = (
            "cuda"
            if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()))
            else "cpu"
        )
        quality_arrays["bertscore"] = compute_bertscore(
            preds,
            refs,
            model=args.bertscore_model,
            lang=args.bertscore_lang,
            batch_size=args.bertscore_batch_size,
            device=device,
        )

    methods_in_order = list(method_map.values())
    color_map = build_color_map(methods_in_order)

    summary_rows: List[Dict[str, float]] = []
    plot_formats = tuple(
        fmt.lower().lstrip(".") for fmt in args.plot_formats if fmt and fmt.lower() != "none"
    )

    for quality_name, quality_values in quality_arrays.items():
        quality_label = QUALITY_LABELS.get(quality_name, quality_name)
        LOGGER.info("Quality metric: %s", quality_label)

        perf_results: Dict[str, np.ndarray] = {}
        rc_curves: Dict[str, pd.DataFrame] = {}

        for field, label in method_map.items():
            scores = np.asarray(field_values[field], dtype=float)
            mask = np.isfinite(scores) & np.isfinite(quality_values)
            if mask.sum() < args.min_samples:
                LOGGER.warning("  %s: insufficient samples (%d); skipping.", label, mask.sum())
                continue

            oriented, flipped = orient_uncertainty(scores, quality_values, tag=f"{label} ({quality_label})")
            perf = performance_after_removal(oriented, quality_values, args.removal)
            curve = risk_coverage_curve(oriented, quality_values)
            prr_stats = prr_from_scores(oriented, quality_values)
            rho, _ = spearmanr(oriented[mask], quality_values[mask])

            perf_results[label] = perf
            rc_curves[label] = curve

            quality_dir = QUALITY_LABELS.get(quality_name, quality_name).replace(" ", "_")
            out_base = args.out_dir / quality_dir / label.replace(" ", "_")
            out_base.parent.mkdir(parents=True, exist_ok=True)

            removal_df = pd.DataFrame(
                {"removal_pct": args.removal, f"mean_{quality_name}": perf}
            )
            removal_df.to_csv(out_base.with_suffix(".removal.csv"), index=False)
            curve.to_csv(out_base.with_suffix(".risk_coverage.csv"), index=False)

            summary_rows.append(
                {
                    "method": label,
                    "quality_metric": quality_label,
                    "spearman_u_vs_quality": float(rho) if np.isfinite(rho) else np.nan,
                    "flipped": flipped,
                    "prr": prr_stats["prr"],
                    "auc_unc": prr_stats["auc_unc"],
                    "auc_oracle": prr_stats["auc_oracle"],
                    "auc_rand": prr_stats["auc_rand"],
                    "baseline_quality": float(np.nanmean(quality_values)),
                    "num_samples": int(mask.sum()),
                }
            )

        if perf_results and plot_formats:
            plot_performance(
                quality_name,
                quality_label,
                args.removal,
                perf_results,
                color_map,
                args.out_dir / f"performance_after_removal_{quality_label.replace(' ', '_')}",
                plot_formats,
            )
            plot_risk_coverage(
                quality_name,
                rc_curves,
                color_map,
                args.out_dir / f"risk_coverage_{quality_label.replace(' ', '_')}",
                plot_formats,
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = args.out_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    LOGGER.info("Saved summary → %s", summary_path)
    if not summary_df.empty:
        LOGGER.info("\n%s", summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
