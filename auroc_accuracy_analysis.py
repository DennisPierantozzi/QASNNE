#!/usr/bin/env python3
"""
AUROC and accuracy sweep utilities for QA-SNNE uncertainty scores.

The script expects one or more per-example JSONL files (usually produced by
`compute_metrics.py` or `compute_vl_uncertainty.py`). Each JSON object should
contain:
  - `base_answer` (or a fallback such as `most_likely_answer` / `prediction`)
  - `reference` (ground-truth answer)
  - one or more numeric fields with uncertainty scores

By default we recompute ROUGE-L between `base_answer` and `reference` to serve
as the quality signal. Pass `--quality-field some_key` if the JSON already
stores a numeric quality score you want to use instead.

Example
-------
python auroc_accuracy_analysis.py \\
    --input qwen=outputs/qasnne_qwen.jsonl \\
    --input llama=outputs/qasnne_llama.jsonl \\
    --methods vl_uncertainty snne qa_snne \\
    --tau 0.2 \\
    --out-dir outputs/analysis/auroc

All outputs (per-model summaries, risk–coverage curves, overall aggregates) are
written under `--out-dir`, and key metrics are logged to stdout.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from rouge_score import rouge_scorer
from sklearn.metrics import roc_auc_score

LOGGER = logging.getLogger(__name__)
QUALITY_COL = "quality"

_ROUGE_SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------
def parse_model_inputs(raw_inputs: Iterable[str]) -> Dict[str, Path]:
    """Parse `name=path` entries passed via `--input`."""
    mapping: Dict[str, Path] = {}
    for raw in raw_inputs:
        if "=" not in raw:
            raise argparse.ArgumentTypeError(
                f"Expected MODEL=PATH format for --input, got '{raw}'."
            )
        name, path_str = raw.split("=", 1)
        name = name.strip()
        path = Path(path_str).expanduser().resolve()
        if not name:
            raise argparse.ArgumentTypeError(f"Empty model name in '{raw}'.")
        if not path.exists():
            raise argparse.ArgumentTypeError(f"Input path does not exist: {path}")
        if path.is_dir():
            raise argparse.ArgumentTypeError(f"Input path must be a file, not a dir: {path}")
        mapping[name] = path
    return mapping


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s | %(message)s",
    )


# ---------------------------------------------------------------------------
# Quality computation
# ---------------------------------------------------------------------------
def rougeL(pred: str, ref: str) -> float:
    pred = (pred or "").strip()
    ref = (ref or "").strip()
    if not pred or not ref:
        return np.nan
    try:
        return _ROUGE_SCORER.score(ref, pred)["rougeL"].fmeasure
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.warning("ROUGE-L failed for pred/ref pair: %s", exc)
        return np.nan


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_per_example_jsonl(
    path: Path,
    methods: Optional[List[str]],
    quality_mode: str,
) -> pd.DataFrame:
    """
    Load a per-example JSONL file and return a DataFrame with the requested
    uncertainty columns and a `quality` column.
    """
    rows: List[Dict[str, float]] = []
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
            idx = end_idx

            pred = (
                obj.get("base_answer")
                or obj.get("most_likely_answer")
                or obj.get("prediction")
                or obj.get("answer")
                or ""
            )
            ref = (
                obj.get("reference")
                or obj.get("ground_truth")
                or obj.get("gt_answer")
                or ""
            )

            if quality_mode.lower() == "rougel":
                quality = rougeL(pred, ref)
            else:
                val = obj.get(quality_mode, np.nan)
                quality = float(val) if isinstance(val, (int, float)) else np.nan

            row: Dict[str, float] = {QUALITY_COL: quality}

            if methods is None:
                for key, val in obj.items():
                    if key in {
                        "prediction",
                        "answer",
                        "base_answer",
                        "ground_truth",
                        "reference",
                        "gt_answer",
                        "question",
                        "generated_answers",
                        "semantic_ids",
                        "perturb",
                        "semantic",
                        "id",
                        "image_path",
                        "most_likely_answer",
                        "token_likelihoods",
                        "g_scores",
                        "w_visual",
                        "w_token",
                        "w_total",
                    }:
                        continue
                    if isinstance(val, (int, float)) and np.isfinite(val):
                        row[key] = float(val)
            else:
                for method in methods:
                    val = obj.get(method, np.nan)
                    row[method] = float(val) if isinstance(val, (int, float)) else np.nan

            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        LOGGER.warning("No valid rows loaded from %s", path)
        return df

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[QUALITY_COL]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# AUROC helpers
# ---------------------------------------------------------------------------
def auroc_at_tau(
    scores: np.ndarray,
    quality: np.ndarray,
    tau: float,
    min_samples: int = 30,
    verbose: bool = False,
) -> float:
    """AUROC for classifying quality < τ vs ≥ τ."""
    labels = (quality < tau).astype(int)
    mask = np.isfinite(scores) & np.isfinite(quality)

    if verbose:
        num_bad = int(labels[mask].sum())
        num_good = int(mask.sum() - num_bad)
        LOGGER.info(
            "      τ=%.3f: %d valid, %d bad, %d good", tau, int(mask.sum()), num_bad, num_good
        )

    if mask.sum() < min_samples:
        return np.nan

    if labels[mask].min() == labels[mask].max():
        return np.nan

    auc = roc_auc_score(labels[mask], scores[mask])
    return float(auc)


def orient_for_fixed_tau(
    scores: np.ndarray,
    quality: np.ndarray,
    tau: float,
    min_samples: int = 30,
) -> Tuple[np.ndarray, bool]:
    """Orient scores so that higher values imply more uncertainty."""
    auc = auroc_at_tau(scores, quality, tau, min_samples=min_samples, verbose=False)
    flip = bool(np.isfinite(auc) and auc < 0.5)
    oriented = -scores if flip else scores
    return oriented, flip


def classical_auroc(scores: np.ndarray, labels: np.ndarray, min_samples: int = 30) -> float:
    """Threshold-free AUROC against binary labels."""
    mask = np.isfinite(scores) & np.isfinite(labels)
    if mask.sum() < min_samples or np.unique(labels[mask]).size < 2:
        return np.nan
    auc = roc_auc_score(labels[mask], scores[mask])
    if np.isfinite(auc) and auc < 0.5:
        auc = roc_auc_score(labels[mask], -scores[mask])
    return float(auc)


def _threshold_metrics_from_arrays(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    direction: str,
) -> Dict[str, float]:
    preds = (scores > threshold) if direction == "greater" else (scores < threshold)
    preds = preds.astype(int)
    labels = labels.astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    accuracy = (tp + tn) / total if total > 0 else np.nan
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0
        else np.nan
    )
    balanced_accuracy = (
        0.5 * (recall + specificity)
        if np.isfinite(recall) and np.isfinite(specificity)
        else np.nan
    )
    return {
        "threshold": float(threshold),
        "direction": direction,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "accuracy": accuracy,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "total": total,
        "support_bad": int((labels == 1).sum()),
        "support_good": int((labels == 0).sum()),
    }


def evaluate_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    direction: str = "greater",
) -> Optional[Dict[str, float]]:
    mask = np.isfinite(scores) & np.isfinite(labels)
    if mask.sum() == 0:
        return None
    metrics = _threshold_metrics_from_arrays(scores[mask], labels[mask], threshold, direction)
    if metrics["total"] == 0:
        return None
    return metrics


_METRIC_NAME_MAP = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "f1": "f1",
    "precision": "precision",
    "recall": "recall",
}


def search_best_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    metric: str = "accuracy",
) -> Optional[Dict[str, float]]:
    metric = metric.lower()
    if metric not in _METRIC_NAME_MAP:
        raise ValueError(f"Unsupported metric '{metric}'.")

    mask = np.isfinite(scores) & np.isfinite(labels)
    if mask.sum() == 0 or np.unique(labels[mask]).size < 2:
        return None

    scores = scores[mask]
    labels = labels[mask]
    unique_scores = np.unique(scores)
    if unique_scores.size == 0:
        return None

    low = np.nextafter(unique_scores[0], -np.inf)
    high = np.nextafter(unique_scores[-1], np.inf)
    thresholds = (
        np.concatenate(
            ([low], (unique_scores[:-1] + unique_scores[1:]) / 2.0, [high])
        )
        if unique_scores.size > 1
        else np.array([low, high])
    )

    best: Optional[Dict[str, float]] = None
    best_value = -np.inf
    metric_key = _METRIC_NAME_MAP[metric]
    for direction in ("greater", "less"):
        for thr in thresholds:
            stats = _threshold_metrics_from_arrays(scores, labels, float(thr), direction)
            value = stats.get(metric_key, np.nan)
            if np.isfinite(value) and value > best_value + 1e-12:
                best_value = value
                best = stats.copy()
                best["metric_name"] = metric_key
                best["metric_value"] = value
    return best


# ---------------------------------------------------------------------------
# Risk–coverage
# ---------------------------------------------------------------------------
def risk_coverage_from_unc(
    df: pd.DataFrame,
    unc_col: str,
    quality_col: str = QUALITY_COL,
) -> pd.DataFrame:
    sub = df[[quality_col, unc_col]].dropna().copy()
    if sub.empty:
        return pd.DataFrame(columns=["coverage", "risk", "kept_n", "mean_quality"])
    sub = sub.sort_values(unc_col, ascending=True).reset_index(drop=True)
    q = sub[quality_col].to_numpy()
    n = len(q)
    cumsum = np.cumsum(q)
    kept = np.arange(1, n + 1)
    coverage = kept / n
    mean_quality = cumsum / kept
    risk = 1.0 - mean_quality
    return pd.DataFrame(
        {
            "coverage": coverage,
            "risk": risk,
            "kept_n": kept,
            "mean_quality": mean_quality,
        }
    )


def auarc_from_curve(df_rc: pd.DataFrame) -> float:
    if df_rc.empty:
        return np.nan
    x = df_rc["coverage"].to_numpy()
    y = df_rc["mean_quality"].to_numpy()
    order = np.argsort(x)
    x = np.concatenate([[0.0], x[order]])
    y = np.concatenate([[0.0], y[order]])
    return float(np.trapezoid(y, x))


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------
def run_model(
    model: str,
    df: pd.DataFrame,
    methods: List[str],
    out_dir: Path,
    tau: float,
    min_samples: int,
    uncertainty_threshold: float,
    threshold_direction: str,
    search_thresholds: bool,
    search_metric: str,
) -> pd.DataFrame:
    LOGGER.info("Processing %s (%d samples)", model, len(df))
    quality = df[QUALITY_COL].to_numpy(dtype=float)
    bad_labels = (quality < tau).astype(int)

    LOGGER.info(
        "  quality stats min=%.3f max=%.3f mean=%.3f median=%.3f",
        np.nanmin(quality),
        np.nanmax(quality),
        np.nanmean(quality),
        np.nanmedian(quality),
    )
    LOGGER.info(
        "  quality < τ (%.2f): %d / %d samples (%.1f%%)",
        tau,
        int(bad_labels.sum()),
        len(quality),
        100 * bad_labels.mean(),
    )

    per_model_dir = out_dir / "per_model"
    per_model_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, Union[str, float]]] = []
    rc_rows: Dict[str, pd.DataFrame] = {}

    for method in methods:
        if method not in df.columns:
            LOGGER.warning("  method '%s' not found; skipping.", method)
            continue

        scores_raw = df[method].to_numpy(dtype=float)
        finite_scores = scores_raw[np.isfinite(scores_raw)]
        if finite_scores.size == 0:
            LOGGER.warning("  method '%s': no finite scores; skipping.", method)
            continue

        LOGGER.info(
            "  → %s: %d finite scores (min=%.4f, max=%.4f, mean=%.4f, std=%.4f)",
            method,
            finite_scores.size,
            float(np.nanmin(finite_scores)),
            float(np.nanmax(finite_scores)),
            float(np.nanmean(finite_scores)),
            float(np.nanstd(finite_scores)),
        )

        scores_oriented, flipped = orient_for_fixed_tau(
            scores_raw, quality, tau=tau, min_samples=min_samples
        )
        if flipped:
            LOGGER.info("      orientation: flipped so higher ⇒ more uncertain at τ=%.3f", tau)

        auc_fixed = auroc_at_tau(
            scores_oriented,
            quality,
            tau=tau,
            min_samples=min_samples,
            verbose=True,
        )
        auc_classical = classical_auroc(scores_oriented, bad_labels, min_samples)

        thr_metrics = evaluate_threshold(
            scores_raw,
            bad_labels,
            threshold=uncertainty_threshold,
            direction=threshold_direction,
        )
        if thr_metrics:
            LOGGER.info(
                "      threshold (%s %.3f): accuracy=%.3f precision=%.3f recall=%.3f F1=%.3f",
                threshold_direction,
                uncertainty_threshold,
                thr_metrics["accuracy"],
                thr_metrics["precision"],
                thr_metrics["recall"],
                thr_metrics["f1"],
            )

        best_metrics = None
        if search_thresholds:
            best_metrics = search_best_threshold(scores_raw, bad_labels, metric=search_metric)
            if best_metrics:
                LOGGER.info(
                    "      best %s: %.3f at %.6f (%s) → acc=%.3f prec=%.3f rec=%.3f F1=%.3f",
                    search_metric,
                    best_metrics["metric_value"],
                    best_metrics["threshold"],
                    best_metrics["direction"],
                    best_metrics.get("accuracy", np.nan),
                    best_metrics.get("precision", np.nan),
                    best_metrics.get("recall", np.nan),
                    best_metrics.get("f1", np.nan),
                )

        df_rc = risk_coverage_from_unc(df.assign(**{method: scores_oriented}), method)
        rc_rows[method] = df_rc
        auarc = auarc_from_curve(df_rc)

        summary_rows.append(
            {
                "model": model,
                "method": method,
                "AUROC_at_tau": float(auc_fixed) if np.isfinite(auc_fixed) else np.nan,
                "AUC_classical": float(auc_classical) if np.isfinite(auc_classical) else np.nan,
                "AUARC": float(auarc) if np.isfinite(auarc) else np.nan,
                "threshold": uncertainty_threshold,
                "threshold_direction": threshold_direction,
                "thr_tp": thr_metrics["tp"] if thr_metrics else np.nan,
                "thr_fp": thr_metrics["fp"] if thr_metrics else np.nan,
                "thr_tn": thr_metrics["tn"] if thr_metrics else np.nan,
                "thr_fn": thr_metrics["fn"] if thr_metrics else np.nan,
                "thr_precision": thr_metrics["precision"] if thr_metrics else np.nan,
                "thr_recall": thr_metrics["recall"] if thr_metrics else np.nan,
                "thr_f1": thr_metrics["f1"] if thr_metrics else np.nan,
                "thr_accuracy": thr_metrics["accuracy"] if thr_metrics else np.nan,
                "best_threshold": best_metrics["threshold"] if best_metrics else np.nan,
                "best_direction": best_metrics["direction"] if best_metrics else "",
                "best_metric": best_metrics["metric_name"] if best_metrics else "",
                "best_metric_value": best_metrics["metric_value"] if best_metrics else np.nan,
                "best_precision": best_metrics.get("precision", np.nan) if best_metrics else np.nan,
                "best_recall": best_metrics.get("recall", np.nan) if best_metrics else np.nan,
                "best_f1": best_metrics.get("f1", np.nan) if best_metrics else np.nan,
                "best_accuracy": best_metrics.get("accuracy", np.nan) if best_metrics else np.nan,
                "n_samples": int(np.isfinite(scores_raw).sum()),
            }
        )

        (per_model_dir / f"{model}__performance_coverage__{method}.csv").write_text(
            df_rc.to_csv(index=False), encoding="utf-8"
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("AUROC_at_tau", ascending=False)
    (per_model_dir / f"{model}__summary.csv").write_text(
        summary_df.to_csv(index=False), encoding="utf-8"
    )

    curves_df = summary_df[["method", "AUROC_at_tau"]].copy()
    curves_df.insert(1, "tau", tau)
    (per_model_dir / f"{model}__auroc_curves.csv").write_text(
        curves_df.to_csv(index=False), encoding="utf-8"
    )

    LOGGER.info("Top methods for %s:\n%s", model, summary_df.head().to_string(index=False))
    return summary_df


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def aggregate_overall(summary_frames: List[pd.DataFrame], out_path: Path) -> None:
    if not summary_frames:
        LOGGER.info("No models processed; skipping overall summary.")
        return

    combined = pd.concat(summary_frames, ignore_index=True)
    overall = (
        combined.groupby("method", as_index=False)
        .agg(
            threshold=("threshold", "first"),
            threshold_direction=("threshold_direction", "first"),
            mean_AUROC_at_tau=("AUROC_at_tau", "mean"),
            std_AUROC_at_tau=("AUROC_at_tau", "std"),
            mean_AUC_classical=("AUC_classical", "mean"),
            std_AUC_classical=("AUC_classical", "std"),
            mean_AUARC=("AUARC", "mean"),
            std_AUARC=("AUARC", "std"),
            total_tp=("thr_tp", "sum"),
            total_fp=("thr_fp", "sum"),
            total_tn=("thr_tn", "sum"),
            total_fn=("thr_fn", "sum"),
            mean_thr_precision=("thr_precision", "mean"),
            mean_thr_recall=("thr_recall", "mean"),
            mean_thr_f1=("thr_f1", "mean"),
            mean_thr_accuracy=("thr_accuracy", "mean"),
            best_threshold=("best_threshold", "mean"),
            best_direction=("best_direction", "first"),
            best_metric=("best_metric", "first"),
            mean_best_metric_value=("best_metric_value", "mean"),
            std_best_metric_value=("best_metric_value", "std"),
            mean_best_precision=("best_precision", "mean"),
            mean_best_recall=("best_recall", "mean"),
            mean_best_f1=("best_f1", "mean"),
            mean_best_accuracy=("best_accuracy", "mean"),
            models=("model", "nunique"),
        )
        .sort_values("mean_AUROC_at_tau", ascending=False)
    )

    overall["agg_precision"] = np.where(
        overall["total_tp"] + overall["total_fp"] > 0,
        overall["total_tp"] / (overall["total_tp"] + overall["total_fp"]),
        np.nan,
    )
    overall["agg_recall"] = np.where(
        overall["total_tp"] + overall["total_fn"] > 0,
        overall["total_tp"] / (overall["total_tp"] + overall["total_fn"]),
        np.nan,
    )
    overall["agg_accuracy"] = np.where(
        (overall["total_tp"] + overall["total_fp"] + overall["total_tn"] + overall["total_fn"]) > 0,
        (overall["total_tp"] + overall["total_tn"])
        / (overall["total_tp"] + overall["total_fp"] + overall["total_tn"] + overall["total_fn"]),
        np.nan,
    )
    overall["agg_f1"] = np.where(
        np.isfinite(overall["agg_precision"])
        & np.isfinite(overall["agg_recall"])
        & ((overall["agg_precision"] + overall["agg_recall"]) > 0),
        2
        * overall["agg_precision"]
        * overall["agg_recall"]
        / (overall["agg_precision"] + overall["agg_recall"]),
        np.nan,
    )

    out_path.write_text(overall.to_csv(index=False), encoding="utf-8")
    LOGGER.info("Saved overall summary → %s", out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AUROC / accuracy analysis for uncertainty scores.")
    parser.add_argument(
        "--input",
        "-i",
        action="append",
        required=True,
        help="ModelName=/path/to/per_example.jsonl (repeatable).",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Explicit list of uncertainty fields to evaluate. Autodetected if omitted.",
    )
    parser.add_argument(
        "--method-labels",
        nargs="+",
        default=None,
        help="Optional display labels in the form field=label.",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.5,
        help="Quality threshold τ for AUROC computation.",
    )
    parser.add_argument("--min-samples", type=int, default=30, help="Minimum samples per score.")
    parser.add_argument(
        "--quality-field",
        default="rougeL",
        help="Use 'rougeL' to recompute ROUGE-L, or provide a numeric JSON key already present.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/analysis/auroc"),
        help="Directory for CSV outputs.",
    )
    parser.add_argument(
        "--unc-threshold",
        type=float,
        default=-3.5,
        help="Fixed uncertainty threshold applied to raw scores for diagnostics.",
    )
    parser.add_argument(
        "--threshold-direction",
        choices=["greater", "less"],
        default="greater",
        help="Interpretation of --unc-threshold (greater ⇒ score > thr flagged).",
    )
    parser.add_argument(
        "--search-thresholds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run a per-method threshold sweep to maximise --search-metric.",
    )
    parser.add_argument(
        "--search-metric",
        choices=sorted(_METRIC_NAME_MAP.keys()),
        default="accuracy",
        help="Metric maximised when sweeping thresholds.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    model_inputs = parse_model_inputs(args.input)
    LOGGER.info("Processing models: %s", ", ".join(model_inputs.keys()))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_model_frames: List[pd.DataFrame] = []

    method_labels: Dict[str, str] = {}
    if args.method_labels:
        for item in args.method_labels:
            if "=" not in item:
                raise ValueError(f"Invalid method-label entry '{item}'. Expected field=label.")
            field, label = item.split("=", 1)
            method_labels[field.strip()] = label.strip()

    methods = args.methods
    if methods is None:
        first_model = next(iter(model_inputs.values()))
        df_peek = load_per_example_jsonl(first_model, methods=None, quality_mode=args.quality_field)
        if df_peek.empty:
            raise RuntimeError(f"Could not auto-detect methods from {first_model}.")
        methods = [c for c in df_peek.columns if c != QUALITY_COL]
        LOGGER.info("Auto-detected methods: %s", ", ".join(methods))

    for model, path in model_inputs.items():
        df = load_per_example_jsonl(path, methods=methods, quality_mode=args.quality_field)
        if df.empty:
            LOGGER.warning("Skipping %s: no usable data.", model)
            continue
        df = df.rename(columns={m: method_labels.get(m, m) for m in methods})
        renamed_methods = [method_labels.get(m, m) for m in methods]
        summary_df = run_model(
            model=model,
            df=df,
            methods=renamed_methods,
            out_dir=args.out_dir,
            tau=args.tau,
            min_samples=args.min_samples,
            uncertainty_threshold=args.unc_threshold,
            threshold_direction=args.threshold_direction,
            search_thresholds=args.search_thresholds,
            search_metric=args.search_metric,
        )
        per_model_frames.append(summary_df)

    aggregate_overall(per_model_frames, args.out_dir / "overall_summary.csv")


if __name__ == "__main__":
    run()
