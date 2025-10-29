#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_combined_snne.py

Adds two QA similarity alternatives:
  (1) BGE cross-encoder reranker scores
  (2) Bi-directional NLI compatibility (entailment↑, contradiction↓)
Keeps the original QA-aware gating.

Usage (same as before):
  python compute_combined_snne.py --input_file data.jsonl --mode text
  python compute_combined_snne.py --input_file data.jsonl --mode visual
  python compute_combined_snne.py --input_file data.jsonl --mode both

New relevant flags:
  --bge_model_name BAAI/bge-reranker-large
  --nli_model_name microsoft/deberta-large-mnli
  --beta 10.0                  # softmax sharpness for gating
  --nli_alpha 1.0 --nli_lambda 0.5  # weight ent/contra in NLI score
  --fusion_weight 1.0          # BGE+NLI mixture weight (equal by default)
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import evaluate
from sentence_transformers import SentenceTransformer

from transformers import AutoTokenizer, AutoModelForSequenceClassification 
from utils.semantic_nn_entropy import snne, lexical_similarity_matrix
from uncertainty.semantic_entropy import (
    cluster_assignment_entropy,
    get_semantic_ids,
    predictive_entropy,
    EntailmentDeberta
)

# --------------------------- IO Helpers ---------------------------

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def _to_jsonable(obj: Any) -> Any:
    import numpy as _np
    import torch as _torch
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (_np.integer, _np.floating)):
        return obj.item()
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    if isinstance(obj, _torch.Tensor):
        return obj.detach().cpu().tolist() if obj.ndim > 0 else obj.detach().cpu().item()
    return obj

def save_results(results: List[Dict[str, Any]], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(_to_jsonable(r), ensure_ascii=False) + "\n")
    print(f"Saved processed results to {output_path}")



@torch.inference_mode()
def encode_sentence_embeddings(model: SentenceTransformer, texts: List[str], device: str) -> torch.Tensor:
    if not texts:
        return torch.empty((0, model.get_sentence_embedding_dimension()), device=device)
    emb = model.encode(
        texts,
        convert_to_tensor=True,
        device=device,
        normalize_embeddings=True,
    )
    return emb.to(device)
# --------------------------- NEW: QA similarity backends ---------------------------

@torch.inference_mode()
def score_bge_reranker(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    question: str,
    answers: List[str],
    device: str,
    max_length: int = 512,
) -> torch.Tensor:
    """
    Returns a score vector s[i] = BGE cross-encoder logit for (q, a_i).
    Per BGE card, logits are unbounded relevance scores. (Higher = more relevant.)
    """
    pairs = [[question, a] for a in answers]
    inputs = tokenizer(pairs, padding=True, truncation=True,
                       max_length=max_length, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    logits = model(**inputs, return_dict=True).logits.view(-1).float()  # shape [N]
    return logits

def _index_of_label(id2label: Dict[int, str], name: str) -> Optional[int]:
    name = name.lower()
    for idx, lab in id2label.items():
        if lab.lower().startswith(name):
            return int(idx)
    return None

@torch.inference_mode()
def score_nli_bidirectional(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    question: str,
    answers: List[str],
    device: str,
    max_length: int = 256,
    alpha: float = 1.0,
    lambda_contra: float = 0.5,
) -> torch.Tensor:
    """
    Bi-directional NLI compatibility:
      s_nli = alpha * (ent(q->a) + ent(a->q)) - lambda * (contra(q->a) + contra(a->q))
    Uses raw logits (pre-softmax). Robustly maps label ids using model.config.id2label.
    """
    id2label = getattr(model.config, "id2label", {0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"})
    ent_idx = _index_of_label(id2label, "entail")
    con_idx = _index_of_label(id2label, "contrad")
    if ent_idx is None or con_idx is None:
        # Fallback to common MNLI order [contradiction, neutral, entailment]
        ent_idx, con_idx = 2, 0

    def _pair_logits(prem, hypo):
        enc = tokenizer(
            prem, hypo, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc, return_dict=True).logits  # [B, 3]
        return out

    # Batch q->a
    q_list = [question] * len(answers)
    a_list = answers
    logits_q2a = _pair_logits(q_list, a_list)  # [N, 3]
    logits_a2q = _pair_logits(a_list, q_list)  # [N, 3]

    ent = logits_q2a[:, ent_idx] + logits_a2q[:, ent_idx]
    contra = logits_q2a[:, con_idx] + logits_a2q[:, con_idx]
    s_nli = alpha * ent - lambda_contra * contra
    return s_nli.float()  # shape [N]

def softmax_weights(scores: torch.Tensor, beta: float = 10.0) -> torch.Tensor:
    """Turn a score vector into relevance weights r_i via softmax(beta * score)."""
    x = beta * (scores - scores.max())  # stability
    w = torch.softmax(x, dim=-1)
    return w

def apply_gate_to_matrix(S: np.ndarray, r: torch.Tensor) -> np.ndarray:
    """Return S' = diag(r) * S * diag(r)."""
    r_np = r.detach().cpu().numpy().astype(np.float32)
    S_out = (r_np[:, None] * S * r_np[None, :]).astype(np.float32)
    return S_out

# --------------------------- Main Function ---------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute metrics from a JSONL file"
    )
    # I/O
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Text-only SNNE params
    parser.add_argument("--similarity", type=str, default="rouge",
                        choices=["rouge", "meteor", "embedding"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--include_self", action="store_false")
    parser.add_argument("--st_model_name", type=str, default="all-MiniLM-L6-v2")

    # Visual-linguistic params (unchanged)
    parser.add_argument("--hf_hub_id", type=str,
                        default="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    parser.add_argument("--precision", type=str, default="fp16",
                        choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--k_visual", type=float, default=4.0)
    parser.add_argument("--eta_token", type=float, default=1.0)
    parser.add_argument("--grid", type=int, default=7)
    parser.add_argument("--num_scales", type=int, default=3)
    parser.add_argument("--crop_batch", type=int, default=64)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--tau_crop", type=float, default=10.0)

    # NEW: QA backends
    parser.add_argument("--bge_model_name", type=str, default="BAAI/bge-reranker-large",
                        help="Cross-encoder reranker for q→a scoring")
    parser.add_argument("--nli_model_name", type=str, default="microsoft/deberta-large-mnli",
                        help="NLI model for bidirectional compatibility (MNLI)")
    parser.add_argument("--beta", type=float, default=10.0, help="Softmax sharpness for gating")
    parser.add_argument("--nli_alpha", type=float, default=1.0, help="Weight of entailment logits")
    parser.add_argument("--nli_lambda", type=float, default=0.5, help="Weight of contradiction logits")
    parser.add_argument("--fusion_weight", type=float, default=1.0,
                        help="Relative weight of NLI vs BGE in fused gate; r ∝ softmax(beta*(bge + fusion_weight*nli))")

    parser.add_argument("--csls_k", type=int, default=5,
                        help="Neighborhood size for CSLS hubness-reduced QA alignment")

    args = parser.parse_args()

    # Output dir
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Data
    print(f"Loading data from {args.input_file}...")
    data = load_jsonl(args.input_file)
    if args.max_samples:
        data = data[:args.max_samples]
    print(f"Processing {len(data)} samples")

    
    print("Loading PubMedBERT text encoder (pritamdeka/S-PubMedBert-MS-MARCO)...")
    text_encoder = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO", device=args.device)
    text_encoder = text_encoder.to(args.device)

    rouge = evaluate.load('rouge', keep_in_memory=True)

    # NEW: Load BGE reranker & NLI once
    print(f"Loading BGE reranker: {args.bge_model_name}")
    bge_tok = AutoTokenizer.from_pretrained(args.bge_model_name)
    bge_model = AutoModelForSequenceClassification.from_pretrained(args.bge_model_name).to(args.device).eval()

    print(f"Loading NLI model: {args.nli_model_name}")
    nli_tok = AutoTokenizer.from_pretrained(args.nli_model_name)
    nli_model = AutoModelForSequenceClassification.from_pretrained(args.nli_model_name).to(args.device).eval()

    deberta_model = EntailmentDeberta()

    results = []
    for sample in tqdm(data, desc="Computing SNNE and VL_SNNE with QA gates"):
        question = sample.get("question", "")
        gen_answers = sample.get("generated_answers", []) or []
        prediction = sample.get("most_likely_answer", "")
        ground_truth = sample.get("reference", "")
        image_path = sample.get("image_path", "")

        result = {
            "image_path": image_path,
            "question": question,
            "prediction": prediction,
            "ground_truth": ground_truth,
            "snne": None,
            "similarity": args.similarity,
            "tau": args.temperature,
            "exclude_diagonal": not args.include_self,
        }

        semantic_ids = get_semantic_ids(gen_answers, deberta_model)
        if not isinstance(semantic_ids, list):
            semantic_ids = list(semantic_ids)

        # 3) Discrete semantic entropy
        dse = float(cluster_assignment_entropy(semantic_ids)) if len(semantic_ids) > 0 else None

        # ---------- Base text similarity & SNNE (unchanged) ----------
        S_text = lexical_similarity_matrix(rouge, gen_answers)
        S_text_np = np.asarray(S_text, dtype=np.float32)
        snne_val = snne(
            S_text, labels=None, variant="only_denom",
            temperature=args.temperature, epsilon=1e-8,
            exclude_diagonal=not args.include_self, weight=None
        )

        # ---------- Original QA-aware gating (your version) ----------
        if gen_answers:
            q_emb = encode_sentence_embeddings(
                text_encoder, [question], args.device
            )  # (1, d)
            a_embs = encode_sentence_embeddings(
                text_encoder, gen_answers, args.device
            )  # (N, d)
            qa_alignment = (q_emb @ a_embs.T).squeeze(0)
            qa_weights = softmax_weights(qa_alignment, beta=args.beta)
            S_text_gated_cosine = apply_gate_to_matrix(S_text_np, qa_weights)
            S_text_gated_cosine /= S_text_gated_cosine.max() + 1e-8  # optional rescale
            qa_snne_cosine = snne(
                S_text_gated_cosine, labels=None, variant="only_denom",
                temperature=args.temperature, epsilon=1e-8,
                exclude_diagonal=not args.include_self, weight=None
)
        else:
            qa_snne_cosine = None

        # ---------- NEW: BGE reranker gate ----------
        bge_scores = None
        if gen_answers:
            bge_scores = score_bge_reranker(bge_model, bge_tok, question, gen_answers, args.device)  # [N]
            bge_weights = softmax_weights(bge_scores, beta=args.beta)
            S_text_gated_crossenc = apply_gate_to_matrix(S_text_np, bge_weights)
            S_text_gated_crossenc /= S_text_gated_crossenc.max() + 1e-8  # optional rescale
            qa_snne_crossenc = snne(
                S_text_gated_crossenc, labels=None, variant="only_denom",
                temperature=args.temperature, epsilon=1e-8,
                exclude_diagonal=not args.include_self, weight=None
            )
        else:
            qa_snne_crossenc, r_bge = None, None

        # ---------- NEW: NLI bi-directional gate ----------
        nli_scores = None
        if gen_answers:
            nli_scores = score_nli_bidirectional(
                nli_model, nli_tok, question, gen_answers, args.device,
                alpha=args.nli_alpha, lambda_contra=args.nli_lambda
            )  # [N]
            S_text_gated_ent = apply_gate_to_matrix(S_text_np, bge_weights)
            S_text_gated_ent /= S_text_gated_ent.max() + 1e-8  # optional rescale
            qa_snne_ent = snne(
                S_text_gated_ent, labels=None, variant="only_denom",
                temperature=args.temperature, epsilon=1e-8,
                exclude_diagonal=not args.include_self, weight=None
            )
        else:
            qa_snne_ent, r_nli = None, None

        # Collect
        result.update({
            "se": dse,
            "snne": snne_val,
            "qa-snne-emb": qa_snne_cosine,  
            "qa-snne-crossenc": qa_snne_crossenc,                    
            "qa-snne-ent": qa_snne_ent,                  
        })

        results.append(result)

    # Save results
    stem = Path(args.input_file).stem
    output_file = Path(args.output_dir) / f"{stem}_combined_metrics.jsonl"
    save_results(results, str(output_file))
    print("=" * 60)
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    main()
