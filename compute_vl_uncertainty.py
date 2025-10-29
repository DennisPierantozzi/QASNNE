#!/usr/bin/env python3
"""
vl_uncertainty_end2end.py

Minimal, self-contained implementation of **VL-Uncertainty only**, now with
**dataset argument + dataloaders** matching your previous script.

This script:
  1) Loads your requested dataset via `--dataset_name` **or** a JSONL via `--input_jsonl`.
  2) Generates one low-temp base answer and N paired answers under semantic-
     equivalent visual + textual perturbations.
  3) Clusters the N answers by bidirectional NLI entailment and computes entropy
     of the cluster distribution as **vl_uncertainty**.
  4) Saves one JSONL row per sample for later analysis.

Datasets supported (as in your script):
- `Endovis18VQA_new_template`
- `Endovis18VQA_old_template`
- `RealColon`

Output JSONL fields per sample:
  - id, image_path, question, reference
  - base_answer
  - perturb: blur_radius, text_temps, rephrased_questions, answers
  - semantic: cluster_index, cluster_sizes, entailment_threshold
  - vl_uncertainty (float)

Notes
-----
- Only computes VL-Uncertainty; no VL-SNNE.
- Visual perturbations: Gaussian blur (see `visual_perturbation.image_blurring`).
- Textual perturbations: small text LLM rephrasing; can disable with `--text_llm_id none`.
- Entailment backend: `roberta-large-mnli` by default (configurable).
"""

from __future__ import annotations
import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModelForSequenceClassification
from huggingface_hub import login

# === your uploaded utilities (must be importable or placed alongside this file) ===
from utils.vl_uncertainty_utils.visual_perturbation import image_blurring  # semantic-equivalent visual perturb

# === your dataloaders (same import path/signature as your previous script) ===
from dataloader import EndoVis18VQA
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import DataLoader

from utils.data_utils import build_dataloader
from utils.vlm_utils import (
VLMClient
)

# ---------------------- I/O helpers ----------------------
def _to_jsonable(x: Any) -> Any:
    import numpy as _np, torch as _th
    if isinstance(x, dict): return {k:_to_jsonable(v) for k,v in x.items()}
    if isinstance(x, (list, tuple)): return [_to_jsonable(v) for v in x]
    if isinstance(x, _np.ndarray): return x.tolist()
    if isinstance(x, _np.floating): return float(x)
    if isinstance(x, _np.integer): return int(x)
    if isinstance(x, _th.Tensor): return x.detach().cpu().tolist() if x.ndim>0 else x.item()
    return x

def save_jsonl(rows: List[Dict[str, Any]], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            json.dump(_to_jsonable(r), f, ensure_ascii=False)
            f.write("\n")  # <-- one JSON object per line
    print(f"✓ Saved -> {out_path}")

# ---------------------- model loaders ----------------------
@dataclass
class TextLLM:
    tok: AutoTokenizer
    mdl: AutoModelForCausalLM
    eos_id: Optional[int]


def get_text_llm(text_llm_id: Optional[str]) -> Optional[TextLLM]:
    if text_llm_id is None or str(text_llm_id).lower() in {"none", ""}:
        return None
    tok = AutoTokenizer.from_pretrained(text_llm_id)
    mdl = AutoModelForCausalLM.from_pretrained(
        text_llm_id, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True
    ).eval()
    eos_id = getattr(mdl.generation_config, "eos_token_id", None)
    if isinstance(eos_id, list):
        eos_id = eos_id[0] if eos_id else None
    return TextLLM(tok, mdl, eos_id)

@dataclass
class EntailmentModel:
    tok: AutoTokenizer
    cls: AutoModelForSequenceClassification
    idx_ent: int
    idx_con: int


def get_entailment_model(model_id: str, device: str) -> EntailmentModel:
    tok = AutoTokenizer.from_pretrained(model_id)
    cls = AutoModelForSequenceClassification.from_pretrained(model_id).to(device).eval()
    # roberta-large-mnli: 0=contradiction, 1=neutral, 2=entailment
    id2label = getattr(cls.config, "id2label", {0:"contradiction",1:"neutral",2:"entailment"})
    lut = {int(k): v.lower() for k,v in id2label.items()}
    def find(s):
        for i,n in lut.items():
            if s in n:
                return i
        raise RuntimeError(f"label '{s}' not found in {lut}")
    return EntailmentModel(tok, cls, idx_ent=find("entail"), idx_con=find("contrad"))

# ---------------------- generation utils ----------------------

@torch.inference_mode()
def text_llm_rephrase(llm: TextLLM, question: str, temperature: float, max_new_tokens=64) -> str:
    if llm is None:
        return question
    prompt = (
        f"Given the input question: '{question}', generate a semantically equivalent variation "
        f"by changing the wording, structure, grammar, or narrative. Ensure the perturbed question "
        f"maintains the same meaning as the original. Provide only the rephrased question as the output."
    )
    toks = llm.tok(prompt, return_tensors="pt").to(llm.mdl.device)
    gen = llm.mdl.generate(
        **toks,
        do_sample=True,
        temperature=float(temperature),
        top_p=0.9,
        max_new_tokens=max_new_tokens,
        eos_token_id=llm.eos_id,
        pad_token_id=llm.tok.eos_token_id,
    )
    txt = llm.tok.decode(gen[0], skip_special_tokens=True)
    if txt.startswith(prompt):
        txt = txt[len(prompt):].strip()
    return txt.strip().strip('"')

@torch.inference_mode()
def entail_prob(ent: EntailmentModel, a: str, b: str) -> float:
    inp = ent.tok(a, b, return_tensors="pt", truncation=True).to(ent.cls.device)
    logits = ent.cls(**inp).logits[0].float()
    prob = torch.softmax(logits, dim=-1)
    return float(prob[ent.idx_ent].item())

# ---------------------- VL-Uncertainty core ----------------------
@dataclass
class VLUConfig:
    blur_radius: List[float]
    text_temps: List[float]
    sampling_temp: float
    base_temp: float
    entail_thresh: float


def paired_perturbations(cfg: VLUConfig, image: Image.Image, question: str, llm: Optional[TextLLM]) -> Tuple[List[Image.Image], List[str]]:
    # Pair i-th blur with i-th rephrasing (semantic-equivalent pairing)
    imgs = [image_blurring(image, r) for r in cfg.blur_radius]
    qs   = [text_llm_rephrase(llm, question, t) if llm is not None else question for t in cfg.text_temps]
    L = min(len(imgs), len(qs))
    return imgs[:L], qs[:L]


def cluster_by_semantics(answers: List[str], ent: EntailmentModel, thresh: float) -> Tuple[List[int], List[int]]:
    N = len(answers)
    cluster_idx = [-1] * N
    cur = 0
    for i in range(N):
        if cluster_idx[i] != -1:
            continue
        cluster_idx[i] = cur
        for j in range(i+1, N):
            if cluster_idx[j] != -1:
                continue
            pij = entail_prob(ent, answers[i], answers[j])
            pji = entail_prob(ent, answers[j], answers[i])
            if (pij >= thresh) and (pji >= thresh):
                cluster_idx[j] = cur
        cur += 1
    # sizes
    counts = {}
    for c in cluster_idx:
        counts[c] = counts.get(c, 0) + 1
    sizes = [counts[k] for k in sorted(counts.keys())]
    return cluster_idx, sizes


def entropy_from_sizes(sizes: List[int]) -> float:
    if not sizes:
        return 0.0
    N = float(sum(sizes))
    H = 0.0
    for s in sizes:
        p = s / N
        H -= p * math.log2(max(p, 1e-12))
    return float(H)

# ---------------------- main pipeline ----------------------

def main():
    ap = argparse.ArgumentParser(description="VL-Uncertainty only: generate answers + compute uncertainty + save JSONL")
    # Data: choose either a dataset or a JSONL
    ap.add_argument("--dataset_name", type=str, default=None,
                    choices=["Endovis18VQA_new_template","Endovis18VQA_old_template", "PitVQASentence"],
                    help="Use one of the predefined datasets. If omitted, --input_jsonl is required.")
    ap.add_argument("--num_samples", type=int, default=None, help="Optional cap on number of samples from dataset")
    ap.add_argument("--input_jsonl", type=str, default=None, help="Alternative to --dataset_name: id,image_path,question[,reference]")
    ap.add_argument("--output_jsonl", type=str, required=True)

    # Models
    ap.add_argument("--model_id", type=str, required=True,
        choices=[
            "Qwen/Qwen2.5-VL-3B-Instruct",
            "google/medgemma-4b-it",
            "meta-llama/Llama-3.2-11B-Vision-Instruct",
            "pitLoRA",
            "surgicalGPT"
        ])
    ap.add_argument("--text_llm_id", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                    help="Text-only LLM for rephrasing. Use 'none' to disable textual perturbation.")
    ap.add_argument("--entailment_model_id", type=str, default="roberta-large-mnli",
                    help="NLI model for bidirectional entailment.")
    ap.add_argument("--system_prompt", type=str, required=True, help="System message text")

    # Generation
    ap.add_argument("--base_temp", type=float, default=0.1, help="Temperature for base (unperturbed) answer")
    ap.add_argument("--sampling_temp", type=float, default=1.0, help="Temperature for LVLM answers to paired perturbations")
    ap.add_argument("--max_new_tokens", type=int, default=50)
    ap.add_argument(
        "--hf_token_env",
        type=str,
        default="HF_TOKEN",
        help="Environment variable that stores the Hugging Face access token required for gated models.",
    )

    # Perturbations (defaults follow the paper)
    ap.add_argument("--blur_radius", type=float, nargs="+", default=[0.6, 0.8, 1.0, 1.2, 1.4])
    ap.add_argument("--text_temps", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5])

    # Entailment
    ap.add_argument("--entail_thresh", type=float, default=0.5, help="P(entail) threshold for two-way entailment")

    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if (args.dataset_name is None) == (args.input_jsonl is None):
        raise SystemExit("Please provide exactly one of --dataset_name or --input_jsonl")

    # VLM
    if args.model_id == "meta-llama/Llama-3.2-11B-Vision-Instruct":
        hf_token = os.getenv(args.hf_token_env)
        if not hf_token:
            raise SystemExit(
                f"Hugging Face token required for {args.model_id}. "
                f"Set the {args.hf_token_env} environment variable (see .env.example)."
            )
        login(token=hf_token)

    vlm_model = VLMClient(args.model_id, device=device)
    # Text LLM (optional)
    rephraser = get_text_llm(args.text_llm_id)
    # Entailment model (for clustering)
    ent = get_entailment_model(args.entailment_model_id, device=device)

    cfg = VLUConfig(
        blur_radius=args.blur_radius,
        text_temps=args.text_temps,
        sampling_temp=args.sampling_temp,
        base_temp=args.base_temp,
        entail_thresh=args.entail_thresh,
    )

    rows: List[Dict[str,Any]] = []
    processed = 0

    print(f"=== VL-Uncertainty with {args.model_id} ===")

    if args.dataset_name is not None:
        loader = build_dataloader(args.dataset_name)
        for batch in loader:
            # simple_collate_fn_with_paths returns (images, questions, refs, paths)
            if len(batch) == 4:
                batch_images, batch_questions, batch_refs, batch_paths = batch
            else:
                # EndoVis18VQA path (no paths)
                batch_images, batch_questions, batch_refs = batch
                batch_paths = [None]*len(batch_questions)

            B = len(batch_questions)
            for b in range(B):
                if args.num_samples and processed >= args.num_samples:
                    break

                pil_image = to_pil_image(batch_images[b].cpu())
                question  = batch_questions[b]
                reference = batch_refs[b]
                img_path  = batch_paths[b]

                # base answer at low temp
                base_answer, _ = vlm_model.vlm_answer(
                    pil_image, question, device=device,
                    temperature=cfg.base_temp, max_new_tokens=args.max_new_tokens, system_prompt=args.system_prompt
                )

                # paired perturbations
                img_list, q_list = paired_perturbations(cfg, pil_image, question, rephraser)

                # answers on pairs
                answers = []
                for im_i, q_i in zip(img_list, q_list):
                    a_i, _ = vlm_model.vlm_answer(
                        im_i, q_i, device=device,
                        temperature=cfg.sampling_temp, max_new_tokens=args.max_new_tokens, system_prompt=args.system_prompt
                    )
                    answers.append(a_i)

                # cluster by semantics (two-way entailment)
                cl_idx, cl_sizes = cluster_by_semantics(answers, ent, cfg.entail_thresh)
                H = entropy_from_sizes(cl_sizes)

                sid = img_path if isinstance(img_path, str) and len(img_path) else f"{args.dataset_name}_{processed}"
                rows.append({
                    "image_path": img_path,
                    "question": question,
                    "reference": reference,
                    "base_answer": base_answer,
                    "perturb": {
                        "blur_radius": cfg.blur_radius[:len(answers)],
                        "text_temps": cfg.text_temps[:len(answers)],
                        "rephrased_questions": q_list,
                        "answers": answers,
                    },
                    "semantic": {
                        "cluster_index": cl_idx,
                        "cluster_sizes": cl_sizes,
                        "entailment_threshold": cfg.entail_thresh,
                    },
                    "vl_uncertainty": H,
                })
                processed += 1

            if args.num_samples and processed >= args.num_samples:
                break
    else:
        with open(args.input_jsonl, "r", encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        for sample in tqdm(lines):
            sid = sample.get("id")
            q   = sample.get("question")
            pth = sample.get("image_path")
            ref = sample.get("reference", None)
            if not q or not pth or not Path(pth).exists():
                continue

            img = Image.open(pth).convert("RGB")
            # base answer at low temp
            base_answer = vlm_answer(
                vlm, vlm_proc, img, q, device=device,
                temperature=cfg.base_temp, max_new_tokens=args.max_new_tokens, system_prompt=args.system_prompt
            )

            # paired perturbations
            img_list, q_list = paired_perturbations(cfg, img, q, rephraser)

            # answers on pairs
            answers = []
            for im_i, q_i in zip(img_list, q_list):
                a_i, _ = vlm_answer(
                    vlm, vlm_proc, im_i, q_i, device=device,
                    temperature=cfg.sampling_temp, max_new_tokens=args.max_new_tokens, system_prompt=args.system_prompt
                )
                answers.append(a_i)

            # cluster by semantics (two-way entailment)
            cl_idx, cl_sizes = cluster_by_semantics(answers, ent, cfg.entail_thresh)
            H = entropy_from_sizes(cl_sizes)

            rows.append({
                "id": sid,
                "image_path": pth,
                "question": q,
                "reference": ref,
                "base_answer": base_answer,
                "perturb": {
                    "blur_radius": cfg.blur_radius[:len(answers)],
                    "text_temps": cfg.text_temps[:len(answers)],
                    "rephrased_questions": q_list,
                    "answers": answers,
                },
                "semantic": {
                    "cluster_index": cl_idx,
                    "cluster_sizes": cl_sizes,
                    "entailment_threshold": cfg.entail_thresh,
                },
                "vl_uncertainty": H,
            })

    save_jsonl(rows, Path(args.output_jsonl))


if __name__ == "__main__":
    main()
