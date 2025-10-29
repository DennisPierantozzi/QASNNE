#!/usr/bin/env python3
"""
vl_uncertainty_metrics.py

Compute uncertainty-based metrics for VLMs exactly as in the paper:
- AvgProb (mean NLL of chosen tokens)
- MaxProb (max NLL of chosen tokens)
- AvgEnt  (mean full-vocab token entropy)
- MaxEnt  (max full-vocab token entropy)

Notes
-----
- Main response should be greedy (temperature=0.0) per paper.
- We exclude the EOS token from all four aggregates by truncating at EOS.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# Dataloaders (same signatures as your script)
from torchvision.transforms.functional import to_pil_image
from utils.data_utils import build_dataloader
from utils.vlm_utils import VLMClient


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

def save_jsonl(rows: List[Dict[str,Any]], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(_to_jsonable(r), ensure_ascii=False) + "\n")
    print(f"✓ Saved {len(rows)} rows -> {out_path}")


# ---------------------- Main ----------------------
def main():
    ap = argparse.ArgumentParser(description="Compute AvgProb/MaxProb/AvgEnt/MaxEnt for VLM generations")
    # Data
    ap.add_argument("--dataset_name", type=str, default=None,
                    choices=["Endovis18VQA_new_template", "Endovis18VQA_old_template", "PitVQASentence"],
                    help="Use one of the predefined datasets. If omitted, --input_jsonl is required.")
    ap.add_argument("--input_jsonl", type=str, default=None,
                    help="Alternative to --dataset_name: JSONL with id,image_path,question[,reference]")
    ap.add_argument("--num_samples", type=int, default=None, help="Optional cap on number of samples")
    ap.add_argument("--num_generations", type=int, default=20, help="Number of generations")
    ap.add_argument("--output_jsonl", type=str, required=False)

    # Model
    ap.add_argument("--model_id", type=str, required=True,
                    choices=[
                        "Qwen/Qwen2.5-VL-3B-Instruct",
                        "google/medgemma-4b-it",
                        "meta-llama/Llama-3.2-11B-Vision-Instruct",
                        "pitLoRA",
                        "surgicalGPT",
                    ])
    ap.add_argument("--system_prompt", type=str, required=False, help="System message text")

    # Generation
    ap.add_argument("--temperature", type=float, default=1.0, help="0.0 = greedy main response (paper default)")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # Enforce exactly one of dataset/jsonl
    if (args.dataset_name is None) == (args.input_jsonl is None):
        raise SystemExit("Please provide exactly one of --dataset_name or --input_jsonl")
    
    if (args.model_id == "pivqalora" or args.model_id == "surgicalGPT"):
        is_peft = True
    else:
        is_peft = False

    # Load model
    vlm_model = VLMClient(args.model_id, is_peft=is_peft)

    rows: List[Dict[str, Any]] = []
    processed = 0

    if args.dataset_name is not None:
        loader = build_dataloader(args.dataset_name)
        for batch in tqdm(loader, desc="Batches"):
            if len(batch) == 4:
                batch_images, batch_questions, batch_refs, batch_paths = batch
            else:
                batch_images, batch_questions, batch_refs = batch
                batch_paths = [None] * len(batch_questions)

            B = len(batch_questions)
            for b in range(B):
                if args.num_samples and processed >= args.num_samples:
                    break
                
                pil_image = to_pil_image(batch_images[b].cpu()) if torch.is_tensor(batch_images[b]) else batch_images[b]
                
                question = batch_questions[b]
                reference = batch_refs[b]
                img_path = batch_paths[b]

                # Most likely answer + metrics

                mostl_likely_answer, metrics = vlm_model.vlm_answer(
                    pil_image, question,
                    device=args.device,
                    temperature=0.1,
                    max_new_tokens=args.max_new_tokens,
                    system_prompt=args.system_prompt,
                    sampled= False
                )
                
                generated_answers = []
                if args.model_id == "pitLoRA" or args.model_id == "surgicalGPT":
                    for i in range(args.num_generations):
                        answer = vlm_model.vlm_answer(
                            pil_image, question,
                            device= "cuda",
                            temperature= args.temperature,
                            max_new_tokens = args.max_new_tokens,
                            top_p= args.top_p,
                            top_k= args.top_k,
                            system_prompt = args.system_prompt,
                            sampled = True,
                            seed=i,
                            num_samples=args.num_generations
                        )
                        generated_answers.append(answer)
                else: 
                    generated_answers = vlm_model.vlm_answer(
                        pil_image,
                        question,
                        device=args.device,
                        temperature=args.temperature,
                        max_new_tokens=args.max_new_tokens,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        system_prompt=args.system_prompt,
                        sampled=True,
                        num_samples=args.num_generations,
                    ) 

                rows.append({
                    "image_path": img_path,
                    "question": question,
                    "reference": reference,
                    "most_likely_answer": mostl_likely_answer,
                    "avg_nll": metrics["avg_prob"],         # mean NLL
                    "max_nll": metrics["max_prob"],         # max  NLL
                    "avg_entropy": metrics["avg_entropy"],
                    "max_entropy": metrics["max_entropy"],
                    "generated_answers": generated_answers
                })

                processed += 1

            if args.num_samples and processed >= args.num_samples:
                break

    save_jsonl(rows, Path(args.output_jsonl))
    print(f"✓ Processed {processed} samples")


if __name__ == "__main__":
    main()
