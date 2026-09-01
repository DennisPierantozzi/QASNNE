<div align="center">

<h2>When to Trust the Answer: Question-Aligned Semantic Neighbour Entropy for Safer Surgical VQA</h2>

<p>
Dennis Pierantozzi*, Luca Carlini*, Mauro Orazio Drago, <br>
Chiara Lena, Cesare Hassan, Elena De Momi, <br>
Danail Stoyanov, Sophia Bano, and Mobarak I. Hoque*
</p>

---

<table align="center">
  <tr>
    <td><b><a href="https://arxiv.org/abs/2511.01458">📄 arXiv Preprint</a></b></td>
    <td><b><a href="https://doi.org/10.1007/s11548-026-03750-9">🏛️ Springer Journal</a></b></td>
  </tr>
</table>

</div>

## Overview
QA-SNNE is a black-box, **question-aligned** uncertainty estimator for surgical visual question answering (VQA). It computes semantic nearest-neighbor entropy over sampled answers and **weights similarities by question–answer alignment**, enabling reference-free hallucination and ambiguity detection without accessing model internals.

## Highlights
- **Question-aware gating** focuses uncertainty on answers that actually address the query.
- **Black-box compatibility** with fine-tuned and proprietary LVLMs.
- **Robustness to paraphrases** and out-of-template wording.
- **EndoVis Out-of-Template benchmark** released for evaluating QA-SNNE under distribution shifts (download instructions below).

## Method (at a glance)
1. Sample multiple answers at high temperature.  
2. Build an answer–answer similarity matrix in an embedding space.  
3. Gate similarities by question–answer alignment scores.  
4. Compute semantic nearest-neighbor entropy and threshold for detection or use as a continuous risk score.

## Diagram
![QA-SNNE Pipeline](docs/QA-SNNE-diagram.png)

## Installation
- Clone the repository and create a virtual environment:

```bash
git clone https://github.com/DennisPierantozzi/QA-SNNE.git
cd QA-SNNE
python3 -m venv .venv
source .venv/bin/activate
```

- Upgrade `pip` and install the dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

- (Optional) Log into Hugging Face so the scripts can download gated checkpoints:

```bash
huggingface-cli login
```

- Alternatively, copy `.env.example` to `.env`, populate `HF_TOKEN`, and export it before running scripts that require gated models:

```bash
cp .env.example .env
source .env  # or use your preferred dotenv loader
```

## Usage
- **Generate LVLM answers and uncertainty rows**

  ```bash
  python generate_answers.py \
      --dataset_name Endovis18VQA_new_template \
      --model_id Qwen/Qwen2.5-VL-3B-Instruct \
      --num_generations 20 \
      --output_jsonl outputs/qasnne_qwen.jsonl
  ```

- **Compute QA-SNNE / VL-SNNE metrics**

  ```bash
  python compute_metrics.py \
      --input_file outputs/qasnne_qwen.jsonl \
      --output_dir outputs/metrics \
      --device cuda \
      --bge_model_name BAAI/bge-reranker-large \
      --nli_model_name microsoft/deberta-large-mnli
  ```

- **VL-Uncertainty only (paired perturbations)**

```bash
python compute_vl_uncertainty.py \
    --dataset_name Endovis18VQA_new_template \
    --model_id Qwen/Qwen2.5-VL-3B-Instruct \
    --system_prompt "You are a helpful vision-language assistant." \
    --output_jsonl outputs/vl_uncertainty.jsonl
```

If you switch to `meta-llama/Llama-3.2-11B-Vision-Instruct`, make sure the `HF_TOKEN` environment variable is set (see `.env.example`) or pass a different variable name via `--hf_token_env`.

- **Risk–coverage / coverage curve analysis**

  ```bash
  python coverage_curves_analysis.py \
      --input outputs/metrics/qasnne_metrics.jsonl \
      --method vl_uncertainty \
      --quality rougeL bertscore \
      --out-dir outputs/analysis/coverage
  ```

- **AUROC / accuracy sweeps**

  ```bash
  python auroc_accuracy_analysis.py \
      --input qwen=outputs/metrics/qasnne_metrics.jsonl \
      --methods vl_uncertainty snne \
      --tau 0.2 \
      --out-dir outputs/analysis/auroc
  ```

- **Text-level utility metrics (ROUGE/BLEU/METEOR)**

  ```bash
  python compute_utility.py \
      --input outputs/qasnne_qwen.jsonl \
      --outdir outputs/analysis/text-metrics
  ```

### Python API example

```python
from uncertainty.utils.vlm_utils import VLMClient
from uncertainty.semantic_entropy import predictive_entropy

vlm = VLMClient("Qwen/Qwen2.5-VL-3B-Instruct", device="cuda")
answer, token_metrics = vlm.vlm_answer(pil_image, question, sampled=False)
entropy = predictive_entropy(token_metrics["avg_prob"])
print(answer, entropy)
```

### Dataset paths

Update the dataset roots in `utils/data_utils.py` before running dataset-driven scripts; the defaults point to institutional storage under `/SAN/...`.

To pull the EndoVis and the EndoVis Out-of-Template evaluation split introduced in this work, you can download it directly from our public Hugging Face repository. Ensure you have the `huggingface_hub` library installed (`pip install huggingface_hub`), then run:

```bash
huggingface-cli download --repo-type dataset DennisPolimi/QASNNE --local-dir ./dataset_folder
```

### Batch scripts

Reusable HPC job wrappers live under `scripts/`. They assume the institution’s cluster setup (SGE directives, `/SAN/...` paths, and a `conda` environment) and can be customised:
- `scripts/generate_answers.sh` launches LVLM generation with the paper’s system prompt.
- `scripts/compute_vl_uncertainty.sh` runs the perturbation-based VL uncertainty pipeline.
- `scripts/compute_metrics.sh` is a template for SNNE/VL-SNNE metric calculation jobs.

Edit environment activation, storage paths, and model IDs before submitting them to your scheduler.

## Citation
```bibtex
@inproceedings{Pierantozzi2025QASNNE,
  title     = {When to Trust the Answer: Question-Aligned Semantic Neighbour Entropy for Safer Surgical VQA},
  author    = {Pierantozzi Dennis and Carlini Luca and Drago Mauro O. and Lena Chiara and Hassan Cesare and Stoyanov Danail and Bano Sophia and Hoque Mobarak I.},
  year      = {2025},
  note      = {Under submission}
}
