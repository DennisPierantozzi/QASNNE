<div align="center">

<h2>When to Trust the Answer: Question-Aligned Semantic Nearest Neighbor Entropy for Safer Surgical VQA</h2>

<p>
Luca Carlini*, Dennis Pierantozzi*, Mauro Orazio Drago, <br>
Chiara Lena, Cesare Hassan, Elena De Momi, <br>
Danail Stoyanov, Sophia Bano, and Mobarak I. Hoque
</p>

<p><sub>* Equal contribution (shared first authorship)</sub></p>

<p><i>International Journal of Computer Assisted Radiology and Surgery (IJCARS), 2026 · Open access</i></p>

---

<table align="center">
  <tr>
    <td><b><a href="https://doi.org/10.1007/s11548-026-03750-9">🏛️ Paper (IJCARS)</a></b></td>
    <td><b><a href="https://arxiv.org/abs/2511.01458">📄 arXiv preprint</a></b></td>
    <td><b><a href="https://huggingface.co/datasets/DennisPolimi/QASNNE">🤗 Dataset</a></b></td>
  </tr>
</table>

</div>

## Overview
QA-SNNE is a black-box, **question-aligned** uncertainty estimator for surgical visual question answering (VQA). It computes semantic nearest-neighbor entropy over sampled answers and **weights similarities by question–answer alignment**, so it can flag answers that are consistent with each other but do not address the question, without accessing model internals.

This is a preclinical methodological study: QA-SNNE is intended for evaluating automatic failure detection, not for clinical use.

## Highlights
- **Question-aware gating** focuses uncertainty on answers that actually address the query.
- **Black-box compatibility** with fine-tuned and proprietary LVLMs.
- **Robustness to paraphrases** and out-of-template wording.
- **EndoVis18-VQA Out-of-Template benchmark** released: 2,754 image–question pairs in which only the question wording changes, while images, answers and splits stay identical to the original validation split.

## Method (at a glance)
1. Sample multiple answers at high temperature.
2. Build an answer–answer similarity matrix.
3. Gate similarities by question–answer alignment (embedding, NLI or cross-encoder).
4. Compute semantic nearest-neighbor entropy and threshold it for detection, or use it as a continuous risk score.

## Diagram
![QA-SNNE Pipeline](docs/QA-SNNE-diagram.png)

## Installation
Clone the repository and create a virtual environment:

```bash
git clone https://github.com/DennisPierantozzi/QASNNE.git
cd QASNNE
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Some checkpoints (e.g. `meta-llama/Llama-3.2-11B-Vision-Instruct`) are gated on Hugging Face. Either log in:

```bash
huggingface-cli login
```

or copy `.env.example` to `.env`, add your token and load it:

```bash
cp .env.example .env   # then edit .env and set HF_TOKEN
source .env
```

`compute_vl_uncertainty.py` reads the token from `HF_TOKEN` by default; pass `--hf_token_env` to use a different variable name.

## Data
The datasets are on Hugging Face at [DennisPolimi/QASNNE](https://huggingface.co/datasets/DennisPolimi/QASNNE):

| Archive | Content |
| --- | --- |
| `Endovis18-VQA Out-of-Template.zip` | Our rephrased (out-of-template) version of EndoVis18-VQA, used as the out-of-template split |
| `EndoVis-18-VQA.zip` | The original EndoVis18-VQA data, used as the in-template split |

```bash
pip install huggingface_hub
huggingface-cli download --repo-type dataset DennisPolimi/QASNNE --local-dir ./data
```

EndoVis18-VQA was introduced by Seenivasan et al., *Surgical-VQA: Visual Question Answering in Surgical Scenes Using Transformer* (MICCAI 2022). If you use it, please cite the original work and follow its terms of use.

### Paths to update
Default paths point to the UCL cluster (`/SAN/...`). Update them before running:

| File | What to change |
| --- | --- |
| `utils/data_utils.py` | Dataset roots for `Endovis18VQA_new_template` (out-of-template), `Endovis18VQA_old_template` (in-template) and `PitVQASentence` |
| `utils/vlm_utils.py` | Local weights for the fine-tuned PitVQA and SurgicalGPT models |
| `compute_utility.py` | Default input files, only used with `--use-default-files` |
| `scripts/*.sh` | Cluster directives, environment activation and storage paths |

## Usage
**1. Generate answers and uncertainty rows**

```bash
python generate_answers.py \
    --dataset_name Endovis18VQA_new_template \
    --model_id Qwen/Qwen2.5-VL-3B-Instruct \
    --num_generations 20 \
    --output_jsonl outputs/qasnne_qwen.jsonl
```

**2. Compute SNNE / QA-SNNE scores**

```bash
python compute_metrics.py \
    --input_file outputs/qasnne_qwen.jsonl \
    --output_dir outputs/metrics \
    --device cuda \
    --bge_model_name BAAI/bge-reranker-large \
    --nli_model_name microsoft/deberta-large-mnli
```

**3. VL-Uncertainty baseline (paired perturbations)**

```bash
python compute_vl_uncertainty.py \
    --dataset_name Endovis18VQA_new_template \
    --model_id Qwen/Qwen2.5-VL-3B-Instruct \
    --system_prompt "You are a helpful vision-language assistant." \
    --output_jsonl outputs/vl_uncertainty.jsonl
```

**4. AUROC and thresholded metrics**

In the paper, an answer counts as a failure when its ROUGE-L against the reference is below 0.5 (`--tau 0.5`, the default).

```bash
python auroc_accuracy_analysis.py \
    --input qwen=outputs/metrics/qasnne_metrics.jsonl \
    --methods vl_uncertainty snne \
    --tau 0.5 \
    --out-dir outputs/analysis/auroc
```

**5. Risk–coverage curves**

```bash
python coverage_curves_analysis.py \
    --input outputs/metrics/qasnne_metrics.jsonl \
    --method vl_uncertainty \
    --quality rougeL bertscore \
    --out-dir outputs/analysis/coverage
```

**6. Text-level utility metrics (ROUGE, BLEU, METEOR)**

```bash
python compute_utility.py \
    --input outputs/qasnne_qwen.jsonl \
    --outdir outputs/analysis/text-metrics
```

### Python example

```python
from PIL import Image
from utils.vlm_utils import VLMClient

vlm = VLMClient("Qwen/Qwen2.5-VL-3B-Instruct", device="cuda")
image = Image.open("frame000.png").convert("RGB")
answer, metrics = vlm.vlm_answer(image, "What organ is being operated?", sampled=False)

# metrics["avg_prob"] is the mean token negative log-likelihood of the answer
print(answer, metrics["avg_prob"])
```

### Batch scripts
Job wrappers for an SGE cluster live under `scripts/`:
- `scripts/generate_answers.sh` launches answer generation with the paper's system prompt.
- `scripts/compute_vl_uncertainty.sh` runs the perturbation-based VL-Uncertainty baseline.
- `scripts/compute_metrics.sh` is a template for SNNE / QA-SNNE scoring jobs.

## Citation
```bibtex
@article{Carlini2026QASNNE,
  title   = {When to Trust the Answer: Question-Aligned Semantic Nearest Neighbor Entropy for Safer Surgical {VQA}},
  author  = {Carlini, Luca and Pierantozzi, Dennis and Drago, Mauro Orazio and Lena, Chiara and Hassan, Cesare and De Momi, Elena and Stoyanov, Danail and Bano, Sophia and Hoque, Mobarak I.},
  journal = {International Journal of Computer Assisted Radiology and Surgery},
  year    = {2026},
  doi     = {10.1007/s11548-026-03750-9}
}
```

## License
The code is released under the [MIT License](LICENSE). The datasets are derived from EndoVis18-VQA: check the original dataset's terms before reusing them.
