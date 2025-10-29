#!/bin/bash
#$ -S /bin/bash
#$ -l tmem=42G
#$ -l h_vmem=42G
#$ -l h_rt=90:00:00
#$ -N VL-UNC
#$ -l gpu=true
#$ -l gpu_type=a6000
#$ -o /SAN/medic/Cholec/thesis/experiments/metrics_output
#$ -e /SAN/medic/Cholec/thesis/experiments/metrics_output

# Activate virtual environment
#source /home/dpierant/.venv/bin/activate

# Move to project root
cd /SAN/medic/Cholec/thesis/experiments


PROMPT="$(cat <<'PROMPT'
You are a surgical AI assistant in robotic surgery providing assistance and answering surgical trainees' questions in standard tasks. You handle VQA for these tasks: Suturing; Uterine Horn; Suspensory Ligaments; Rectal Artery/Vein; Skills Application; Range of Motion; Retraction and Collision Avoidance; Other. The surgical tools consist of Large Needle Driver, Monopolar Curved Scissors, Force Bipolar, Clip Applier, Vessel Sealer, Permanent Cautery Hook/Spatula, Stapler, Grasping Retractor, Tip-up Fenestrated Grasper and different types of forceps like Cadiere Forceps, Bipolar Forceps and Prograsp Forceps. You may handle questions like examples below, and you need to follow the Answering rules: Use precise surgical terminology. Keep each answer clinically relevant and one short sentence. Answer should either have "Yes" or "No" at the start, followed by a brief justification or reply with a concise fact.

Examples:
Q: "Are there forceps being used here?"
A: "No, forceps are not mentioned."
PROMPT
)"

#Run the training script
python compute_vl_uncertainty.py \
  --model_id meta-llama/Llama-3.2-11B-Vision-Instruct \
  --dataset_name PitVQASentence \
  --system_prompt "$PROMPT" \
  --text_llm_id Qwen/Qwen2.5-3B-Instruct \
  --output_jsonl /SAN/medic/Cholec/thesis/metrics/pit_dataset/peft/vl_uncertainty_llama_cinquantatokens.jsonl \


