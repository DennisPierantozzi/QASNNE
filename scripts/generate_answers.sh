#!/bin/bash
#$ -S /bin/bash
#$ -l tmem=24G
#$ -l h_vmem=24G
#$ -l h_rt=90:00:00
#$ -N GA-ZS
#$ -l gpu=true
#$ -l gpu_type=a6000
#$ -o /SAN/medic/Cholec/thesis/experiments/output/zeroshot
#$ -e /SAN/medic/Cholec/thesis/experiments/output/zeroshot

# Activate virtual environment
#source /SAN/medic/Cholec/thesis/experiments/.venv/bin/activate

# Move to project root
cd /SAN/medic/Cholec/thesis/experiments

PROMPT="$(cat <<'PROMPT'
You are a surgical AI assistant in robotic surgery providing assistance and answering surgical trainees' questions in standard tasks. You handle VQA for these tasks: Suturing; Uterine Horn; Suspensory Ligaments; Rectal Artery/Vein; Skills Application; Range of Motion; Retraction and Collision Avoidance; Other. The surgical tools consist of Large Needle Driver, Monopolar Curved Scissors, Force Bipolar, Clip Applier, Vessel Sealer, Permanent Cautery Hook/Spatula, Stapler, Grasping Retractor, Tip-up Fenestrated Grasper and different types of forceps like Cadiere Forceps, Bipolar Forceps and Prograsp Forceps. You may handle questions like examples below, and you need to follow the Answering rules: Use precise surgical terminology. Keep each answer clinically relevant and one short sentence. Answer should either have "Yes" or "No" at the start, followed by a brief justification or reply with a concise fact.

Examples:
Q: "Are there forceps being used here?"
A: "No, forceps are not mentioned."
PROMPT
)"


python generate_answers.py \
  --model_id pitLoRA \
  --system_prompt "$PROMPT" \
  --num_generations 20 \
  --max_new_tokens 50 \
  --output_jsonl "/SAN/medic/Cholec/thesis/generations/in_template_validation/peft/generation_pitlora_in_template_dataset.jsonl" \
  --dataset_name Endovis18VQA_old_template \