#!/bin/bash
#$ -S /bin/bash
#$ -l tmem=24G
#$ -l h_vmem=24G
#$ -l h_rt=40:00:00
#$ -N CM-QWEN
#$ -l gpu=true
#$ -l gpu_type=a6000
#$ -o /SAN/medic/Cholec/thesis/experiments/metrics_output
#$ -e /SAN/medic/Cholec/thesis/experiments/metrics_output

# Activate virtual environment
#source /home/dpierant/.venv/bin/activate

# Move to project root
cd /SAN/medic/Cholec/thesis/experiments


# Run the training script

# qwen
python compute_snne_qa_alternatives.py \
    --input_file /SAN/medic/Cholec/thesis/generations/pit_dataset/zero_shot/generation_llama_zs_dataset.jsonl \
    --output_dir /SAN/medic/Cholec/thesis/metrics/pit_dataset/zero_shot/
