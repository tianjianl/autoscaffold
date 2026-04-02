#!/bin/bash
#SBATCH --job-name=query_gpt5mini_aime
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/query_gpt5mini_aime_%j.out
#SBATCH --error=slurm_logs/query_gpt5mini_aime_%j.err

set -eo pipefail

set +u
source ~/.bashrc
set -u

echo "=== GPT-5-mini AIME 2026 Batch Job ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "API key set: $([ -n "${OPENAI_API_KEY:-}" ] && echo 'yes' || echo 'no')"
echo ""

python query_openai_batch.py \
    --output model_outputs/predictions_gpt5mini_aime.jsonl \
    --model gpt-5-mini \
    --dataset MathArena/aime_2026 \
    --max-completion-tokens 32000 \
    --temperature 1 \
    --poll-interval 30 \
    --run

echo ""
echo "--- Grading results ---"
python grade.py \
    --predictions model_outputs/predictions_gpt5mini_aime.jsonl \
    --dataset MathArena/aime_2026 \
    --workers 8 \
    --verbose

echo ""
echo "=== Job completed at $(date) ==="
