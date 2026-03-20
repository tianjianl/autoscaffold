#!/bin/bash
#SBATCH --job-name=query_hmmt
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=query_hmmt_%j.out
#SBATCH --error=query_hmmt_%j.err

set -eo pipefail

# Source API key (disable nounset temporarily for ~/.bashrc compatibility)
set +u
source ~/.bashrc
set -u

echo "=== HMMT Feb 2026 Query Job ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "API key set: $([ -n "${OPENROUTER_API_KEY:-}" ] && echo 'yes' || echo 'no')"
echo ""

# Query all 33 problems
python query_model.py \
    --output predictions.jsonl \
    --model qwen/qwen3.5-9b \
    --workers 33 \
    --max-total-tokens 2500000 \
    --max-completion-tokens 32000 \
    --temperature 0.7

echo ""
echo "--- Grading results ---"
python grade.py \
    --predictions predictions.jsonl \
    --workers 8 \
    --verbose

echo ""
echo "=== Job completed at $(date) ==="
