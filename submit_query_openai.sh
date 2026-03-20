#!/bin/bash
#SBATCH --job-name=query_gpt5nano
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=query_gpt5nano_%j.out
#SBATCH --error=query_gpt5nano_%j.err

set -eo pipefail

# Source API key (disable nounset temporarily for ~/.bashrc compatibility)
set +u
source ~/.bashrc
set -u

echo "=== GPT-5-nano HMMT Feb 2026 Batch Job ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "API key set: $([ -n "${OPENAI_API_KEY:-}" ] && echo 'yes' || echo 'no')"
echo ""

# Full pipeline: submit batch, wait for completion, download and grade
python query_openai_batch.py \
    --output predictions_gpt5nano.jsonl \
    --model gpt-5-nano \
    --max-completion-tokens 32000 \
    --temperature 1 \
    --poll-interval 30 \
    --run

echo ""
echo "--- Grading results ---"
python grade.py \
    --predictions predictions_gpt5nano.jsonl \
    --workers 8 \
    --verbose

echo ""
echo "=== Job completed at $(date) ==="
