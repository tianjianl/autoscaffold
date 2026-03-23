#!/bin/bash
#SBATCH --job-name=auto_rc_hmmt
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=auto_rc_hmmt_%j.out
#SBATCH --error=auto_rc_hmmt_%j.err

set -eo pipefail

set +u
source ~/.bashrc
set -u

echo "=== Auto-RC Scaffold: GPT-5-nano HMMT Feb 2026 ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "API key set: $([ -n "${OPENAI_API_KEY:-}" ] && echo 'yes' || echo 'no')"
echo ""

python auto_rc.py \
    --output predictions_auto_rc.jsonl \
    --model gpt-5-nano \
    --max-completion-tokens 20000 \
    --max-summarization-tokens 4000 \
    --temperature 1 \
    --max-steps 5 \
    --poll-interval 30 \
    --run

echo ""
echo "--- Grading results ---"
python grade.py \
    --predictions predictions_auto_rc.jsonl \
    --workers 8 \
    --verbose

echo ""
echo "=== Job completed at $(date) ==="
