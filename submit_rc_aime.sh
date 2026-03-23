#!/bin/bash
#SBATCH --job-name=rc_aime2026
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=rc_aime2026_%j.out
#SBATCH --error=rc_aime2026_%j.err

set -eo pipefail

set +u
source ~/.bashrc
set -u

echo "=== RC Scaffold: GPT-5-nano AIME 2026 ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "API key set: $([ -n "${OPENAI_API_KEY:-}" ] && echo 'yes' || echo 'no')"
echo ""

python rc_scaffold.py \
    --output predictions_rc_aime.jsonl \
    --model gpt-5-nano \
    --max-completion-tokens 20000 \
    --max-summarization-tokens 4000 \
    --temperature 1 \
    --max-steps 3 \
    --n 1 \
    --dataset MathArena/aime_2026 \
    --poll-interval 30 \
    --run

echo ""
echo "--- Grading results ---"
python grade.py \
    --predictions predictions_rc_aime.jsonl \
    --dataset MathArena/aime_2026 \
    --workers 8 \
    --verbose

echo ""
echo "=== Job completed at $(date) ==="
