#!/bin/bash
#SBATCH --job-name=rc_gpt5nano
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/rc_gpt5nano_%j.out
#SBATCH --error=slurm_logs/rc_gpt5nano_%j.err

set -eo pipefail

# Source API key (disable nounset temporarily for ~/.bashrc compatibility)
set +u
source ~/.bashrc
set -u

echo "=== RC Scaffold: GPT-5-nano HMMT Feb 2026 ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "API key set: $([ -n "${OPENAI_API_KEY:-}" ] && echo 'yes' || echo 'no')"
echo ""

python rc_scaffold.py \
    --output model_outputs/predictions_rc.jsonl \
    --model gpt-5-nano \
    --max-completion-tokens 20000 \
    --max-summarization-tokens 4000 \
    --temperature 1 \
    --max-steps 3 \
    --n 1 \
    --poll-interval 30 \
    --run

echo ""
echo "--- Grading results ---"
python grade.py \
    --predictions model_outputs/predictions_rc.jsonl \
    --workers 8 \
    --verbose

echo ""
echo "=== Job completed at $(date) ==="
