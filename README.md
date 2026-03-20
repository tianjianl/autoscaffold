# autoscaffold

Autonomous scaffolding loop for solving [HMMT February 2026](https://www.hmmt.org/) competition math problems using GPT-5-nano via the OpenAI Batch API.

The agent iterates on the scaffolding code to maximize accuracy on 33 competition math problems within a **2M total token budget**.

## Current results

| Model | Accuracy | Algebra | Combinatorics | Geometry | Number Theory |
|-------|----------|---------|---------------|----------|---------------|
| GPT-5-nano | 18/33 (54.5%) | 3/5 (60%) | 5/12 (42%) | 5/12 (42%) | 6/8 (75%) |
| Qwen3.5-9B | 16/33 (48.5%) | 4/5 (80%) | 5/12 (42%) | 4/12 (33%) | 4/8 (50%) |

## What the agent CAN modify

- **`query_openai_batch.py`** — The main scaffold. The agent should modify this file to implement scaffolding strategies (majority voting, self-verification, prompt engineering, multi-round solving, answer aggregation, etc.). All scaffold logic lives here.
- **`submit_query_openai.sh`** — SLURM job script. Adjust flags, arguments, or pipeline steps as needed.

## What the agent CANNOT modify

- **`grade.py`** — Read-only. Grading logic is fixed. The agent must produce outputs compatible with this grader.
- **`query_model.py`** — Reference only. OpenRouter query script, not used in the main pipeline.
- **`README.md`** — Do not modify.

## How to run verification and get results

### Step 1: Run the scaffold

```bash
# Run the full pipeline (submit batch → wait → download → grade)
python query_openai_batch.py \
    --output /scratch/dkhasha1/tli104/autoscaffold/generations/predictions.jsonl \
    --model gpt-5-nano \
    --temperature 1 \
    --run
```

Or submit via SLURM:

```bash
sbatch submit_query_openai.sh
```

### Step 2: Grade the results

```bash
python grade.py \
    --predictions /scratch/dkhasha1/tli104/autoscaffold/generations/predictions.jsonl \
    --workers 8 \
    --verbose
```

This prints:
- Overall accuracy: `Accuracy: X/33 (Y%)`
- Per-problem pass/fail with gold and predicted answers
- Per-problem-type breakdown (Algebra, Combinatorics, Geometry, Number Theory)
- Detailed results saved to `predictions_results.json` alongside the predictions file

### Step 3: Check the output

The grader expects a JSONL file with one JSON object per line:
```json
{"problem_idx": 1, "model_answer": "The answer is $\\boxed{42}$"}
```

Required fields: `problem_idx` (1-33) and `model_answer` (string containing the answer, ideally with `\boxed{}`).

## Where results are saved

All generation outputs go to the scratch space:

```
/scratch/dkhasha1/tli104/autoscaffold/generations/
├── predictions.jsonl              — Final predictions for grading (one per problem)
├── predictions_results.json       — Detailed grading results with per-problem breakdown
├── predictions_batch_input.jsonl  — Raw batch input sent to OpenAI
├── predictions_batch_state.json   — Batch job state (file_id, batch_id) for resuming
├── predictions_raw.jsonl          — Raw batch output from OpenAI
├── predictions_errors.jsonl       — Any batch errors (if applicable)
└── *.out, *.err                   — SLURM job logs
```

The agent should use descriptive filenames when running multiple experiments, e.g.:
- `predictions_majority_k3.jsonl` for a 3-sample majority vote run
- `predictions_verify_2round.jsonl` for a 2-round verification run

## Key constraints

- **Total token budget**: 2M tokens per run (across all 33 problems, all rounds/samples combined). No per-question cap.
- **Model**: `gpt-5-nano` (temperature fixed at 1 — only supported value).
- **Metric**: Accuracy on HMMT Feb 2026 (33 problems, graded by `math_verify`).
- **Dataset**: [MathArena/hmmt_feb_2026](https://huggingface.co/datasets/MathArena/hmmt_feb_2026)

## Possible strategies

Some ideas that may help (not prescriptive):

- **Majority voting (best-of-N)**: Generate K solutions per problem, extract `\boxed{}` answers, take the most frequent.
- **Prompt engineering**: Modify system/user prompts to reduce common failure modes.
- **Self-verification**: Multi-round: generate solutions, then ask the model to verify/critique/correct.
- **Model-aggregated consensus**: Generate N solutions, then ask the model to pick the best answer.
- **Adaptive token allocation**: Spend more tokens on harder problem types, fewer on easier ones.
- **Hybrid approaches**: Combine any of the above.

## Project structure

```
query_openai_batch.py  — Main scaffold (MODIFY THIS)
grade.py               — Grading with math_verify (READ ONLY)
query_model.py         — OpenRouter queries (reference only)
submit_query_openai.sh — SLURM job script (modify as needed)
submit_query.sh        — SLURM job script for OpenRouter
```

## License

MIT
