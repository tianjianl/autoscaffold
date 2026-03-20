# Agent Instructions — autoscaffold

## Goal

Maximize accuracy on 33 HMMT February 2026 competition math problems using GPT-5-nano, within a 2M total token budget per run. The current baseline is 18/33 (54.5%). Your job is to beat it by modifying the scaffolding code.

## What you can modify

- **`query_openai_batch.py`** — The main scaffold. All strategy logic goes here: prompting, sampling, voting, verification, aggregation, answer extraction.
- **`submit_query_openai.sh`** — SLURM job script. Adjust flags, arguments, or pipeline steps.

## What you cannot modify

- **`grade.py`** — Grading logic. Read-only. Your outputs must be compatible with it.
- **`query_model.py`** — OpenRouter reference script. Not part of the pipeline.
- **`README.md`** — Do not modify.
- **`program.md`** — Do not modify.

## Experiment loop

Repeat the following cycle:

1. **Plan a change.** Pick one scaffold modification to test. Write a clear hypothesis (e.g., "majority vote with K=5 should improve accuracy by reducing random errors").

2. **Implement the change** in `query_openai_batch.py` (and `submit_query_openai.sh` if needed).

3. **Commit the change.**
   ```bash
   git add query_openai_batch.py submit_query_openai.sh
   git commit -m "<concise description of what changed>"
   ```

4. **Run the scaffold.**
   ```bash
   python query_openai_batch.py \
       --output /scratch/dkhasha1/tli104/autoscaffold/generations/predictions_<experiment_name>.jsonl \
       --model gpt-5-nano \
       --temperature 1 \
       --run
   ```
   Use a descriptive filename for each experiment (e.g., `predictions_majority_k5.jsonl`, `predictions_verify_2round.jsonl`).

5. **Grade the results.**
   ```bash
   python grade.py \
       --predictions /scratch/dkhasha1/tli104/autoscaffold/generations/predictions_<experiment_name>.jsonl \
       --workers 8 \
       --verbose
   ```
   Parse the output for `Accuracy: X/33` and the per-type breakdown.

6. **Decide: keep or revert.**
   - If accuracy improved: keep the commit. Record the result.
   - If accuracy stayed the same or got worse: revert.
     ```bash
     git reset --hard HEAD~1
     ```

7. **Log the result** in `results.tsv` (untracked, tab-separated). Columns:
   ```
   commit	accuracy	tokens_used	status	description
   ```
   Example row:
   ```
   a1b2c3d	21/33	1,847,000	keep	majority vote K=5 with frequency-weighted selection
   ```

## Output format

The grader expects a JSONL file with one JSON object per line:
```json
{"problem_idx": 1, "model_answer": "The answer is $\\boxed{42}$"}
```
Required fields: `problem_idx` (integer, 1-33) and `model_answer` (string containing the answer, ideally with `\boxed{}`). Every problem must have exactly one final prediction line.

## Constraints

- **2M total token budget** per run — prompt + completion tokens across all problems, all samples, all rounds combined. No per-question cap, but stay under 2M total. Track token usage from the batch API response `usage` fields.
- **Temperature**: Always 1. This is the only supported value for GPT-5-nano.
- **Model**: `gpt-5-nano`. Do not switch models.
- **No new dependencies.** Only use packages already installed.
- **Batch API only.** All GPT-5-nano calls go through the OpenAI Batch API. You can submit multiple batches per experiment if your strategy requires multiple rounds.

## Strategies to explore

Try these in roughly this order, from simplest to most complex:

1. **Majority voting (best-of-N).** Generate K solutions per problem (start with K=3, then K=5, K=8). Extract `\boxed{}` answers. Take the most frequent answer. This is the single highest-value change — implement it first.

2. **Prompt engineering.** Experiment with the system prompt and user prompt. Try adding specific instructions for common failure modes (e.g., "double-check your algebra", "consider edge cases", "verify your answer by substitution"). Try few-shot examples if token budget allows.

3. **Self-verification.** Two-round approach: first round generates solutions, second round asks the model to verify each solution and confirm or correct the answer. Budget: ~1M tokens per round.

4. **Model-aggregated consensus.** Generate N solutions, then in a second batch, present all N solutions to the model and ask it to select the best answer. More expensive per problem but can outperform simple majority vote.

5. **Adaptive token allocation.** Allocate more samples to harder problem types (geometry, combinatorics) and fewer to easier ones (number theory). Use per-type accuracy from previous runs to guide allocation.

6. **Answer extraction improvements.** Improve how `\boxed{}` answers are extracted from model outputs. Handle edge cases: multiple `\boxed{}` in one response, malformed LaTeX, answers stated in prose without `\boxed{}`.

7. **Hybrid strategies.** Combine majority voting with self-verification: generate K solutions, have the model verify each, then vote among verified answers.

## Multi-round batch workflow

For strategies requiring multiple rounds (verification, aggregation), you will need to:

1. Submit the first batch and wait for results.
2. Parse the first-round outputs.
3. Construct second-round prompts using first-round results.
4. Submit a second batch and wait.
5. Parse and aggregate into final predictions.

Implement this as sequential stages within `query_openai_batch.py`. Each batch call uses the existing `upload_file` / `create_batch` / `wait_for_batch` / `download_results` functions. Track cumulative token usage across all batches to stay under 2M.

## Results directory

All outputs go to:
```
/scratch/dkhasha1/tli104/autoscaffold/generations/
```
Use descriptive filenames. Never overwrite a previous experiment's output — always use a new filename.

## Key files for reference

- The grader uses `math_verify` to parse and verify answers. It handles `\boxed{}` extraction, LaTeX parsing, and numeric comparison. See `grade.py` for the exact logic.
- The dataset is `MathArena/hmmt_feb_2026` on HuggingFace (33 problems, split="train").
- Problem types: Algebra (5), Combinatorics (12), Geometry (12), Number Theory (8). Geometry and Combinatorics have the most room for improvement.
