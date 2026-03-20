# autoscaffold

This is an experiment to have an LLM scaffold solve competition math problems autonomously.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar20`). The branch `autoscaffold/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoscaffold/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context and constraints.
   - `grade.py` — fixed grading logic. Do not modify.
   - `query_openai_batch.py` — the file you modify. Scaffold logic, prompting, batching, aggregation.
4. **Verify API key**: Check that `OPENAI_API_KEY` is set in the environment.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment queries GPT-5-nano on 33 HMMT February 2026 competition math problems via the OpenAI Batch API. You launch it as:

```bash
python query_openai_batch.py \
    --output /scratch/dkhasha1/tli104/autoscaffold/generations/predictions_<name>.jsonl \
    --model gpt-5-nano \
    --temperature 1 \
    --run > /scratch/dkhasha1/tli104/autoscaffold/generations/query_<name>.log 2>&1
```

Then grade:

```bash
python grade.py \
    --predictions /scratch/dkhasha1/tli104/autoscaffold/generations/predictions_<name>.jsonl \
    --workers 8 \
    --verbose > /scratch/dkhasha1/tli104/autoscaffold/generations/grade_<name>.log 2>&1
```

Redirect everything — do NOT let output flood your context. Extract what you need:

```bash
grep "Accuracy:" /scratch/dkhasha1/tli104/autoscaffold/generations/grade_<name>.log
grep "Total:" /scratch/dkhasha1/tli104/autoscaffold/generations/query_<name>.log
```

The first line gives accuracy (e.g. `Accuracy: 18/33 (54.5%)`). The second gives total token usage (e.g. `Total:      485,122`).

**What you CAN do:**
- Modify `query_openai_batch.py` — this is the only file you edit. Everything is fair game: prompting strategy, number of samples, voting/aggregation logic, multi-round verification, answer extraction, token allocation.
- Modify `submit_query_openai.sh` if needed for SLURM job configuration.

**What you CANNOT do:**
- Modify `grade.py`. It is read-only. It contains the fixed grading logic using `math_verify`.
- Modify `query_model.py`, `README.md`, or `program.md`.
- Install new packages or add dependencies. You can only use what's already installed.
- Switch models. Only `gpt-5-nano` is allowed.
- Change the temperature. It must always be 1 — this is the only supported value for GPT-5-nano. When modifying `query_openai_batch.py`, never change the temperature value.

**The goal is simple: get the highest accuracy on HMMT Feb 2026.** The token budget is fixed at 2M total tokens per run (prompt + completion, across all problems, all samples, all rounds). No per-question cap. Everything is fair game: change the prompting strategy, the number of samples, add voting, add verification rounds, change answer extraction. The only constraint is that total token usage stays under 2M.

**Token budget arithmetic**: At the default settings, a single-pass run (1 sample per problem, ~32K max completion tokens) uses ~500K tokens. This means you can afford roughly K=3-4 samples per problem at full completion length, or more samples with shorter `max_completion_tokens`. Plan your budget before implementing.

**Simplicity criterion**: All else being equal, simpler is better. A 1/33 improvement from a clean approach is worth keeping. A 1/33 improvement from 200 lines of fragile multi-step logic is probably not. Conversely, removing complexity while maintaining accuracy is a win — that's a simplification you should keep.

**The first run**: Your very first run should always be to establish the baseline — run the scaffold as-is.

## Output format

The grader expects a JSONL file with one JSON object per line:

```json
{"problem_idx": 1, "model_answer": "The answer is $\\boxed{42}$"}
```

Required fields: `problem_idx` (integer, 1-33) and `model_answer` (string containing the answer, ideally with `\boxed{}`). Every problem must have exactly one final prediction line.

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	accuracy	tokens_used	status	description
```

1. git commit hash (short, 7 chars)
2. accuracy achieved (e.g. 18/33)
3. total tokens used (e.g. 1847000)
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	accuracy	tokens_used	status	description
a1b2c3d	18/33	485000	keep	baseline
b2c3d4e	21/33	1847000	keep	majority vote K=5
c3d4e5f	17/33	1200000	discard	aggressive prompt rewrite
d4e5f6g	0/33	0	crash	batch API timeout
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoscaffold/mar20`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on.
2. Modify `query_openai_batch.py` with an experimental idea.
3. git commit.
4. Run the experiment (redirect output to log file):
   ```bash
   python query_openai_batch.py --output /scratch/dkhasha1/tli104/autoscaffold/generations/predictions_<name>.jsonl --model gpt-5-nano --temperature 1 --run > /scratch/dkhasha1/tli104/autoscaffold/generations/query_<name>.log 2>&1
   ```
5. **Validate the run.** Check for failures before grading:
   - If the log contains "Batch ended with status: failed" or the predictions file doesn't exist, the run crashed. Run `tail -n 50 <log>` to diagnose.
   - Check the predictions file has 33 lines: `wc -l < predictions_<name>.jsonl`. If fewer than 33, the run is incomplete.
6. Grade the results (redirect output to log file):
   ```bash
   python grade.py --predictions /scratch/dkhasha1/tli104/autoscaffold/generations/predictions_<name>.jsonl --workers 8 --verbose > /scratch/dkhasha1/tli104/autoscaffold/generations/grade_<name>.log 2>&1
   ```
7. Extract results:
   ```bash
   grep "Accuracy:" /scratch/dkhasha1/tli104/autoscaffold/generations/grade_<name>.log
   grep "Total:" /scratch/dkhasha1/tli104/autoscaffold/generations/query_<name>.log
   ```
8. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git).
9. If accuracy improved (higher), you "advance" the branch, keeping the git commit.
10. If accuracy is equal: keep the change only if it uses meaningfully fewer tokens (leaving budget for future improvements). Otherwise, discard.
11. If accuracy is worse, `git reset --hard HEAD~1` back to where you started.
12. **Analyze failures.** After each experiment, read the grade log to see which specific problems failed and what the model produced. Look for patterns: are failures concentrated in certain problem types? Is the model producing malformed answers? Is it running out of tokens on hard problems? Use this analysis to guide your next experiment.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Batch turnaround**: Each experiment takes ~10-20 minutes for the batch to complete. Be patient — poll every 30 seconds. If a batch has not completed after 45 minutes, cancel it via the API, log "crash" as the status, discard, and move on.

**Crashes**: If a run crashes or the batch fails, use your judgment: If it's something easy to fix (e.g. a formatting error, a bad prompt), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the grading logic for edge cases, analyze which problems are failing and why, try combining previous near-misses, try more radical scaffold changes. The loop runs until the human interrupts you, period.

## Possible strategies

Some ideas that may help (not prescriptive — use your judgment):

- **Majority voting (best-of-N).** Generate K solutions per problem, extract `\boxed{}` answers, take the most frequent.
- **Prompt engineering.** Modify system/user prompts to reduce common failure modes.
- **Self-verification.** Multi-round: generate solutions, then ask the model to verify/critique/correct.
- **Model-aggregated consensus.** Generate N solutions, then ask the model to pick the best answer.
- **Adaptive token allocation.** Spend more tokens on harder problem types, fewer on easier ones.
- **Answer extraction improvements.** Better handling of `\boxed{}` extraction edge cases.
- **Hybrid approaches.** Combine any of the above.

## Results directory

All outputs go to:
```
/scratch/dkhasha1/tli104/autoscaffold/generations/
```
Use descriptive filenames. Never overwrite a previous experiment's output — always use a new filename. When starting a new experiment, always use a fresh `--output` path to avoid batch state collisions from previous runs.

## Key files for reference

- The grader uses `math_verify` to parse and verify answers. It handles `\boxed{}` extraction, LaTeX parsing, and numeric comparison. See `grade.py` for the exact logic.
- The dataset is `MathArena/hmmt_feb_2026` on HuggingFace (33 problems, split="train").
- Problem types: Algebra (5), Combinatorics (12), Geometry (12), Number Theory (8). Geometry and Combinatorics have the most room for improvement.
