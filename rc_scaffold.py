#!/usr/bin/env python3
"""
RC (Reasoning Cache) scaffold for GPT-5-nano on HMMT Feb 2026 via OpenAI API.

Multi-step pipeline (T steps per problem):
  For each step t = 1..T:
    1. Reasoning: solve/re-solve conditioned on cached summary
    2. Summarization: compress reasoning into compact summary
  Final answer: \\boxed{} from last reasoning step (majority vote if N > 1)

Supports both direct concurrent API calls (default) and batch API (--batch).

Prompts from: context_engineering/rc (Reasoning Cache, 2026)

Usage:
    python rc_scaffold.py --output predictions_rc.jsonl --run
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from datasets import load_dataset
from math_verify import parse, verify

OPENAI_URL = "https://api.openai.com/v1"

# ---------------------------------------------------------------------------
# Prompts: exact match to context_engineering/rc/inference/prompts/
# These are Python .format() templates: {{ → {, }} → }
# ---------------------------------------------------------------------------

REASONING_PROMPT = """\
You are given a maths problem. You may also be given a summary of a previous attempt to solve it. This previous attempt may or may not be correct.

### PROBLEM
{problem}

### SUMMARY OF PREVIOUS ATTEMPT
{curr_summary}

### INSTRUCTIONS
If no summary of a previous attempt is provided, solve the problem from scratch.

If a summary of a previous attempt is provided, your task is to improve upon this attempt. You should rely on this summary to guide your thinking.
Some strategies you could use include:
- Verifying the previous solution.
- Proving the result in a different way.
- Finding alternative problem-solving strategies.
- Continuing from where the previous solution left off, assuming that the previous solution is incomplete.

Reason step-by-step and return your final answer in \\\\boxed{{}}."""

SUMMARIZATION_PROMPT = """\
You are given a maths problem and a candidate solution to it. You may also be given a summary of a previous candidate solution to the problem. If this is provided, you may assume that the current candidate solution was generated conditioned on the summary of the previous candidate solution.
Your task is to write a summary of the current candidate solution.

The new summary you generate should possess the following characteristics:
- It should provide a detailed overview of what occurred in the current candidate solution. This may include a summary of the high-level problem-solving strategy, a description of theorems used, verification attempts, calculations and logical deductions etc.
- It should summarize the current candidate solution in light of any previous summaries, if provided. We should be able to understand the relationship between the previous solution and the current solution by reading the summary. Make sure any important information contained in the existing summary is retained in the new one.
- It should be no more than two paragraph long and written in paragraph form, without headers or subheaders.
- It should be written in the first person, as if though it is being written by the person solving the problem.
- The candidate solution may not be complete. In this case, the summary should still attempt to summarize the partial solution.

IMPORTANT: Do not under any circumstances add any additional reasoning not contained in the latest reasoning step. Your task is only to summarize what is given to you.

### PROBLEM
{problem}

### EXISTING SUMMARY
{existing_summary}

### LATEST CANDIDATE SOLUTION
{reasoning}"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_boxed(text):
    """Extract the last \\boxed{...} answer from text, handling nested braces."""
    if not text:
        return None
    results = []
    start = 0
    while True:
        pos = text.find("\\boxed{", start)
        if pos == -1:
            break
        depth = 0
        i = pos + 7
        begin = i
        while i < len(text):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                if depth == 0:
                    results.append(text[begin:i])
                    break
                depth -= 1
            i += 1
        start = pos + 1
    return results[-1] if results else None


def majority_vote(samples_by_problem):
    """Majority vote across N samples per problem using math_verify grouping.

    samples_by_problem: {problem_idx: [(content, usage), ...]}
    Returns: {problem_idx: best_content}
    """
    results = {}
    for problem_idx in sorted(samples_by_problem.keys()):
        samples = samples_by_problem[problem_idx]
        answer_data = []
        for content, usage in samples:
            if not content:
                continue
            boxed = extract_boxed(content)
            if boxed is not None:
                rt = (
                    (usage.get("completion_tokens_details") or {})
                    .get("reasoning_tokens", 0)
                )
                answer_data.append((boxed, content, rt))

        if not answer_data:
            # No boxed answer; use raw content as fallback
            for content, usage in samples:
                if content:
                    results[problem_idx] = content
                    break
            else:
                results[problem_idx] = ""
            continue

        # Group semantically equivalent answers
        groups = []  # (rep_boxed, best_content, count, max_rt)
        for boxed, content, rt in answer_data:
            matched = False
            try:
                parsed_new = parse(f"${boxed}$")
            except Exception:
                parsed_new = None

            if parsed_new:
                for i, (rep_boxed, rep_content, count, max_rt) in enumerate(groups):
                    try:
                        rep_parsed = parse(f"${rep_boxed}$")
                        if rep_parsed and verify(rep_parsed, parsed_new):
                            if rt > max_rt:
                                groups[i] = (rep_boxed, content, count + 1, rt)
                            else:
                                groups[i] = (rep_boxed, rep_content, count + 1, max_rt)
                            matched = True
                            break
                    except Exception:
                        continue

            if not matched:
                groups.append((boxed, content, 1, rt))

        groups.sort(key=lambda g: (g[2], g[3]), reverse=True)
        best_boxed, best_content, _, _ = groups[0]
        results[problem_idx] = best_content or f"\\boxed{{{best_boxed}}}"

    return results


# ---------------------------------------------------------------------------
# Batch API helpers
# ---------------------------------------------------------------------------

def upload_file(api_key, filepath):
    """Upload batch input file to OpenAI."""
    headers = {"Authorization": f"Bearer {api_key}"}
    with open(filepath, "rb") as f:
        resp = requests.post(
            f"{OPENAI_URL}/files",
            headers=headers,
            files={"file": (os.path.basename(filepath), f, "application/jsonl")},
            data={"purpose": "batch"},
            timeout=120,
        )
    resp.raise_for_status()
    file_id = resp.json()["id"]
    print(f"    Uploaded file: {file_id}")
    return file_id


def create_batch(api_key, file_id):
    """Create a batch job."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    resp = requests.post(
        f"{OPENAI_URL}/batches",
        headers=headers,
        json={
            "input_file_id": file_id,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        },
        timeout=60,
    )
    resp.raise_for_status()
    batch = resp.json()
    batch_id = batch["id"]
    print(f"    Created batch: {batch_id} (status: {batch['status']})")
    return batch_id


def wait_for_batch(api_key, batch_id, poll_interval=30):
    """Poll until batch completes."""
    print(f"    Waiting for batch {batch_id}...")
    while True:
        try:
            headers = {"Authorization": f"Bearer {api_key}"}
            resp = requests.get(
                f"{OPENAI_URL}/batches/{batch_id}",
                headers=headers, timeout=30,
            )
            resp.raise_for_status()
            batch = resp.json()
        except requests.exceptions.RequestException as e:
            print(f"      Poll error (will retry): {e}")
            time.sleep(poll_interval)
            continue

        status = batch["status"]
        counts = batch.get("request_counts", {})
        completed = counts.get("completed", 0)
        failed = counts.get("failed", 0)
        total = counts.get("total", 0)
        print(f"      {status} | {completed}/{total} done, {failed} failed")

        if status in ("completed", "failed", "expired", "cancelled"):
            return batch
        time.sleep(poll_interval)


def download_batch_output(api_key, batch, label, output_dir):
    """Download raw batch output. Returns raw text or None."""
    output_file_id = batch.get("output_file_id")
    error_file_id = batch.get("error_file_id")
    headers = {"Authorization": f"Bearer {api_key}"}

    if error_file_id:
        resp = requests.get(
            f"{OPENAI_URL}/files/{error_file_id}/content",
            headers=headers, timeout=120,
        )
        if resp.ok:
            error_path = os.path.join(output_dir, f"{label}_errors.jsonl")
            with open(error_path, "w") as f:
                f.write(resp.text)
            print(f"    Errors saved: {error_path}")

    if not output_file_id:
        print("    No output file available")
        return None

    resp = requests.get(
        f"{OPENAI_URL}/files/{output_file_id}/content",
        headers=headers, timeout=300,
    )
    resp.raise_for_status()
    raw_path = os.path.join(output_dir, f"{label}_raw.jsonl")
    with open(raw_path, "w") as f:
        f.write(resp.text)
    print(f"    Raw output saved: {raw_path}")
    return resp.text


def submit_and_wait(api_key, batch_input_path, label, output_dir, poll_interval,
                    state_path):
    """Upload → create → wait → download. Returns raw text or None."""
    file_id = upload_file(api_key, batch_input_path)
    batch_id = create_batch(api_key, file_id)

    # Persist batch ID for crash recovery
    state = {}
    if os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)
    state[label] = {"batch_id": batch_id, "file_id": file_id}
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)

    batch = wait_for_batch(api_key, batch_id, poll_interval)
    if batch["status"] != "completed":
        print(f"    Batch {label} ended: {batch['status']}")
        if batch["status"] == "failed":
            for err in batch.get("errors", {}).get("data", [])[:5]:
                print(f"      {err}")
        return None

    return download_batch_output(api_key, batch, label, output_dir)


def call_api_direct(api_key, custom_id, body, max_retries=5):
    """Make a single direct chat completion call with retries."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"{OPENAI_URL}/chat/completions",
                headers=headers,
                json=body,
                timeout=600,
            )
            if resp.status_code == 429 or resp.status_code >= 500:
                wait = min(2 ** attempt * 5, 60)
                print(f"    {custom_id}: {resp.status_code}, retry in {wait}s")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            return custom_id, data, None
        except requests.exceptions.RequestException as e:
            wait = min(2 ** attempt * 5, 60)
            print(f"    {custom_id}: error {e}, retry in {wait}s")
            time.sleep(wait)
    return custom_id, None, f"failed after {max_retries} retries"


def submit_direct(api_key, batch_input_path, label, output_dir, workers=10):
    """Send requests concurrently via direct API. Returns raw JSONL text."""
    # Read batch input
    reqs = []
    with open(batch_input_path) as f:
        for line in f:
            if line.strip():
                reqs.append(json.loads(line))

    print(f"    Sending {len(reqs)} requests ({workers} concurrent)...")
    results = []
    completed = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for req in reqs:
            fut = executor.submit(
                call_api_direct, api_key, req["custom_id"], req["body"]
            )
            futures[fut] = req["custom_id"]

        for fut in as_completed(futures):
            custom_id, data, error = fut.result()
            completed += 1
            if error:
                results.append({
                    "custom_id": custom_id,
                    "response": {"body": {}},
                    "error": error,
                })
            else:
                results.append({
                    "custom_id": custom_id,
                    "response": {"body": data},
                    "error": None,
                })
            if completed % max(1, len(reqs) // 5) == 0 or completed == len(reqs):
                print(f"      {completed}/{len(reqs)} done")

    # Save raw output
    raw_path = os.path.join(output_dir, f"{label}_raw.jsonl")
    lines = []
    for r in results:
        lines.append(json.dumps(r))
    raw_text = "\n".join(lines)
    with open(raw_path, "w") as f:
        f.write(raw_text)
    print(f"    Raw output saved: {raw_path}")
    return raw_text


def parse_batch_results(raw_text):
    """Parse raw batch JSONL into {custom_id: (content, usage, error)}."""
    results = {}
    totals = {"prompt": 0, "completion": 0, "reasoning": 0}

    for line in raw_text.strip().split("\n"):
        if not line.strip():
            continue
        obj = json.loads(line)
        custom_id = obj["custom_id"]
        response = obj.get("response", {})
        body = response.get("body", {})
        error = obj.get("error")

        if error:
            results[custom_id] = ("", {}, str(error))
            continue

        choices = body.get("choices", [])
        usage = body.get("usage", {})
        totals["prompt"] += usage.get("prompt_tokens", 0)
        totals["completion"] += usage.get("completion_tokens", 0)
        totals["reasoning"] += (
            (usage.get("completion_tokens_details") or {})
            .get("reasoning_tokens", 0)
        )

        content = ""
        if choices:
            content = choices[0].get("message", {}).get("content") or ""
        results[custom_id] = (content, usage, None)

    return results, totals


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RC scaffold for GPT-5-nano via OpenAI Batch API"
    )
    parser.add_argument("--output", "-o", default="predictions_rc.jsonl",
                        help="Output JSONL path (default: predictions_rc.jsonl)")
    parser.add_argument("--model", default="gpt-5-nano", help="Model ID")
    parser.add_argument("--max-completion-tokens", type=int, default=20000,
                        help="Max tokens per reasoning call (default: 20000)")
    parser.add_argument("--max-summarization-tokens", type=int, default=4000,
                        help="Max tokens per summarization call (default: 4000)")
    parser.add_argument("--temperature", type=float, default=1,
                        help="Sampling temperature (default: 1)")
    parser.add_argument("--dataset", default="MathArena/hmmt_feb_2026",
                        help="HuggingFace dataset")
    parser.add_argument("--api-key", default=None, help="OpenAI API key")
    parser.add_argument("--poll-interval", type=int, default=30,
                        help="Batch poll interval in seconds (default: 30)")
    parser.add_argument("--max-steps", type=int, default=3,
                        help="Reasoning-summarization iterations T (default: 3)")
    parser.add_argument("--n", type=int, default=1,
                        help="Samples per problem N (default: 1)")
    parser.add_argument("--workers", type=int, default=10,
                        help="Concurrent API workers for direct mode (default: 10)")
    parser.add_argument("--batch", action="store_true",
                        help="Use Batch API instead of direct concurrent calls")
    parser.add_argument("--run", action="store_true",
                        help="Run full RC pipeline")
    parser.add_argument("--submit", action="store_true",
                        help="Submit first reasoning batch only (batch mode)")

    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: set OPENAI_API_KEY or pass --api-key", file=sys.stderr)
        sys.exit(1)

    if not any([args.run, args.submit]):
        print("Specify --run (full pipeline) or --submit (first batch only)")
        sys.exit(1)

    # Load dataset
    ds = load_dataset(args.dataset, split="train")
    problems = []
    for row in ds:
        problems.append({
            "problem_idx": row["problem_idx"],
            "problem": row["problem"],
        })
    print(f"Loaded {len(problems)} problems from {args.dataset}")

    # Output paths
    output_dir = os.path.dirname(os.path.abspath(args.output)) or "."
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.output))[0]
    state_path = os.path.join(output_dir, f"{base}_batch_state.json")

    N = args.n
    T = args.max_steps
    n_problems = len(problems)
    n_states = n_problems * N

    # Initialize per-(problem, sample) state
    states = {}
    for p in problems:
        idx = p["problem_idx"]
        for s in range(N):
            states[(idx, s)] = {
                "problem": p["problem"],
                "curr_summary": "",
                "curr_reasoning": "",
                "reasoning_store": [],
                "summarization_store": [],
            }

    total_tokens = {"prompt": 0, "completion": 0, "reasoning": 0}

    print(f"\n{'=' * 60}")
    print(f"  RC Scaffold: T={T} steps, N={N} samples/problem")
    print(f"  Model: {args.model}, Temp: {args.temperature}")
    print(f"  Reasoning max tokens: {args.max_completion_tokens}")
    print(f"  Summarization max tokens: {args.max_summarization_tokens}")
    print(f"  Total batches: {T * 2} ({T} reasoning + {T} summarization)")
    print(f"  Total requests: {n_states * T * 2}")
    print(f"{'=' * 60}")

    for step in range(T):
        print(f"\n{'=' * 60}")
        print(f"  STEP {step + 1}/{T}")
        print(f"{'=' * 60}")

        # --- Phase A: Reasoning ---
        print(f"\n  [Reasoning] {n_states} requests")
        reason_input = os.path.join(
            output_dir, f"{base}_step{step + 1}_reasoning_input.jsonl"
        )
        with open(reason_input, "w") as f:
            for (idx, s) in sorted(states.keys()):
                st = states[(idx, s)]
                filled = REASONING_PROMPT.format(
                    problem=st["problem"],
                    curr_summary=st["curr_summary"],
                )
                request = {
                    "custom_id": f"rc-{idx}-s{s}-step{step + 1}-reason",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": args.model,
                        "messages": [
                            {"role": "user", "content": filled},
                        ],
                        "max_completion_tokens": args.max_completion_tokens,
                        "temperature": args.temperature,
                    },
                }
                f.write(json.dumps(request) + "\n")

        if args.submit and step == 0:
            # Submit-only mode: just submit first reasoning batch
            file_id = upload_file(api_key, reason_input)
            batch_id = create_batch(api_key, file_id)
            state_data = {}
            if os.path.exists(state_path):
                with open(state_path) as f:
                    state_data = json.load(f)
            state_data["step1_reasoning"] = {
                "batch_id": batch_id, "file_id": file_id
            }
            with open(state_path, "w") as f:
                json.dump(state_data, f, indent=2)
            print(f"\n  Batch submitted: {batch_id}")
            print(f"  State saved: {state_path}")
            print("  Use --run for full pipeline.")
            return

        label = f"{base}_step{step + 1}_reasoning"
        if args.batch:
            raw_text = submit_and_wait(
                api_key, reason_input, label, output_dir,
                args.poll_interval, state_path,
            )
        else:
            raw_text = submit_direct(
                api_key, reason_input, label, output_dir, args.workers,
            )
        if not raw_text:
            print(f"  FATAL: Reasoning batch failed at step {step + 1}")
            sys.exit(1)

        results, tokens = parse_batch_results(raw_text)
        total_tokens["prompt"] += tokens["prompt"]
        total_tokens["completion"] += tokens["completion"]
        total_tokens["reasoning"] += tokens["reasoning"]

        for (idx, s), st in states.items():
            cid = f"rc-{idx}-s{s}-step{step + 1}-reason"
            content, usage, error = results.get(cid, ("", {}, "missing"))
            if error:
                print(f"    WARNING: problem {idx} sample {s}: {error}")
            st["curr_reasoning"] = content
            st["reasoning_store"].append(content)

        r_total = tokens["prompt"] + tokens["completion"]
        print(f"  Reasoning: {r_total:,} tokens")

        # --- Phase B: Summarization ---
        print(f"\n  [Summarization] {n_states} requests")
        summ_input = os.path.join(
            output_dir, f"{base}_step{step + 1}_summarization_input.jsonl"
        )
        with open(summ_input, "w") as f:
            for (idx, s) in sorted(states.keys()):
                st = states[(idx, s)]
                filled = SUMMARIZATION_PROMPT.format(
                    problem=st["problem"],
                    existing_summary=st["curr_summary"],
                    reasoning=st["curr_reasoning"],
                )
                request = {
                    "custom_id": f"rc-{idx}-s{s}-step{step + 1}-summ",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": args.model,
                        "messages": [
                            {"role": "user", "content": filled},
                        ],
                        "max_completion_tokens": args.max_summarization_tokens,
                        "temperature": args.temperature,
                    },
                }
                f.write(json.dumps(request) + "\n")

        label = f"{base}_step{step + 1}_summarization"
        if args.batch:
            raw_text = submit_and_wait(
                api_key, summ_input, label, output_dir,
                args.poll_interval, state_path,
            )
        else:
            raw_text = submit_direct(
                api_key, summ_input, label, output_dir, args.workers,
            )
        if not raw_text:
            print(f"  WARNING: Summarization failed at step {step + 1}")
            print("  Continuing with current summaries")
            continue

        results, tokens = parse_batch_results(raw_text)
        total_tokens["prompt"] += tokens["prompt"]
        total_tokens["completion"] += tokens["completion"]
        total_tokens["reasoning"] += tokens["reasoning"]

        for (idx, s), st in states.items():
            cid = f"rc-{idx}-s{s}-step{step + 1}-summ"
            content, usage, error = results.get(cid, ("", {}, "missing"))
            if error:
                print(f"    WARNING: problem {idx} sample {s} summ: {error}")
            st["curr_summary"] = content
            st["summarization_store"].append(content)

        s_total = tokens["prompt"] + tokens["completion"]
        cumulative = total_tokens["prompt"] + total_tokens["completion"]
        print(f"  Summarization: {s_total:,} tokens")
        print(f"  Step {step + 1} total: {r_total + s_total:,} | "
              f"Cumulative: {cumulative:,}")

    # --- Extract final answers ---
    print(f"\n{'=' * 60}")
    print(f"  EXTRACTING FINAL ANSWERS")
    print(f"{'=' * 60}")

    # Group last reasoning by problem
    final_by_problem = {}
    for (idx, s), st in states.items():
        final_by_problem.setdefault(idx, [])
        last = st["reasoning_store"][-1] if st["reasoning_store"] else ""
        final_by_problem[idx].append((last, {}))

    if N > 1:
        final_results = majority_vote(final_by_problem)
    else:
        final_results = {}
        for idx, samples in final_by_problem.items():
            final_results[idx] = samples[0][0] if samples else ""

    # Write predictions (grade.py compatible)
    predictions = []
    for idx in sorted(final_results.keys()):
        predictions.append({
            "problem_idx": idx,
            "model_answer": final_results[idx],
        })

    with open(args.output, "w") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")

    grand_total = total_tokens["prompt"] + total_tokens["completion"]
    print(f"\n{'=' * 60}")
    print(f"  {len(predictions)} predictions → {args.output}")
    print(f"  Tokens: {grand_total:,} "
          f"(prompt {total_tokens['prompt']:,} + "
          f"completion {total_tokens['completion']:,}, "
          f"reasoning {total_tokens['reasoning']:,})")
    print(f"{'=' * 60}")

    # Save detailed state for analysis
    state_output = os.path.join(output_dir, f"{base}_states.json")
    state_data = []
    for (idx, s) in sorted(states.keys()):
        st = states[(idx, s)]
        state_data.append({
            "problem_idx": idx,
            "sample_id": s,
            "reasoning_store": st["reasoning_store"],
            "summarization_store": st["summarization_store"],
        })
    with open(state_output, "w") as f:
        json.dump(state_data, f, indent=2)
    print(f"  States saved: {state_output}")


if __name__ == "__main__":
    main()
