#!/usr/bin/env python3
"""
Query GPT-5-nano on HMMT Feb 2026 via OpenAI Batch API.

Usage:
    python query_openai_batch.py --output predictions_gpt5nano.jsonl [--submit] [--poll] [--download]

Workflow:
    1. --submit:   Create batch JSONL, upload, and submit batch job
    2. --poll:     Check batch status (or wait with --wait)
    3. --download: Download results and convert to grade.py format

    Or just run with --run to do all three (submit, wait, download).
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter

import requests
from datasets import load_dataset
from math_verify import parse, verify

OPENAI_URL = "https://api.openai.com/v1"

# Number of samples per problem for majority voting
NUM_SAMPLES = 3

SYSTEM_PROMPT = """\
You are an expert competition mathematician solving HMMT February problems.

Solve the following problem step by step.

Important guidelines:
- Focus on key insights. For combinatorics: look for bijections, recursions, generating functions, or inclusion-exclusion. For geometry: consider coordinates, trigonometric identities, or projective methods.
- Double-check your arithmetic and your final answer before writing \\boxed{}.
- Verify edge cases and small examples when possible.
- You MUST present your final answer inside \\boxed{} at the end. Even if uncertain, give your best answer.
- Simplify fractions. Give exact values (not decimals).\
"""


def extract_boxed(text):
    """Extract the last \\boxed{...} answer from text, handling nested braces."""
    if not text:
        return None
    # Find all \boxed{ positions
    results = []
    start = 0
    while True:
        pos = text.find("\\boxed{", start)
        if pos == -1:
            break
        # Find matching closing brace
        depth = 0
        i = pos + 7  # skip past \boxed{
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
    if results:
        return results[-1]  # Return the last \boxed{} match
    return None


def normalize_answer(ans):
    """Normalize a LaTeX answer string for comparison during majority voting.

    Handles common formatting differences:
    - \\tfrac, \\dfrac -> \\frac
    - Whitespace normalization
    - a/b -> \\frac{a}{b} for simple fractions
    """
    if ans is None:
        return None
    s = ans.strip()
    # Remove leading/trailing whitespace variants (\\, and friends)
    s = s.strip('\\,').strip()
    # Normalize fraction commands
    s = s.replace("\\tfrac", "\\frac").replace("\\dfrac", "\\frac")
    # Normalize whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    # Convert simple "a/b" to "\frac{a}{b}" if no backslash commands present
    # Match patterns like "123/456" or "7\sqrt{3}/3"
    simple_frac = re.match(r'^([^/]+)/(\d+)$', s)
    if simple_frac and '\\frac' not in s:
        num, den = simple_frac.groups()
        s = f"\\frac{{{num.strip()}}}{{{den.strip()}}}"
    return s


def make_batch_jsonl(dataset_name, model, max_completion_tokens, temperature,
                     output_path):
    """Create the batch input JSONL file with NUM_SAMPLES requests per problem."""
    ds = load_dataset(dataset_name, split="train")
    print(f"Loaded {len(ds)} problems")

    n_requests = 0
    with open(output_path, "w") as f:
        for row in ds:
            idx = row["problem_idx"]
            for k in range(NUM_SAMPLES):
                request = {
                    "custom_id": f"problem-{idx}-sample-{k}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": row["problem"]},
                        ],
                        "max_completion_tokens": max_completion_tokens,
                        "temperature": temperature,
                    },
                }
                f.write(json.dumps(request) + "\n")
                n_requests += 1

    print(f"Wrote {n_requests} requests ({len(ds)} problems x {NUM_SAMPLES} samples) to {output_path}")
    return output_path


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
    file_obj = resp.json()
    file_id = file_obj["id"]
    print(f"Uploaded file: {file_id}")
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
    print(f"Created batch: {batch_id}")
    print(f"  Status: {batch['status']}")
    return batch_id


def check_batch(api_key, batch_id):
    """Check batch status."""
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = requests.get(
        f"{OPENAI_URL}/batches/{batch_id}",
        headers=headers,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def wait_for_batch(api_key, batch_id, poll_interval=30):
    """Poll until batch completes."""
    print(f"Waiting for batch {batch_id}...")
    while True:
        try:
            batch = check_batch(api_key, batch_id)
        except requests.exceptions.RequestException as e:
            print(f"  Poll error (will retry): {e}")
            time.sleep(poll_interval)
            continue
        status = batch["status"]
        counts = batch.get("request_counts", {})
        completed = counts.get("completed", 0)
        failed = counts.get("failed", 0)
        total = counts.get("total", 0)

        print(f"  Status: {status} | {completed}/{total} completed, "
              f"{failed} failed")

        if status in ("completed", "failed", "expired", "cancelled"):
            return batch

        time.sleep(poll_interval)


def download_results(api_key, batch, output_path):
    """Download batch results, apply majority voting, and convert to grade.py format."""
    output_file_id = batch.get("output_file_id")
    error_file_id = batch.get("error_file_id")

    if error_file_id:
        headers = {"Authorization": f"Bearer {api_key}"}
        resp = requests.get(
            f"{OPENAI_URL}/files/{error_file_id}/content",
            headers=headers,
            timeout=120,
        )
        if resp.ok:
            error_path = output_path.replace(".jsonl", "_errors.jsonl")
            with open(error_path, "w") as f:
                f.write(resp.text)
            print(f"Errors saved to: {error_path}")

    if not output_file_id:
        print("No output file available")
        return

    headers = {"Authorization": f"Bearer {api_key}"}
    resp = requests.get(
        f"{OPENAI_URL}/files/{output_file_id}/content",
        headers=headers,
        timeout=300,
    )
    resp.raise_for_status()

    # Save raw batch output
    raw_path = output_path.replace(".jsonl", "_raw.jsonl")
    with open(raw_path, "w") as f:
        f.write(resp.text)
    print(f"Raw output saved to: {raw_path}")

    # Collect all samples per problem
    samples = {}  # problem_idx -> list of (content, usage)
    total_prompt = 0
    total_completion = 0
    total_reasoning = 0

    for line in resp.text.strip().split("\n"):
        if not line.strip():
            continue
        obj = json.loads(line)
        custom_id = obj["custom_id"]  # "problem-N-sample-K"
        parts = custom_id.split("-")
        problem_idx = int(parts[1])

        response = obj.get("response", {})
        body = response.get("body", {})
        error = obj.get("error")

        if error:
            samples.setdefault(problem_idx, []).append(("", {}, str(error)))
            continue

        choices = body.get("choices", [])
        usage = body.get("usage", {})

        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        reasoning_tokens = (usage.get("completion_tokens_details") or {}
                           ).get("reasoning_tokens", 0)
        total_prompt += prompt_tokens
        total_completion += completion_tokens
        total_reasoning += reasoning_tokens

        if choices:
            message = choices[0].get("message", {})
            content = message.get("content") or ""
        else:
            content = ""

        samples.setdefault(problem_idx, []).append((content, usage, None))

    # Majority vote for each problem
    results = []
    for problem_idx in sorted(samples.keys()):
        problem_samples = samples[problem_idx]
        # Collect (raw_boxed, content, reasoning_tokens) for each sample
        answer_data = []
        for content, usage, error in problem_samples:
            if error or not content:
                continue
            boxed = extract_boxed(content)
            if boxed is not None:
                rt = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
                answer_data.append((boxed, content, rt))

        if answer_data:
            # Group equivalent answers using math_verify
            # Track total reasoning tokens per group as tiebreaker
            groups = []  # list of (representative_boxed, best_content, count, max_reasoning)
            for boxed, content, rt in answer_data:
                matched = False
                boxed_text = f"${boxed}$"
                try:
                    parsed_new = parse(boxed_text)
                except Exception:
                    parsed_new = None
                if parsed_new:
                    for i, (rep_boxed, rep_content, count, max_rt) in enumerate(groups):
                        try:
                            rep_parsed = parse(f"${rep_boxed}$")
                            if rep_parsed and verify(rep_parsed, parsed_new):
                                # Keep the content from the sample with most reasoning
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

            # Pick the group with the most votes; break ties by reasoning tokens
            groups.sort(key=lambda g: (g[2], g[3]), reverse=True)
            best_boxed, best_content, best_count, _ = groups[0]
            model_answer = best_content or f"\\boxed{{{best_boxed}}}"
        else:
            # No valid boxed answers from any sample — use first non-empty response
            model_answer = ""
            for content, usage, error in problem_samples:
                if content:
                    model_answer = content
                    break

        results.append({
            "problem_idx": problem_idx,
            "model_answer": model_answer,
            "num_samples": len(problem_samples),
            "num_valid_answers": len(answer_data) if answer_data else 0,
            "error": None,
        })

    # Sort by problem index and write
    results.sort(key=lambda r: r["problem_idx"])
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    n_success = sum(1 for r in results if r.get("model_answer") and not r.get("error"))
    n_error = sum(1 for r in results if r.get("error"))

    print(f"\n{'=' * 60}")
    print(f"  Results: {n_success} success, {n_error} errors out of {len(results)}")
    print(f"  Token usage:")
    print(f"    Prompt:     {total_prompt:>10,}")
    print(f"    Completion: {total_completion:>10,}")
    if total_reasoning:
        print(f"    (Reasoning: {total_reasoning:>9,})")
    print(f"    Total:      {total_prompt + total_completion:>10,}")
    print(f"  Output: {output_path}")
    print(f"{'=' * 60}")


def save_state(state_path, **kwargs):
    """Save batch state for resuming."""
    state = {}
    if os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)
    state.update(kwargs)
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)


def load_state(state_path):
    """Load batch state."""
    if os.path.exists(state_path):
        with open(state_path) as f:
            return json.load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(
        description="Query GPT-5-nano via OpenAI Batch API"
    )
    parser.add_argument(
        "--output", "-o", default="predictions_gpt5nano.jsonl",
        help="Output JSONL path (default: predictions_gpt5nano.jsonl)"
    )
    parser.add_argument(
        "--model", default="gpt-5-nano",
        help="OpenAI model ID (default: gpt-5-nano)"
    )
    parser.add_argument(
        "--max-completion-tokens", type=int, default=32000,
        help="Max completion tokens per question (default: 32000)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature (default: 0.7)"
    )
    parser.add_argument(
        "--dataset", default="MathArena/hmmt_feb_2026",
        help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--api-key", default=None,
        help="OpenAI API key (default: $OPENAI_API_KEY)"
    )
    parser.add_argument(
        "--poll-interval", type=int, default=30,
        help="Poll interval in seconds (default: 30)"
    )

    # Actions
    parser.add_argument("--submit", action="store_true", help="Submit batch")
    parser.add_argument("--poll", action="store_true", help="Check/show batch status")
    parser.add_argument("--download", action="store_true", help="Download results")
    parser.add_argument("--run", action="store_true",
                        help="Full pipeline: submit, wait, download")
    parser.add_argument("--batch-id", default=None,
                        help="Batch ID (for --poll/--download without --submit)")

    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: set OPENAI_API_KEY or pass --api-key", file=sys.stderr)
        sys.exit(1)

    state_path = args.output.replace(".jsonl", "_batch_state.json")
    state = load_state(state_path)

    if not any([args.submit, args.poll, args.download, args.run]):
        print("Specify an action: --submit, --poll, --download, or --run")
        sys.exit(1)

    # --- Submit ---
    if args.submit or args.run:
        batch_input = args.output.replace(".jsonl", "_batch_input.jsonl")
        print("Creating batch input JSONL...")
        make_batch_jsonl(args.dataset, args.model, args.max_completion_tokens,
                         args.temperature, batch_input)

        print("\nUploading to OpenAI...")
        file_id = upload_file(api_key, batch_input)
        save_state(state_path, file_id=file_id)

        print("\nCreating batch...")
        batch_id = create_batch(api_key, file_id)
        save_state(state_path, batch_id=batch_id)
        print(f"\nBatch ID: {batch_id}")
        print(f"State saved to: {state_path}")

    # --- Poll / Wait ---
    state = load_state(state_path)
    batch_id = args.batch_id or state.get("batch_id")
    if not batch_id:
        print("Error: no batch_id available. Run --submit first or pass --batch-id.",
              file=sys.stderr)
        sys.exit(1)

    if args.poll and not args.run:
        batch = check_batch(api_key, batch_id)
        counts = batch.get("request_counts", {})
        print(f"Batch: {batch_id}")
        print(f"  Status: {batch['status']}")
        print(f"  Completed: {counts.get('completed', 0)}/{counts.get('total', 0)}")
        print(f"  Failed: {counts.get('failed', 0)}")
        if batch["status"] not in ("completed", "failed", "expired", "cancelled"):
            print("  (Still in progress)")
            sys.exit(0)

    if args.run:
        batch = wait_for_batch(api_key, batch_id, args.poll_interval)
        if batch["status"] != "completed":
            print(f"\nBatch ended with status: {batch['status']}")
            if batch["status"] == "failed":
                errors = batch.get("errors", {}).get("data", [])
                for err in errors[:5]:
                    print(f"  {err}")
            sys.exit(1)

    # --- Download ---
    if args.download or args.run:
        if args.download and not args.run:
            batch = check_batch(api_key, batch_id)
            if batch["status"] != "completed":
                print(f"Batch not yet completed (status: {batch['status']})")
                sys.exit(1)

        print("\nDownloading results...")
        download_results(api_key, batch, args.output)


if __name__ == "__main__":
    main()
