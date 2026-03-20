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
import sys
import time

import requests
from datasets import load_dataset

OPENAI_URL = "https://api.openai.com/v1"

SYSTEM_PROMPT = """\
You are an expert mathematician solving competition-level math problems.
Think step by step, showing your full reasoning.
Present your final answer inside \\boxed{} at the end of your solution.\
"""


def make_batch_jsonl(dataset_name, model, max_completion_tokens, temperature,
                     output_path):
    """Create the batch input JSONL file."""
    ds = load_dataset(dataset_name, split="train")
    print(f"Loaded {len(ds)} problems")

    with open(output_path, "w") as f:
        for row in ds:
            idx = row["problem_idx"]
            request = {
                "custom_id": f"problem-{idx}",
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

    print(f"Wrote {len(ds)} requests to {output_path}")
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
        batch = check_batch(api_key, batch_id)
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
    """Download batch results and convert to grade.py format."""
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

    # Convert to grade.py format
    results = []
    total_prompt = 0
    total_completion = 0
    total_reasoning = 0

    for line in resp.text.strip().split("\n"):
        if not line.strip():
            continue
        obj = json.loads(line)
        custom_id = obj["custom_id"]  # "problem-N"
        problem_idx = int(custom_id.split("-")[1])

        response = obj.get("response", {})
        body = response.get("body", {})
        error = obj.get("error")

        if error:
            results.append({
                "problem_idx": problem_idx,
                "model_answer": "",
                "error": str(error),
                "usage": {},
            })
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
            finish_reason = choices[0].get("finish_reason", "unknown")
        else:
            content = ""
            finish_reason = "no_choices"

        results.append({
            "problem_idx": problem_idx,
            "model_answer": content,
            "finish_reason": finish_reason,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "reasoning_tokens": reasoning_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
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
