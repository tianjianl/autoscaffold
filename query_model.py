#!/usr/bin/env python3
"""
Query an API model on HMMT Feb 2026 problems and save predictions for grading.

Usage:
    python query_model.py --output predictions.jsonl [--workers 4] [--max-total-tokens 2500000]

Outputs JSONL compatible with grade.py.
"""

import argparse
import json
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from datasets import load_dataset

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM_PROMPT = """\
You are an expert mathematician solving competition-level math problems.
Think step by step, showing your full reasoning.
Present your final answer inside \\boxed{} at the end of your solution.\
"""

# Lock for thread-safe token accounting
token_lock = threading.Lock()
total_tokens_used = 0


def query_single(
    problem_idx: int,
    problem_text: str,
    api_key: str,
    model: str,
    max_completion_tokens: int,
    temperature: float,
    max_retries: int = 5,
) -> dict:
    """Send a single problem to the API. Returns result dict."""
    global total_tokens_used

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem_text},
        ],
        "max_completion_tokens": max_completion_tokens,
        "temperature": temperature,
    }

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(
                OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=600,  # 10 min timeout for long generations
            )

            if resp.status_code == 429:
                # Rate limited — back off
                retry_after = float(resp.headers.get("Retry-After", 2 ** attempt))
                print(f"  [Problem {problem_idx}] Rate limited, waiting {retry_after:.0f}s "
                      f"(attempt {attempt}/{max_retries})")
                time.sleep(retry_after)
                continue

            if resp.status_code in (500, 502, 503):
                wait = 2 ** attempt
                print(f"  [Problem {problem_idx}] Server error {resp.status_code}, "
                      f"waiting {wait}s (attempt {attempt}/{max_retries})")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            data = resp.json()

            # Check for API-level error
            if "error" in data:
                err = data["error"]
                msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
                print(f"  [Problem {problem_idx}] API error: {msg} "
                      f"(attempt {attempt}/{max_retries})")
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                    continue
                return {
                    "problem_idx": problem_idx,
                    "model_answer": "",
                    "raw_response": "",
                    "error": msg,
                    "usage": {},
                }

            # Extract response
            choice = data["choices"][0]
            message = choice["message"]
            content = message.get("content") or ""
            reasoning = message.get("reasoning") or ""
            finish_reason = choice.get("finish_reason", "unknown")
            usage = data.get("usage", {})

            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            reasoning_tokens = (usage.get("completion_tokens_details") or {}
                                ).get("reasoning_tokens", 0)
            this_total = prompt_tokens + completion_tokens

            with token_lock:
                total_tokens_used += this_total

            # If content is empty but reasoning exists, the model ran out of
            # tokens during thinking. Try to extract an answer from reasoning.
            model_answer = content
            if not model_answer and reasoning:
                model_answer = reasoning

            return {
                "problem_idx": problem_idx,
                "model_answer": model_answer,
                "content": content,
                "reasoning": reasoning,
                "finish_reason": finish_reason,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "reasoning_tokens": reasoning_tokens,
                    "total_tokens": this_total,
                },
                "error": None,
            }

        except requests.exceptions.Timeout:
            print(f"  [Problem {problem_idx}] Request timeout "
                  f"(attempt {attempt}/{max_retries})")
            if attempt < max_retries:
                time.sleep(2 ** attempt)
                continue
            return {
                "problem_idx": problem_idx,
                "model_answer": "",
                "raw_response": "",
                "error": "timeout",
                "usage": {},
            }
        except requests.exceptions.RequestException as e:
            print(f"  [Problem {problem_idx}] Request error: {e} "
                  f"(attempt {attempt}/{max_retries})")
            if attempt < max_retries:
                time.sleep(2 ** attempt)
                continue
            return {
                "problem_idx": problem_idx,
                "model_answer": "",
                "raw_response": "",
                "error": str(e),
                "usage": {},
            }

    # Exhausted retries
    return {
        "problem_idx": problem_idx,
        "model_answer": "",
        "raw_response": "",
        "error": "max_retries_exceeded",
        "usage": {},
    }


def main():
    parser = argparse.ArgumentParser(
        description="Query API model on HMMT Feb 2026 problems"
    )
    parser.add_argument(
        "--output", "-o", default="predictions.jsonl",
        help="Output JSONL path (default: predictions.jsonl)"
    )
    parser.add_argument(
        "--model", default="qwen/qwen3.5-9b",
        help="OpenRouter model ID (default: qwen/qwen3.5-9b)"
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of concurrent API requests (default: 4)"
    )
    parser.add_argument(
        "--max-total-tokens", type=int, default=2_500_000,
        help="Maximum total tokens budget (default: 2500000)"
    )
    parser.add_argument(
        "--max-completion-tokens", type=int, default=None,
        help="Max completion tokens per question (default: auto from budget)"
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
        help="OpenRouter API key (default: $OPENROUTER_API_KEY)"
    )
    parser.add_argument(
        "--problems", type=str, default=None,
        help="Comma-separated problem indices to query (default: all)"
    )
    args = parser.parse_args()

    # API key
    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: set OPENROUTER_API_KEY or pass --api-key", file=sys.stderr)
        sys.exit(1)

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    ds = load_dataset(args.dataset, split="train")
    problems = {row["problem_idx"]: row for row in ds}
    print(f"  {len(problems)} problems loaded")

    # Filter problems if requested
    if args.problems:
        selected = [int(x.strip()) for x in args.problems.split(",")]
        problems = {k: v for k, v in problems.items() if k in selected}
        print(f"  Selected {len(problems)} problems: {sorted(problems.keys())}")

    n = len(problems)

    # Compute per-question token budget
    # Reserve ~1500 tokens input per question
    input_reserve_per_q = 1500
    total_input_reserve = input_reserve_per_q * n
    output_budget = args.max_total_tokens - total_input_reserve

    if args.max_completion_tokens:
        max_comp = args.max_completion_tokens
    else:
        max_comp = output_budget // n
        # Cap at 200K (well within 256K context window)
        max_comp = min(max_comp, 200_000)

    print(f"\nToken budget:")
    print(f"  Total budget:         {args.max_total_tokens:>12,}")
    print(f"  Input reserve:        {total_input_reserve:>12,} ({input_reserve_per_q}/question)")
    print(f"  Output budget:        {output_budget:>12,}")
    print(f"  Max tokens/question:  {max_comp:>12,}")
    print(f"  Workers:              {args.workers:>12}")
    print(f"  Model:                {args.model}")
    print(f"  Temperature:          {args.temperature}")

    # Load existing results to support resuming
    existing = {}
    if os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    if obj.get("model_answer") and not obj.get("error"):
                        existing[obj["problem_idx"]] = obj
        if existing:
            print(f"\n  Found {len(existing)} existing results in {args.output}, "
                  f"resuming remaining")

    # Determine which problems still need querying
    to_query = {k: v for k, v in problems.items() if k not in existing}
    if not to_query:
        print("\nAll problems already have results. Nothing to do.")
        return

    print(f"\nQuerying {len(to_query)} problems...")
    start_time = time.time()

    results = dict(existing)  # Start with existing results
    completed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for idx in sorted(to_query.keys()):
            row = to_query[idx]
            fut = executor.submit(
                query_single,
                idx,
                row["problem"],
                api_key,
                args.model,
                max_comp,
                args.temperature,
            )
            futures[fut] = idx

        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                result = fut.result()
            except Exception as e:
                result = {
                    "problem_idx": idx,
                    "model_answer": "",
                    "raw_response": "",
                    "error": str(e),
                    "usage": {},
                }

            results[idx] = result
            completed += 1

            # Progress
            usage = result.get("usage", {})
            comp_tok = usage.get("completion_tokens", 0)
            reason_tok = usage.get("reasoning_tokens", 0)
            err = result.get("error")
            fr = result.get("finish_reason", "")
            has_content = bool(result.get("content"))
            if err:
                status = f"error={err}"
            else:
                parts = [f"{comp_tok:,} tokens"]
                if reason_tok:
                    parts.append(f"reasoning={reason_tok:,}")
                if fr == "length":
                    parts.append("TRUNCATED")
                if not has_content:
                    parts.append("no-content")
                status = ", ".join(parts)

            with token_lock:
                current_total = total_tokens_used

            elapsed = time.time() - start_time
            print(f"  [{completed}/{len(to_query)}] Problem {idx:2d}: {status} "
                  f"| cumulative: {current_total:,}/{args.max_total_tokens:,} tokens "
                  f"| {elapsed:.0f}s")

            # Check token budget
            if current_total >= args.max_total_tokens:
                print(f"\n  TOKEN BUDGET EXHAUSTED ({current_total:,} >= "
                      f"{args.max_total_tokens:,}). Stopping.")
                # Cancel remaining futures
                for f in futures:
                    f.cancel()
                break

            # Write incrementally (append mode for resume support)
            with open(args.output, "a") as f:
                f.write(json.dumps(result) + "\n")

    # Write final consolidated output (overwrite with all results sorted)
    with open(args.output, "w") as f:
        for idx in sorted(results.keys()):
            f.write(json.dumps(results[idx]) + "\n")

    elapsed = time.time() - start_time

    # Summary
    with token_lock:
        final_tokens = total_tokens_used

    n_success = sum(1 for r in results.values()
                    if r.get("model_answer") and not r.get("error"))
    n_error = sum(1 for r in results.values() if r.get("error"))
    n_truncated = sum(1 for r in results.values()
                      if r.get("finish_reason") == "length" and not r.get("error"))
    n_reasoning_only = sum(1 for r in results.values()
                          if not r.get("content") and r.get("reasoning")
                          and not r.get("error"))
    total_prompt = sum(r.get("usage", {}).get("prompt_tokens", 0)
                       for r in results.values())
    total_completion = sum(r.get("usage", {}).get("completion_tokens", 0)
                          for r in results.values())
    total_reasoning = sum(r.get("usage", {}).get("reasoning_tokens", 0)
                         for r in results.values())

    print(f"\n{'=' * 60}")
    print(f"  Completed in {elapsed:.1f}s")
    print(f"  Success: {n_success}/{n}")
    print(f"  Errors:  {n_error}/{n}")
    if n_truncated:
        print(f"  Truncated (hit token limit): {n_truncated}/{n}")
    if n_reasoning_only:
        print(f"  Reasoning-only (no content): {n_reasoning_only}/{n}")
    print(f"  Token usage:")
    print(f"    Prompt:     {total_prompt:>10,}")
    print(f"    Completion: {total_completion:>10,}")
    if total_reasoning:
        print(f"    (Reasoning: {total_reasoning:>9,})")
    print(f"    Total:      {total_prompt + total_completion:>10,} / "
          f"{args.max_total_tokens:,}")
    print(f"  Output: {args.output}")
    print(f"{'=' * 60}")

    if n_error > 0:
        print(f"\nProblems with errors:")
        for idx in sorted(results.keys()):
            r = results[idx]
            if r.get("error"):
                print(f"  Problem {idx}: {r['error']}")


if __name__ == "__main__":
    main()
