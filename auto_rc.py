#!/usr/bin/env python3
"""
Auto-RC scaffold for GPT-5-nano on HMMT Feb 2026 via OpenAI Batch API.

Like rc_scaffold.py but the model decides when to cache via tool calling:
  - Model receives reasoning prompt + a "cache_reasoning" tool
  - If it returns \\boxed{}: done
  - If it calls cache_reasoning: we summarize, feed summary back, repeat
  - Max T rounds as safety limit

Same prompts as rc_scaffold.py for summarization. The only difference is the
tool definition that lets the model trigger caching autonomously.

Prompts from: context_engineering/rc (Reasoning Cache, 2026)

Usage:
    python auto_rc.py --output predictions_auto_rc.jsonl --run
"""

import argparse
import json
import os
import sys
import time

import requests
from datasets import load_dataset
from math_verify import parse, verify

OPENAI_URL = "https://api.openai.com/v1"

# ---------------------------------------------------------------------------
# Prompts: exact match to context_engineering/rc/inference/prompts/
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
# System prompt: guides tool usage behavior
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert competition mathematician. You have a tool called \
cache_reasoning that lets you save your progress and get a fresh attempt.

When to use the tool:
- Your current approach is stuck or leading to a dead end.
- The problem is complex and you want to try a different strategy.
- You have partial results worth preserving but need to restart reasoning.

When NOT to use the tool:
- You are confident in your solution — just return \\boxed{answer} directly.
- The problem is straightforward and you can solve it in one pass.

If you call cache_reasoning, write your COMPLETE work so far (all reasoning, \
calculations, intermediate results) in the current_work parameter. This will \
be summarized and fed back to you for your next attempt. The more detail you \
provide, the better your next attempt will be."""

# ---------------------------------------------------------------------------
# Tool definition: model calls this to trigger caching
# ---------------------------------------------------------------------------

CACHE_TOOL = {
    "type": "function",
    "function": {
        "name": "cache_reasoning",
        "description": (
            "Save your progress and get a fresh attempt at this problem. "
            "Your work will be summarized and you will try again building "
            "on that summary. Write your COMPLETE reasoning so far in "
            "current_work — all steps, calculations, and partial results."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "current_work": {
                    "type": "string",
                    "description": (
                        "Your COMPLETE reasoning so far: all steps, "
                        "calculations, intermediate results, and partial "
                        "answers. Be thorough — this is what gets summarized "
                        "for your next attempt."
                    ),
                },
            },
            "required": ["current_work"],
        },
    },
}


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


# ---------------------------------------------------------------------------
# Batch API helpers
# ---------------------------------------------------------------------------

def upload_file(api_key, filepath):
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
    file_id = upload_file(api_key, batch_input_path)
    batch_id = create_batch(api_key, file_id)

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


def parse_batch_results_with_tools(raw_text):
    """Parse batch output, detecting tool calls vs direct answers.

    Returns:
        results: {custom_id: {
            "content": str,         # visible content (may be empty)
            "tool_call": str|None,  # tool arg 'current_work' if called
            "usage": dict,
            "error": str|None,
            "used_tool": bool,
        }}
        totals: token counts
    """
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
            results[custom_id] = {
                "content": "", "tool_call": None, "usage": {},
                "error": str(error), "used_tool": False,
            }
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
        tool_call_arg = None
        used_tool = False

        if choices:
            msg = choices[0].get("message", {})
            content = msg.get("content") or ""
            finish_reason = choices[0].get("finish_reason", "")

            tool_calls = msg.get("tool_calls")
            if tool_calls:
                used_tool = True
                for tc in tool_calls:
                    if tc.get("function", {}).get("name") == "cache_reasoning":
                        try:
                            args = json.loads(tc["function"]["arguments"])
                            tool_call_arg = args.get("current_work", "")
                        except (json.JSONDecodeError, KeyError):
                            tool_call_arg = tc["function"].get("arguments", "")

        results[custom_id] = {
            "content": content,
            "tool_call": tool_call_arg,
            "usage": usage,
            "error": None,
            "used_tool": used_tool,
        }

    return results, totals


def parse_batch_results_simple(raw_text):
    """Parse summarization batch output (no tool calls)."""
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
        description="Auto-RC scaffold: model-driven caching via tool calling"
    )
    parser.add_argument("--output", "-o", default="predictions_auto_rc.jsonl",
                        help="Output JSONL path")
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
    parser.add_argument("--max-steps", type=int, default=5,
                        help="Max reasoning rounds (safety limit, default: 5)")
    parser.add_argument("--run", action="store_true",
                        help="Run full auto-RC pipeline")

    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: set OPENAI_API_KEY or pass --api-key", file=sys.stderr)
        sys.exit(1)

    if not args.run:
        print("Specify --run to execute the pipeline")
        sys.exit(1)

    # Load dataset
    ds = load_dataset(args.dataset, split="train")
    problems = {}
    for row in ds:
        problems[row["problem_idx"]] = row["problem"]
    print(f"Loaded {len(problems)} problems from {args.dataset}")

    # Output paths
    output_dir = os.path.dirname(os.path.abspath(args.output)) or "."
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.output))[0]
    state_path = os.path.join(output_dir, f"{base}_batch_state.json")

    T = args.max_steps

    # State per problem
    active = {}  # idx → {problem, curr_summary, reasoning_store, summarization_store}
    for idx, problem in problems.items():
        active[idx] = {
            "problem": problem,
            "curr_summary": "",
            "reasoning_store": [],
            "summarization_store": [],
        }
    completed = {}  # idx → final content

    total_tokens = {"prompt": 0, "completion": 0, "reasoning": 0}

    print(f"\n{'=' * 60}")
    print(f"  Auto-RC: max {T} rounds, model decides caching")
    print(f"  Model: {args.model}, Temp: {args.temperature}")
    print(f"  Reasoning max tokens: {args.max_completion_tokens}")
    print(f"  Summarization max tokens: {args.max_summarization_tokens}")
    print(f"{'=' * 60}")

    for step in range(T):
        if not active:
            print(f"\n  All {len(completed)} problems completed at step {step}")
            break

        print(f"\n{'=' * 60}")
        print(f"  ROUND {step + 1}/{T} ({len(active)} active, "
              f"{len(completed)} completed)")
        print(f"{'=' * 60}")

        # --- Reasoning batch (with tool) ---
        print(f"\n  [Reasoning + Tool] {len(active)} requests")
        reason_input = os.path.join(
            output_dir, f"{base}_round{step + 1}_reasoning_input.jsonl"
        )
        with open(reason_input, "w") as f:
            for idx in sorted(active.keys()):
                st = active[idx]
                filled = REASONING_PROMPT.format(
                    problem=st["problem"],
                    curr_summary=st["curr_summary"],
                )
                request = {
                    "custom_id": f"auto-{idx}-round{step + 1}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": args.model,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": filled},
                        ],
                        "max_completion_tokens": args.max_completion_tokens,
                        "temperature": args.temperature,
                        "tools": [CACHE_TOOL],
                    },
                }
                f.write(json.dumps(request) + "\n")

        label = f"{base}_round{step + 1}_reasoning"
        raw_text = submit_and_wait(
            api_key, reason_input, label, output_dir,
            args.poll_interval, state_path,
        )
        if not raw_text:
            print(f"  FATAL: Reasoning batch failed at round {step + 1}")
            sys.exit(1)

        results, tokens = parse_batch_results_with_tools(raw_text)
        total_tokens["prompt"] += tokens["prompt"]
        total_tokens["completion"] += tokens["completion"]
        total_tokens["reasoning"] += tokens["reasoning"]

        r_total = tokens["prompt"] + tokens["completion"]
        print(f"  Reasoning: {r_total:,} tokens")

        # Classify results: done vs needs-caching
        needs_cache = {}  # idx → reasoning content to summarize
        for idx in list(active.keys()):
            cid = f"auto-{idx}-round{step + 1}"
            res = results.get(cid)
            if not res or res["error"]:
                err = res["error"] if res else "missing"
                print(f"    WARNING: problem {idx}: {err}")
                # Treat error as done with empty content
                completed[idx] = active[idx].get("reasoning_store", [""])[-1]
                del active[idx]
                continue

            content = res["content"]
            active[idx]["reasoning_store"].append(content)

            if res["used_tool"]:
                # Model wants to cache — collect reasoning for summarization
                # Prefer tool_call (complete work), since content is typically
                # null when reasoning models call tools. Combine both if available.
                tool_work = res["tool_call"] or ""
                if content and tool_work:
                    reasoning_text = content + "\n\n" + tool_work
                else:
                    reasoning_text = tool_work or content
                needs_cache[idx] = reasoning_text or ""
                print(f"    Problem {idx}: tool called → will cache")
            else:
                # Model produced a direct answer
                completed[idx] = content
                del active[idx]

        n_done_this_round = len(completed) - sum(
            1 for i in completed if i not in active
        )
        print(f"\n  Round {step + 1}: {len(needs_cache)} cache, "
              f"{len(active) - len(needs_cache)} ???, "
              f"total completed: {len(completed)}")

        if not needs_cache:
            # No problems need caching — all done or errored
            # Remove remaining active that didn't need cache
            for idx in list(active.keys()):
                if idx not in needs_cache:
                    completed[idx] = active[idx]["reasoning_store"][-1] \
                        if active[idx]["reasoning_store"] else ""
                    del active[idx]
            continue

        # --- Summarization batch (for problems that called the tool) ---
        print(f"\n  [Summarization] {len(needs_cache)} requests")
        summ_input = os.path.join(
            output_dir, f"{base}_round{step + 1}_summarization_input.jsonl"
        )
        with open(summ_input, "w") as f:
            for idx in sorted(needs_cache.keys()):
                st = active[idx]
                filled = SUMMARIZATION_PROMPT.format(
                    problem=st["problem"],
                    existing_summary=st["curr_summary"],
                    reasoning=needs_cache[idx],
                )
                request = {
                    "custom_id": f"auto-{idx}-round{step + 1}-summ",
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

        label = f"{base}_round{step + 1}_summarization"
        raw_text = submit_and_wait(
            api_key, summ_input, label, output_dir,
            args.poll_interval, state_path,
        )
        if not raw_text:
            print(f"  WARNING: Summarization failed at round {step + 1}")
            print("  Continuing with current summaries")
            continue

        summ_results, tokens = parse_batch_results_simple(raw_text)
        total_tokens["prompt"] += tokens["prompt"]
        total_tokens["completion"] += tokens["completion"]
        total_tokens["reasoning"] += tokens["reasoning"]

        for idx in needs_cache:
            cid = f"auto-{idx}-round{step + 1}-summ"
            content, usage, error = summ_results.get(cid, ("", {}, "missing"))
            if error:
                print(f"    WARNING: problem {idx} summ: {error}")
            active[idx]["curr_summary"] = content
            active[idx]["summarization_store"].append(content)

        s_total = tokens["prompt"] + tokens["completion"]
        cumulative = total_tokens["prompt"] + total_tokens["completion"]
        print(f"  Summarization: {s_total:,} tokens")
        print(f"  Cumulative: {cumulative:,} tokens")

    # Mark remaining active as completed (hit max_steps)
    for idx in list(active.keys()):
        last = active[idx]["reasoning_store"][-1] \
            if active[idx]["reasoning_store"] else ""
        completed[idx] = last
        print(f"  Problem {idx}: hit max_steps, using last reasoning")

    # Merge active state stores into a combined structure for saving
    all_states = {}
    for idx in problems:
        if idx in active:
            all_states[idx] = active[idx]
        else:
            # Find in completed — reconstruct minimal state
            all_states[idx] = {
                "problem": problems[idx],
                "final_answer": completed.get(idx, ""),
            }

    # --- Write predictions ---
    print(f"\n{'=' * 60}")
    print(f"  WRITING PREDICTIONS")
    print(f"{'=' * 60}")

    predictions = []
    for idx in sorted(completed.keys()):
        predictions.append({
            "problem_idx": idx,
            "model_answer": completed[idx],
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


if __name__ == "__main__":
    main()
