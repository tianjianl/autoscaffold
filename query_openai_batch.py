#!/usr/bin/env python3
"""
Query GPT-5-nano on HMMT Feb 2026 via OpenAI Batch API.

Two-phase pipeline:
  1. Generate K solutions per problem with type-specific prompts, majority vote
  2. For contested problems: adjudicate by sending all solutions to a judge

Usage:
    python query_openai_batch.py --output predictions.jsonl --model gpt-5-nano --temperature 1 --run
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

NUM_SAMPLES = 3

# --- System prompts (type-specific) ---

SYSTEM_PROMPT = """\
You are an expert competition mathematician solving HMMT February problems.

Solve the following problem step by step.

Important guidelines:
- Think carefully about the problem before diving into computation.
- Double-check your arithmetic and your final answer before writing \\boxed{}.
- Verify your answer against the problem constraints.
- You MUST write your final answer using LaTeX \\boxed{} notation, e.g. \\boxed{42} or \\boxed{\\frac{3}{7}}.
- Simplify fractions. Give exact values (not decimals).
- CRITICAL: Your response MUST end with \\boxed{answer}. No other format is accepted.\
"""

TYPE_SYSTEM_PROMPTS = {
    "Geometry": """\
You are an expert competition mathematician solving an HMMT February geometry problem.

Solve the problem step by step.

Strategy guidance:
- Consider setting up coordinates for computational problems.
- For angle/length problems, consider both synthetic and trigonometric approaches.
- Identify key relationships before computing.
- Watch for degenerate cases and hidden symmetry.
- Double-check your arithmetic and verify your answer satisfies all given constraints.
- You MUST write your final answer using LaTeX \\boxed{} notation. CRITICAL: end with \\boxed{answer}.
- Simplify fractions. Give exact values (not decimals).\
""",
    "Combinatorics": """\
You are an expert competition mathematician solving an HMMT February combinatorics problem.

Solve the problem step by step.

Strategy guidance:
- Before computing, check small cases to build intuition and verify your approach.
- Look for bijections, recursions, generating functions, or inclusion-exclusion.
- Be systematic about counting: clearly define what you're counting, check for overcounting.
- After finding your answer, sanity-check against small cases and order of magnitude.
- Double-check your arithmetic.
- You MUST write your final answer using LaTeX \\boxed{} notation. CRITICAL: end with \\boxed{answer}.
- Simplify fractions. Give exact values (not decimals).\
""",
    "Number Theory": """\
You are an expert competition mathematician solving an HMMT February number theory problem.

Solve the problem step by step.

Strategy guidance:
- Consider working modulo small primes to constrain the answer.
- Use properties of divisibility, GCD/LCM, Euler's totient, and CRT when applicable.
- Check small cases to identify patterns.
- For Diophantine equations, look for bounds and factor-based arguments.
- Verify your answer satisfies all conditions in the problem.
- You MUST write your final answer using LaTeX \\boxed{} notation. CRITICAL: end with \\boxed{answer}.
- Simplify fractions. Give exact values (not decimals).\
""",
    "Algebra": """\
You are an expert competition mathematician solving an HMMT February algebra problem.

Solve the problem step by step.

Strategy guidance:
- Look for symmetry, substitutions, or special structures (AM-GM, Cauchy-Schwarz, etc.).
- Simplify expressions carefully and verify each manipulation.
- Check your answer by substituting back into the original equations.
- Watch for extraneous solutions from squaring or other non-invertible operations.
- Double-check your arithmetic.
- You MUST write your final answer using LaTeX \\boxed{} notation. CRITICAL: end with \\boxed{answer}.
- Simplify fractions. Give exact values (not decimals).\
""",
}


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
    if results:
        return results[-1]
    return None


def make_batch_jsonl(dataset_name, model, max_completion_tokens, temperature,
                     output_path):
    """Create batch input JSONL with NUM_SAMPLES requests per problem.

    Uses problem-type-specific system prompts.
    """
    ds = load_dataset(dataset_name, split="train")
    print(f"Loaded {len(ds)} problems")

    n_requests = 0
    with open(output_path, "w") as f:
        for row in ds:
            idx = row["problem_idx"]
            types = row.get("problem_type", [])
            primary_type = types[0].strip() if types else ""
            system_prompt = TYPE_SYSTEM_PROMPTS.get(primary_type, SYSTEM_PROMPT)

            for k in range(NUM_SAMPLES):
                request = {
                    "custom_id": f"problem-{idx}-sample-{k}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": row["problem"]},
                        ],
                        "max_completion_tokens": max_completion_tokens,
                        "temperature": temperature,
                    },
                }
                f.write(json.dumps(request) + "\n")
                n_requests += 1

    print(f"Wrote {n_requests} requests ({len(ds)} problems x {NUM_SAMPLES} samples)")
    return output_path


def make_adjudicate_batch_jsonl(dataset, all_samples, candidates, model,
                                max_completion_tokens, temperature, output_path):
    """Create adjudication batch for contested problems.

    For problems where K samples produced different answers, send all solutions
    to the model for comparative analysis and error detection.
    """
    ds_by_idx = {row["problem_idx"]: row for row in dataset}
    n_requests = 0
    with open(output_path, "w") as f:
        for idx in sorted(all_samples.keys()):
            idx_candidates = candidates.get(idx, [])
            if len(idx_candidates) <= 1:
                continue  # unanimous or no valid answers — skip

            row = ds_by_idx.get(idx)
            if not row:
                continue

            solutions = all_samples[idx]
            solution_texts = []
            for i, (content, usage, error) in enumerate(solutions):
                if error or not content:
                    continue
                label = chr(65 + i)  # A, B, C, ...
                solution_texts.append(f"--- Solution {label} ---\n{content}")

            if len(solution_texts) < 2:
                continue

            all_solutions_text = "\n\n".join(solution_texts)

            # Summarize candidate answers with vote counts
            vote_lines = []
            for ans, count in idx_candidates:
                vote_lines.append(f"  - \\boxed{{{ans}}} ({count} vote{'s' if count != 1 else ''})")
            vote_summary = "\n".join(vote_lines)

            judge_system = (
                "You are a mathematics competition judge. Your ONLY job is to determine "
                "which of the given solutions is correct. You are NOT solving from scratch.\n\n"
                "Method:\n"
                "1. For each solution, trace the reasoning step by step.\n"
                "2. Find the FIRST error in each incorrect solution.\n"
                "3. The solution with no errors (or the least serious error) has the "
                "correct answer.\n"
                "4. A minority answer can be correct — vote counts do not determine truth.\n"
                "5. You MUST pick one of the proposed answers. Do not introduce a new answer.\n\n"
                "Write your final answer using LaTeX \\boxed{} notation."
            )

            user_content = (
                f"{row['problem']}\n\n"
                f"{len(solution_texts)} mathematicians solved this problem independently. "
                f"Here are their solutions:\n\n{all_solutions_text}\n\n"
                f"The proposed answers and their vote counts:\n{vote_summary}\n\n"
                f"Which answer is correct? Trace each solution's reasoning to find errors. "
                f"Then give the correct answer in \\boxed{{}}."
            )

            request = {
                "custom_id": f"adjudicate-{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": judge_system},
                        {"role": "user", "content": user_content},
                    ],
                    "max_completion_tokens": max_completion_tokens,
                    "temperature": temperature,
                },
            }
            f.write(json.dumps(request) + "\n")
            n_requests += 1

    print(f"Wrote {n_requests} adjudication requests (contested problems only)")
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


def download_raw(api_key, batch, label, output_path):
    """Download raw batch output. Returns raw text or None."""
    output_file_id = batch.get("output_file_id")
    error_file_id = batch.get("error_file_id")

    if error_file_id:
        headers = {"Authorization": f"Bearer {api_key}"}
        resp = requests.get(
            f"{OPENAI_URL}/files/{error_file_id}/content",
            headers=headers, timeout=120,
        )
        if resp.ok:
            error_path = output_path.replace(".jsonl", f"_{label}_errors.jsonl")
            with open(error_path, "w") as f:
                f.write(resp.text)
            print(f"Errors saved to: {error_path}")

    if not output_file_id:
        print("No output file available")
        return None

    headers = {"Authorization": f"Bearer {api_key}"}
    resp = requests.get(
        f"{OPENAI_URL}/files/{output_file_id}/content",
        headers=headers, timeout=300,
    )
    resp.raise_for_status()

    raw_path = output_path.replace(".jsonl", f"_{label}_raw.jsonl")
    with open(raw_path, "w") as f:
        f.write(resp.text)
    print(f"Raw output saved to: {raw_path}")

    return resp.text


def process_samples(raw_text):
    """Parse raw batch output into per-problem samples.

    Returns (samples_dict, token_totals).
    """
    samples = {}
    totals = {"prompt": 0, "completion": 0, "reasoning": 0}

    for line in raw_text.strip().split("\n"):
        if not line.strip():
            continue
        obj = json.loads(line)
        custom_id = obj["custom_id"]
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

        totals["prompt"] += usage.get("prompt_tokens", 0)
        totals["completion"] += usage.get("completion_tokens", 0)
        totals["reasoning"] += (
            (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
        )

        content = ""
        if choices:
            content = choices[0].get("message", {}).get("content") or ""

        samples.setdefault(problem_idx, []).append((content, usage, None))

    return samples, totals


def majority_vote(samples):
    """Majority vote with math_verify semantic grouping.

    Returns:
    - results: {problem_idx: model_answer_content}
    - candidates: {problem_idx: [(boxed_str, count), ...]}
    """
    results = {}
    candidates = {}

    for problem_idx in sorted(samples.keys()):
        problem_samples = samples[problem_idx]
        answer_data = []
        for content, usage, error in problem_samples:
            if error or not content:
                continue
            boxed = extract_boxed(content)
            if boxed is not None:
                rt = (
                    (usage.get("completion_tokens_details") or {})
                    .get("reasoning_tokens", 0)
                )
                answer_data.append((boxed, content, rt))

        if answer_data:
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
            best_boxed, best_content, best_count, _ = groups[0]
            results[problem_idx] = best_content or f"\\boxed{{{best_boxed}}}"
            candidates[problem_idx] = [(g[0], g[2]) for g in groups]
        else:
            model_answer = ""
            for content, usage, error in problem_samples:
                if content:
                    model_answer = content
                    break
            results[problem_idx] = model_answer
            candidates[problem_idx] = []

    return results, candidates


def process_adjudicate_results(raw_text):
    """Parse adjudication batch output.

    Returns (answers, token_totals).
    """
    answers = {}
    totals = {"prompt": 0, "completion": 0, "reasoning": 0}

    for line in raw_text.strip().split("\n"):
        if not line.strip():
            continue
        obj = json.loads(line)
        custom_id = obj["custom_id"]
        problem_idx = int(custom_id.split("-")[1])

        response = obj.get("response", {})
        body = response.get("body", {})
        error = obj.get("error")

        if error:
            continue

        choices = body.get("choices", [])
        usage = body.get("usage", {})

        totals["prompt"] += usage.get("prompt_tokens", 0)
        totals["completion"] += usage.get("completion_tokens", 0)
        totals["reasoning"] += (
            (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
        )

        content = ""
        if choices:
            content = choices[0].get("message", {}).get("content") or ""

        if content:
            answers[problem_idx] = content

    return answers, totals


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
        description="Query GPT-5-nano via OpenAI Batch API (K-sample vote + adjudication)"
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
                        help="Full pipeline: submit, wait, download (with adjudication)")
    parser.add_argument("--batch-id", default=None,
                        help="Batch ID (for --poll/--download without --submit)")
    parser.add_argument("--no-adjudicate", action="store_true",
                        help="Skip adjudication phase (vote only)")

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

    # --- Full two-phase pipeline ---
    if args.run:
        # Phase 1: Generate K samples per problem
        print("=" * 60)
        print("  PHASE 1: Generating solutions (K=%d samples per problem)" % NUM_SAMPLES)
        print("=" * 60)

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

        batch = wait_for_batch(api_key, batch_id, args.poll_interval)
        if batch["status"] != "completed":
            print(f"\nBatch ended with status: {batch['status']}")
            if batch["status"] == "failed":
                errors = batch.get("errors", {}).get("data", [])
                for err in errors[:5]:
                    print(f"  {err}")
            sys.exit(1)

        print("\nDownloading phase 1 results...")
        raw_text = download_raw(api_key, batch, "phase1", args.output)
        if not raw_text:
            sys.exit(1)

        samples, phase1_tokens = process_samples(raw_text)
        vote_results, candidates = majority_vote(samples)

        p1_total = phase1_tokens["prompt"] + phase1_tokens["completion"]
        n_contested = sum(1 for c in candidates.values() if len(c) > 1)
        n_unanimous = sum(1 for c in candidates.values() if len(c) == 1)
        print(f"\nPhase 1 complete: {len(vote_results)} problems, {p1_total:,} tokens")
        print(f"  Unanimous: {n_unanimous}, Contested: {n_contested}")

        final_results = dict(vote_results)
        phase2_tokens = {"prompt": 0, "completion": 0, "reasoning": 0}

        if not args.no_adjudicate and n_contested > 0:
            # Phase 2: Adjudicate contested problems
            print("\n" + "=" * 60)
            print("  PHASE 2: Adjudication (%d contested problems)" % n_contested)
            print("=" * 60)

            ds = load_dataset(args.dataset, split="train")
            adj_input = args.output.replace(".jsonl", "_adjudicate_batch_input.jsonl")
            make_adjudicate_batch_jsonl(ds, samples, candidates, args.model,
                                        args.max_completion_tokens, args.temperature,
                                        adj_input)

            print("\nUploading adjudication batch...")
            file_id2 = upload_file(api_key, adj_input)

            print("\nCreating adjudication batch...")
            batch_id2 = create_batch(api_key, file_id2)
            save_state(state_path, adjudicate_batch_id=batch_id2)

            batch2 = wait_for_batch(api_key, batch_id2, args.poll_interval)

            if batch2["status"] == "completed":
                print("\nDownloading phase 2 results...")
                adj_raw = download_raw(api_key, batch2, "phase2", args.output)
                if adj_raw:
                    adj_answers, phase2_tokens = process_adjudicate_results(adj_raw)
                    n_overridden = 0
                    for idx, answer in adj_answers.items():
                        if answer and extract_boxed(answer):
                            final_results[idx] = answer
                            n_overridden += 1
                    print(f"Adjudication: {len(adj_answers)} answers, "
                          f"{n_overridden} overridden")
            else:
                print(f"\nAdjudication batch failed: {batch2['status']}")
                print("Using phase 1 results only")
        elif args.no_adjudicate:
            print("\nSkipping adjudication (--no-adjudicate)")
        else:
            print("\nNo contested problems — skipping adjudication")

        # Write final predictions
        predictions = []
        for idx in sorted(final_results.keys()):
            predictions.append({
                "problem_idx": idx,
                "model_answer": final_results[idx],
            })
        with open(args.output, "w") as f:
            for p in predictions:
                f.write(json.dumps(p) + "\n")

        p2_total = phase2_tokens["prompt"] + phase2_tokens["completion"]
        grand_total = p1_total + p2_total

        print(f"\n{'=' * 60}")
        print(f"  Results: {len(predictions)} predictions written")
        print(f"  Token usage:")
        print(f"    Phase 1:    {p1_total:>10,}")
        print(f"    Phase 2:    {p2_total:>10,}")
        print(f"    Total:      {grand_total:>10,}")
        print(f"  Output: {args.output}")
        print(f"{'=' * 60}")

        return

    # --- Submit only ---
    if args.submit:
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

    # --- Poll ---
    state = load_state(state_path)
    batch_id = args.batch_id or state.get("batch_id")
    if not batch_id:
        print("Error: no batch_id available. Run --submit first or pass --batch-id.",
              file=sys.stderr)
        sys.exit(1)

    if args.poll:
        batch = check_batch(api_key, batch_id)
        counts = batch.get("request_counts", {})
        print(f"Batch: {batch_id}")
        print(f"  Status: {batch['status']}")
        print(f"  Completed: {counts.get('completed', 0)}/{counts.get('total', 0)}")
        print(f"  Failed: {counts.get('failed', 0)}")


if __name__ == "__main__":
    main()
