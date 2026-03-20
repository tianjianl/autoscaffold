#!/usr/bin/env python3
"""
Grade math competition answers using math_verify with parallel verification.

Usage:
    python grade.py --predictions predictions.jsonl [--workers 8] [--verbose]

Predictions file (JSONL), one JSON object per line:
    {"problem_idx": 1, "model_answer": "The answer is $\\boxed{\\frac{7}{2}}$"}
    {"problem_idx": 2, "model_answer": "48"}

Also supports JSON array format:
    [{"problem_idx": 1, "model_answer": "..."}, ...]

Flexible field names: problem_idx/idx/id for index, model_answer/prediction/answer/response for answer.
"""

import argparse
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError

from datasets import load_dataset
from math_verify import parse, verify


def ensure_latex_wrapped(text: str) -> str:
    """Wrap text in $...$ if it contains LaTeX commands but no delimiters.

    Gold answers from datasets often contain raw LaTeX like '\\frac{1}{2}'
    without delimiters. math_verify's LatexExtractionConfig requires $...$
    or \\[...\\] delimiters to parse LaTeX correctly. Without wrapping,
    expressions like '5\\sqrt{5}' incorrectly parse as just [5].
    """
    if not text:
        return text
    latex_commands = [
        "\\frac", "\\sqrt", "\\pi", "\\cdot", "\\times",
        "\\left", "\\right", "\\infty", "\\log", "\\ln",
        "\\sin", "\\cos", "\\tan", "\\circ", "^{", "_{",
    ]
    has_latex = any(cmd in text for cmd in latex_commands)
    has_delimiters = "$" in text or "\\[" in text or "\\(" in text
    if has_latex and not has_delimiters:
        return f"${text}$"
    return text


def grade_single(problem_idx: int, gold_answer: str, model_answer: str) -> tuple:
    """Grade a single problem. Runs in a worker process.

    Returns (problem_idx, correct: bool, detail: str).
    """
    try:
        # Parse gold answer (wrap in $...$ if raw LaTeX)
        gold_text = ensure_latex_wrapped(gold_answer)
        gold_parsed = parse(gold_text)
        if not gold_parsed:
            # Fallback: force-wrap in $...$
            gold_parsed = parse(f"${gold_answer}$")
        if not gold_parsed:
            return (problem_idx, False, f"GOLD_PARSE_FAIL: {gold_answer!r}")

        # Handle empty/None model answers
        if not model_answer or not str(model_answer).strip():
            return (problem_idx, False, "EMPTY_PREDICTION")

        model_text = str(model_answer).strip()

        # Parse model answer (try as-is first, then wrapped)
        pred_parsed = parse(model_text)
        if not pred_parsed:
            pred_parsed = parse(f"${model_text}$")
        if not pred_parsed:
            return (problem_idx, False, f"PRED_PARSE_FAIL: {model_text[:200]!r}")

        # verify() is asymmetric: gold must be first argument
        result = verify(gold_parsed, pred_parsed)
        return (problem_idx, bool(result), "correct" if result else "incorrect")

    except Exception as e:
        return (problem_idx, False, f"ERROR: {type(e).__name__}: {e}")


def load_predictions(path: str) -> dict:
    """Load predictions from JSONL or JSON array file.

    Supports flexible field names for both index and answer fields.
    """
    predictions = {}
    idx_keys = ["problem_idx", "idx", "id", "problem_id", "index"]
    ans_keys = ["model_answer", "prediction", "answer", "response", "output", "solution"]

    def extract(obj):
        idx = None
        for k in idx_keys:
            if k in obj:
                idx = obj[k]
                break
        ans = ""
        for k in ans_keys:
            if k in obj:
                ans = obj[k]
                break
        return idx, ans

    with open(path) as f:
        content = f.read().strip()

    if not content:
        print("Error: predictions file is empty", file=sys.stderr)
        sys.exit(1)

    # Try JSON array first
    try:
        data = json.loads(content)
        if isinstance(data, list):
            for obj in data:
                idx, ans = extract(obj)
                if idx is not None:
                    predictions[int(idx)] = str(ans) if ans is not None else ""
            return predictions
        elif isinstance(data, dict):
            # Single object — treat as one prediction
            idx, ans = extract(data)
            if idx is not None:
                predictions[int(idx)] = str(ans) if ans is not None else ""
            return predictions
    except json.JSONDecodeError:
        pass

    # JSONL format
    for line_num, line in enumerate(content.split("\n"), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"Warning: skipping malformed JSON on line {line_num}: {e}",
                  file=sys.stderr)
            continue
        idx, ans = extract(obj)
        if idx is not None:
            predictions[int(idx)] = str(ans) if ans is not None else ""
        else:
            print(f"Warning: no problem index found on line {line_num}, skipping",
                  file=sys.stderr)

    return predictions


def main():
    parser = argparse.ArgumentParser(
        description="Grade math answers using math_verify with parallel verification"
    )
    parser.add_argument(
        "--predictions", required=True,
        help="Path to predictions file (JSONL or JSON)"
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Number of parallel workers (default: 8)"
    )
    parser.add_argument(
        "--dataset", default="MathArena/hmmt_feb_2026",
        help="HuggingFace dataset name (default: MathArena/hmmt_feb_2026)"
    )
    parser.add_argument(
        "--timeout", type=int, default=60,
        help="Per-problem timeout in seconds (default: 60)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print detailed per-problem results"
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output JSON path for results (default: <predictions>_results.json)"
    )
    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    ds = load_dataset(args.dataset, split="train")
    gold = {row["problem_idx"]: row for row in ds}
    print(f"  {len(gold)} problems loaded")

    # Load predictions
    print(f"Loading predictions: {args.predictions}")
    predictions = load_predictions(args.predictions)
    print(f"  {len(predictions)} predictions loaded")

    if not predictions:
        print("Error: no predictions to grade", file=sys.stderr)
        sys.exit(1)

    # Check coverage
    missing = sorted(set(gold.keys()) - set(predictions.keys()))
    extra = sorted(set(predictions.keys()) - set(gold.keys()))
    if missing:
        print(f"  Warning: {len(missing)} problems with no prediction: {missing}")
    if extra:
        print(f"  Warning: {len(extra)} predictions with no matching problem: {extra}")

    # Grade with ProcessPoolExecutor
    # (math_verify uses SIGALRM for timeouts, which requires the main thread;
    #  ProcessPoolExecutor gives each worker its own main thread)
    print(f"\nGrading with {args.workers} workers...")
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for idx, model_answer in predictions.items():
            if idx not in gold:
                continue
            gold_answer = gold[idx]["answer"]
            fut = executor.submit(grade_single, idx, gold_answer, model_answer)
            futures[fut] = idx

        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                prob_idx, correct, detail = fut.result(timeout=args.timeout)
                results[prob_idx] = (correct, detail)
            except TimeoutError:
                results[idx] = (False, "TIMEOUT")
            except Exception as e:
                results[idx] = (False, f"WORKER_ERROR: {e}")

    # Mark missing predictions
    for idx in missing:
        results[idx] = (False, "NO_PREDICTION")

    # Compute overall accuracy
    total = len(gold)
    correct_count = sum(1 for idx in gold if idx in results and results[idx][0])

    print(f"\n{'=' * 60}")
    print(f"  Accuracy: {correct_count}/{total} ({100 * correct_count / total:.1f}%)")
    print(f"{'=' * 60}")

    # Per-problem breakdown
    print("\nPer-problem results:")
    for idx in sorted(gold.keys()):
        row = gold[idx]
        c, detail = results.get(idx, (False, "MISSING"))
        status = "PASS" if c else "FAIL"
        types = row.get("problem_type", [])
        type_str = ", ".join(t.strip() for t in types) if types else "Unknown"

        if args.verbose or not c:
            print(f"  [{status}] Problem {idx:2d} ({type_str}): {detail}")
            if args.verbose and not c:
                print(f"           Gold: {row['answer']!r}")
                if idx in predictions:
                    print(f"           Pred: {predictions[idx][:200]!r}")

    # Per-type accuracy
    by_type = {}
    for idx in gold:
        row = gold[idx]
        c = results.get(idx, (False,))[0]
        types = row.get("problem_type", [])
        for t in (types or ["Unknown"]):
            t = t.strip()
            by_type.setdefault(t, {"correct": 0, "total": 0})
            by_type[t]["total"] += 1
            if c:
                by_type[t]["correct"] += 1

    print(f"\nBy problem type:")
    for t in sorted(by_type):
        s = by_type[t]
        pct = 100 * s["correct"] / s["total"] if s["total"] > 0 else 0
        print(f"  {t:20s}: {s['correct']:2d}/{s['total']:2d} ({pct:.0f}%)")

    # Detailed results list for JSON output
    detail_list = []
    for idx in sorted(gold.keys()):
        row = gold[idx]
        c, detail = results.get(idx, (False, "MISSING"))
        entry = {
            "problem_idx": idx,
            "correct": c,
            "detail": detail,
            "gold_answer": row["answer"],
            "problem_type": [t.strip() for t in row.get("problem_type", [])],
        }
        if idx in predictions:
            entry["model_answer"] = predictions[idx]
        detail_list.append(entry)

    # Save summary JSON
    output_path = args.output or (
        os.path.splitext(args.predictions)[0] + "_results.json"
    )
    summary = {
        "accuracy": correct_count / total if total > 0 else 0,
        "correct": correct_count,
        "total": total,
        "by_type": by_type,
        "results": detail_list,
    }
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
