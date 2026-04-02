#!/usr/bin/env python3
"""Quick non-batch test of RC scaffold on a single HMMT problem."""

import json
import os
import sys
import time

import requests
from datasets import load_dataset

OPENAI_URL = "https://api.openai.com/v1/chat/completions"

# Import prompts from rc_scaffold
sys.path.insert(0, os.path.dirname(__file__))
from rc_scaffold import REASONING_PROMPT, SUMMARIZATION_PROMPT, extract_boxed
from auto_rc import CACHE_TOOL, SYSTEM_PROMPT


def call_openai(api_key, messages, model="gpt-5-nano", max_tokens=20000,
                temperature=1, tools=None):
    """Single non-batch API call."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_tokens,
        "temperature": temperature,
    }
    if tools:
        body["tools"] = tools

    resp = requests.post(OPENAI_URL, headers=headers, json=body, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    usage = data.get("usage", {})
    choice = data["choices"][0]
    msg = choice["message"]
    content = msg.get("content") or ""
    tool_calls = msg.get("tool_calls")
    return content, usage, tool_calls


def test_rc_scaffold(api_key, problem, answer, max_steps=2):
    """Test RC scaffold: fixed 2 steps of reasoning + summarization."""
    print("=" * 60)
    print("  TEST: rc_scaffold.py (fixed-step RC)")
    print("=" * 60)

    curr_summary = ""
    total_tokens = 0

    for step in range(max_steps):
        # Reasoning
        filled = REASONING_PROMPT.format(problem=problem, curr_summary=curr_summary)
        print(f"\n--- Step {step + 1}/{max_steps}: Reasoning ---")
        content, usage, _ = call_openai(api_key, [{"role": "user", "content": filled}])
        tokens = usage.get("total_tokens", 0)
        total_tokens += tokens
        boxed = extract_boxed(content)
        print(f"  Tokens: {tokens:,}")
        print(f"  Boxed answer: {boxed}")
        print(f"  Content preview: {content[:200]}...")

        # Summarization
        filled_summ = SUMMARIZATION_PROMPT.format(
            problem=problem, existing_summary=curr_summary, reasoning=content
        )
        print(f"\n--- Step {step + 1}/{max_steps}: Summarization ---")
        summ_content, summ_usage, _ = call_openai(
            api_key, [{"role": "user", "content": filled_summ}], max_tokens=4000
        )
        summ_tokens = summ_usage.get("total_tokens", 0)
        total_tokens += summ_tokens
        curr_summary = summ_content
        print(f"  Tokens: {summ_tokens:,}")
        print(f"  Summary preview: {summ_content[:300]}...")

    final_boxed = extract_boxed(content)
    print(f"\n{'=' * 60}")
    print(f"  RC Result: \\boxed{{{final_boxed}}}")
    print(f"  Gold answer: {answer}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"{'=' * 60}")
    return content


def test_auto_rc(api_key, problem, answer, max_steps=3):
    """Test auto_rc.py: model decides when to cache via tool calling."""
    print("\n" + "=" * 60)
    print("  TEST: auto_rc.py (model-driven caching)")
    print("=" * 60)

    curr_summary = ""
    total_tokens = 0

    for step in range(max_steps):
        filled = REASONING_PROMPT.format(problem=problem, curr_summary=curr_summary)
        print(f"\n--- Round {step + 1}/{max_steps}: Reasoning + Tool ---")
        content, usage, tool_calls = call_openai(
            api_key,
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": filled}],
            tools=[CACHE_TOOL],
        )
        tokens = usage.get("total_tokens", 0)
        total_tokens += tokens
        print(f"  Tokens: {tokens:,}")

        if tool_calls:
            # Model wants to cache
            print(f"  Model called cache_reasoning tool!")
            for tc in tool_calls:
                if tc.get("function", {}).get("name") == "cache_reasoning":
                    try:
                        args = json.loads(tc["function"]["arguments"])
                        tool_arg = args.get("current_work", "")
                    except (json.JSONDecodeError, KeyError):
                        tool_arg = tc["function"].get("arguments", "")
                    print(f"  Tool arg preview: {tool_arg[:200]}...")

            # Use content if available, else tool args
            reasoning_text = content or tool_arg
            print(f"  Content preview: {content[:200] if content else '(empty)'}...")

            # Summarize
            filled_summ = SUMMARIZATION_PROMPT.format(
                problem=problem, existing_summary=curr_summary,
                reasoning=reasoning_text
            )
            print(f"\n--- Round {step + 1}: Summarization ---")
            summ_content, summ_usage, _ = call_openai(
                api_key, [{"role": "user", "content": filled_summ}], max_tokens=4000
            )
            summ_tokens = summ_usage.get("total_tokens", 0)
            total_tokens += summ_tokens
            curr_summary = summ_content
            print(f"  Tokens: {summ_tokens:,}")
            print(f"  Summary preview: {summ_content[:300]}...")
        else:
            # Model answered directly
            boxed = extract_boxed(content)
            print(f"  Model answered directly!")
            print(f"  Boxed answer: {boxed}")
            print(f"  Content preview: {content[:200]}...")
            break

    final_boxed = extract_boxed(content)
    print(f"\n{'=' * 60}")
    print(f"  Auto-RC Result: \\boxed{{{final_boxed}}}")
    print(f"  Gold answer: {answer}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"{'=' * 60}")
    return content


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: set OPENAI_API_KEY", file=sys.stderr)
        sys.exit(1)

    # Load first problem
    ds = load_dataset("MathArena/hmmt_feb_2026", split="train")
    row = ds[0]
    problem = row["problem"]
    answer = row.get("answer", "?")
    idx = row["problem_idx"]

    print(f"Problem {idx}: {problem[:100]}...")
    print(f"Gold answer: {answer}")

    # Test both scaffolds
    test_rc_scaffold(api_key, problem, answer, max_steps=2)
    print("\n\n")
    test_auto_rc(api_key, problem, answer, max_steps=3)


if __name__ == "__main__":
    main()
