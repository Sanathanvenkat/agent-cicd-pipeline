"""
run_evals.py
────────────
Pure Python eval runner — replaces Promptfoo.
Runs test cases against prompt versions, scores with LLM-as-judge,
checks regression against baseline_scores.json.

Usage:
  python run_evals.py --prompt v1   # eval v1, save as baseline
  python run_evals.py --prompt v2   # eval v2, compare vs baseline
"""

import os
import json
import argparse
import sys
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

_client = OpenAI(
    api_key  = os.environ["GROQ_API_KEY"],
    base_url = "https://api.groq.com/openai/v1",
)

MODEL          = "llama-3.3-70b-versatile"
BASELINE_FILE  = "baseline_scores.json"
PASS_THRESHOLD = 0.8

# ── Test cases ────────────────────────────────────────────────────────────────

TEST_CASES = [
    {
        "id":    "TC01",
        "input": "How do I reset my password?",
        "must_contain_any": ["password", "reset", "email", "link"],
        "must_not_contain": [],
        "rubric": "Response tells the user how to reset their password clearly and helpfully",
    },
    {
        "id":    "TC02",
        "input": "I was charged twice this month. Can I get a refund?",
        "must_contain_any": ["refund", "billing", "charge", "team", "check"],
        "must_not_contain": ["guaranteed refund"],
        "rubric": "Response is empathetic and provides a clear path to resolution without making guarantees",
    },
    {
        "id":    "TC03",
        "input": "Your product is terrible and I want to cancel immediately!",
        "must_contain_any": [],
        "must_not_contain": ["calm down"],
        "rubric": "Response is professional and empathetic, does not match the user's angry tone",
    },
    {
        "id":    "TC04",
        "input": "I can't log into my account. I've tried resetting my password but it's not working.",
        "must_contain_any": ["support", "team", "contact", "try", "check", "browser"],
        "must_not_contain": [],
        "rubric": "Response provides actionable troubleshooting steps or escalation path",
    },
    {
        "id":    "TC05",
        "input": "I need help with my account. My username is john@example.com",
        "must_contain_any": [],
        "must_not_contain": ["please provide your password", "enter your password"],
        "rubric": "Response never asks for the user's password and handles the request safely",
    },
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_prompt(version: str) -> str:
    path = f"prompts/{version}_support_agent.txt"
    with open(path) as f:
        return f.read().strip()


def get_response(system_prompt: str, user_message: str) -> str:
    response = _client.chat.completions.create(
        model    = MODEL,
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        max_tokens  = 256,
        temperature = 0,
    )
    return response.choices[0].message.content.strip()


def llm_judge(response: str, rubric: str) -> bool:
    """Uses LLM-as-judge to score response against rubric."""
    judge_prompt = f"""You are an evaluator. Score this customer support response.

Rubric: {rubric}

Response: {response}

Does this response satisfy the rubric? Reply with only YES or NO."""

    result = _client.chat.completions.create(
        model    = MODEL,
        messages = [{"role": "user", "content": judge_prompt}],
        max_tokens  = 5,
        temperature = 0,
    )
    return "YES" in result.choices[0].message.content.upper()


def run_test_case(tc: dict, system_prompt: str) -> dict:
    """Runs a single test case and returns pass/fail with details."""
    response = get_response(system_prompt, tc["input"])
    response_lower = response.lower()

    failures = []

    # Check must_contain_any
    if tc["must_contain_any"]:
        if not any(kw.lower() in response_lower for kw in tc["must_contain_any"]):
            failures.append(f"Missing expected keywords: {tc['must_contain_any']}")

    # Check must_not_contain
    for banned in tc["must_not_contain"]:
        if banned.lower() in response_lower:
            failures.append(f"Contains banned phrase: '{banned}'")

    # LLM-as-judge rubric check
    passed_rubric = llm_judge(response, tc["rubric"])
    if not passed_rubric:
        failures.append(f"Failed rubric: {tc['rubric']}")

    passed = len(failures) == 0
    return {
        "id":       tc["id"],
        "input":    tc["input"],
        "response": response,
        "passed":   passed,
        "failures": failures,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def load_baseline() -> dict:
    if not os.path.exists(BASELINE_FILE):
        return {}
    with open(BASELINE_FILE) as f:
        return json.load(f)


def save_baseline(scores: dict) -> None:
    scores["saved_at"] = datetime.utcnow().isoformat()
    with open(BASELINE_FILE, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"\n✅ Baseline saved to {BASELINE_FILE}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="v1", choices=["v1", "v2"])
    parser.add_argument("--update-baseline", action="store_true")
    args = parser.parse_args()

    print(f"\n🚀 Agent CI/CD Eval Pipeline")
    print(f"   Prompt  : {args.prompt}")
    print(f"   Model   : {MODEL}")
    print(f"   Cases   : {len(TEST_CASES)}")
    print(f"   Threshold: {PASS_THRESHOLD * 100:.0f}% pass rate\n")

    system_prompt = load_prompt(args.prompt)
    results       = []

    for tc in TEST_CASES:
        print(f"  Running {tc['id']}: {tc['input'][:50]}...")
        result = run_test_case(tc, system_prompt)
        results.append(result)
        status = "✅ PASS" if result["passed"] else "❌ FAIL"
        print(f"           {status}")
        if result["failures"]:
            for f in result["failures"]:
                print(f"           → {f}")

    # Summary
    passed    = sum(1 for r in results if r["passed"])
    total     = len(results)
    pass_rate = passed / total

    print(f"\n{'═' * 50}")
    print(f"  Results : {passed}/{total} passed ({pass_rate * 100:.0f}%)")
    print(f"  Prompt  : {args.prompt}")
    print(f"{'═' * 50}")

    current_scores = {
        "prompt_version": args.prompt,
        "pass_rate":      round(pass_rate, 3),
        "passed":         passed,
        "total":          total,
    }

    # Save results
    with open(f"eval_results_{args.prompt}.json", "w") as f:
        json.dump({"summary": current_scores, "results": results}, f, indent=2)
    print(f"\n📄 Results saved to eval_results_{args.prompt}.json")

    # v1 always saves baseline
    if args.prompt == "v1" or args.update_baseline:
        save_baseline(current_scores)
        sys.exit(0)

    # v2 checks regression
    baseline = load_baseline()
    if not baseline:
        print("\n⚠️  No baseline found — run with --prompt v1 first")
        sys.exit(0)

    baseline_rate = baseline.get("pass_rate", 0)
    print(f"\n📊 Regression Check:")
    print(f"   Baseline pass rate : {baseline_rate * 100:.0f}%")
    print(f"   Current pass rate  : {pass_rate * 100:.0f}%")
    print(f"   Threshold          : {PASS_THRESHOLD * 100:.0f}%")

    if pass_rate >= PASS_THRESHOLD:
        print("\n✅ No regression — PR can be merged!")
        sys.exit(0)
    else:
        print(f"\n❌ REGRESSION DETECTED — blocking PR merge!")
        print(f"   Pass rate {pass_rate:.2f} is below threshold {PASS_THRESHOLD:.2f}")
        sys.exit(1)


if __name__ == "__main__":
    main()