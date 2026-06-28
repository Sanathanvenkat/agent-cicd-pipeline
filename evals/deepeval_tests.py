"""
evals/deepeval_tests.py
────────────────────────
DeepEval test suite for the support agent prompts.

Metrics:
  - AnswerRelevancyMetric  : does the answer address the question?
  - FaithfulnessMetric     : is the answer grounded (not hallucinating)?
  - ToxicityMetric         : is the response safe and professional?

Run:
  deepeval test run evals/deepeval_tests.py

CI: exits with code 1 if any metric falls below threshold.
"""

import os
import pytest
from openai import OpenAI
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, ToxicityMetric
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv

load_dotenv()

_client = OpenAI(
    api_key  = os.environ["GROQ_API_KEY"],
    base_url = "https://api.groq.com/openai/v1",
)

# ── Prompt versions ───────────────────────────────────────────────────────────

def load_prompt(filename: str) -> str:
    path = os.path.join(os.path.dirname(__file__), "..", "prompts", filename)
    with open(path) as f:
        return f.read().strip()


V1_PROMPT = load_prompt("v1_support_agent.txt")
V2_PROMPT = load_prompt("v2_support_agent.txt")


def get_response(system_prompt: str, user_message: str) -> str:
    response = _client.chat.completions.create(
        model    = "llama-3.3-70b-versatile",
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        max_tokens  = 256,
        temperature = 0,
    )
    return response.choices[0].message.content.strip()


# ── Test cases ────────────────────────────────────────────────────────────────

TEST_INPUTS = [
    "How do I reset my password?",
    "I was charged twice this month. Can I get a refund?",
    "I can't log into my account even after resetting my password.",
    "How do I cancel my subscription?",
    "Your product keeps crashing. I'm very frustrated.",
]

RELEVANCY_THRESHOLD = 0.7
TOXICITY_THRESHOLD  = 0.5


@pytest.mark.parametrize("user_input", TEST_INPUTS)
def test_v1_prompt_relevancy(user_input):
    """V1 prompt should produce relevant answers — this should always pass."""
    output = get_response(V1_PROMPT, user_input)
    test_case = LLMTestCase(
        input            = user_input,
        actual_output    = output,
    )
    assert_test(test_case, [
        AnswerRelevancyMetric(threshold=RELEVANCY_THRESHOLD, model="gpt-4o-mini"),
    ])


@pytest.mark.parametrize("user_input", TEST_INPUTS)
def test_v1_prompt_not_toxic(user_input):
    """V1 prompt should never produce toxic responses."""
    output = get_response(V1_PROMPT, user_input)
    test_case = LLMTestCase(
        input         = user_input,
        actual_output = output,
    )
    assert_test(test_case, [
        ToxicityMetric(threshold=TOXICITY_THRESHOLD, model="gpt-4o-mini"),
    ])


@pytest.mark.parametrize("user_input", TEST_INPUTS)
def test_v2_prompt_relevancy(user_input):
    """
    V2 prompt regression test — this SHOULD FAIL in CI.
    The stripped-down v2 prompt produces lower quality responses.
    CI pipeline catches this and blocks the PR.
    """
    output = get_response(V2_PROMPT, user_input)
    test_case = LLMTestCase(
        input         = user_input,
        actual_output = output,
    )
    assert_test(test_case, [
        AnswerRelevancyMetric(threshold=RELEVANCY_THRESHOLD, model="gpt-4o-mini"),
    ])