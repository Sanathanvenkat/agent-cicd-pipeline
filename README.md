# Agent CI/CD Pipeline

> I built the CI/CD pipeline that AI teams wish they had.

Every PR that changes a prompt automatically runs evals, scores against a baseline, and **blocks merge on regression**. Prompts are versioned in git like code.

## How it works

```
Developer changes prompts/v2_support_agent.txt
              │
              ▼
      Opens Pull Request
              │
              ▼
┌─────────────────────────────────────┐
│  GitHub Actions triggers            │
│  (on: pull_request, paths: prompts/)│
└─────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Promptfoo runs 5 test cases        │
│  against both prompt versions       │
│  Threshold: 80% pass rate           │
└─────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Regression check vs baseline       │
│  v1 baseline: 90% pass rate         │
│  v2 current:  ~40% pass rate        │
└─────────────────────────────────────┘
              │
         ❌ BLOCKED
              │
    PR comment posted with scores
```

## Prompt versions

| Version | Description | Pass Rate |
|---|---|---|
| `v1_support_agent.txt` | Full prompt with persona, rules, constraints | ~90% |
| `v2_support_agent.txt` | Stripped-down prompt (intentional regression) | ~40% |

## Eval suite

**Promptfoo** (`evals/promptfoo.yaml`):
- 5 test cases covering password reset, refund, angry customer, technical issues, security
- Assertions: `contains-any`, `llm-rubric`, `not-contains`, `javascript` length check
- Threshold: 80% pass rate

**DeepEval** (`evals/deepeval_tests.py`):
- `AnswerRelevancyMetric` — does the answer address the question?
- `ToxicityMetric` — is the response professional and safe?

## Setup

```bash
# Python deps
pip install -r requirements.txt

# Promptfoo (requires Node.js)
npm install -g promptfoo

cp .env.example .env   # add GROQ_API_KEY
```

## Run locally

```bash
# Eval v1 (baseline) — should pass
python run_evals.py --prompt v1

# Eval v2 (regressed) — should fail and show regression
python run_evals.py --prompt v2

# Run Promptfoo directly
promptfoo eval --config evals/promptfoo.yaml
```

## GitHub Actions setup

Add `GROQ_API_KEY` to your repo secrets:
Settings → Secrets and variables → Actions → New repository secret

The workflow triggers automatically on any PR touching `prompts/` or `evals/`.

## Key concepts demonstrated

- **Prompt versioning** — prompts treated as code, tracked in git
- **Automated regression testing** — CI blocks bad prompts before merge
- **Promptfoo** — declarative eval config with multiple assertion types
- **DeepEval** — pytest-style LLM test cases with metric thresholds
- **GitHub Actions** — PR comment with eval results table
- **Baseline comparison** — current scores vs stored baseline, not just absolute threshold