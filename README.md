# SupportTicketEnv

> A production-ready OpenEnv environment for training and evaluating AI agents on real-world customer support ticket triage.

---

## Why This Environment Matters

Customer support is one of the highest-ROI automation targets in enterprise AI. Every company with a customer base processes thousands of support tickets per day — billing disputes, login failures, technical crashes — and the quality of triage (which team handles it, how fast, with what message) directly affects customer retention and operational cost.

Training an AI agent to triage, classify, and respond to support tickets is a hard, multi-dimensional problem:
- **Classification** — tickets are ambiguous and multi-topic
- **Priority** — wrong priority burns engineering time or loses customers
- **Communication** — tone and completeness matter as much as correctness
- **Multi-step reasoning** — the best agents ask clarifying questions before committing

SupportTicketEnv gives researchers and practitioners a rigorous, deterministic benchmark environment for this problem, with a rich reward signal and OpenEnv compliance.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                      inference.py                        │
│  (LLM agent loop — OpenAI client, structured output)     │
└────────────────────────┬─────────────────────────────────┘
                         │ Action (JSON)
                         ▼
┌──────────────────────────────────────────────────────────┐
│                    SupportTicketEnv (env.py)              │
│                                                          │
│  reset(task_name) ──► Observation                        │
│  step(action)     ──► Observation, Reward, done, info    │
│  state()          ──► dict (full internal state)         │
└────────────┬──────────────────────┬──────────────────────┘
             │                      │
             ▼                      ▼
     ┌───────────────┐    ┌─────────────────────┐
     │  tasks.py     │    │   grader.py          │
     │  (3 tasks,    │    │   (deterministic     │
     │  deterministic│    │   reward breakdown)  │
     │  no randomness│    │                      │
     └───────────────┘    └─────────────────────┘
             │
             ▼
     ┌───────────────┐
     │  models.py    │
     │  Observation  │
     │  Action       │
     │  Reward       │
     └───────────────┘
```

---

## Action & Observation Space

### Observation

| Field | Type | Description |
|---|---|---|
| `ticket_id` | str | Unique ticket identifier |
| `ticket_text` | str | Customer complaint text |
| `step` | int | Current step (0-indexed) |
| `max_steps` | int | Maximum steps for this task |
| `previous_actions` | list[dict] | History of actions taken |
| `clarification_answer` | str \| null | Answer to last follow-up question |
| `done` | bool | Whether episode has ended |
| `info` | dict | Metadata and error messages |

### Action

| Field | Type | Description |
|---|---|---|
| `predicted_category` | `"billing" \| "technical" \| "account"` | Ticket category |
| `predicted_priority` | `"low" \| "medium" \| "high"` | Ticket priority |
| `response_message` | str | Customer-facing response |
| `optional_followup_question` | str \| null | Clarification question (triggers multi-step) |

---

## Reward Design

The reward is a dense, continuous scalar in **[0.0, 1.0]** composed of four components:

| Component | Max | Description |
|---|---|---|
| `category_score` | 0.4 | Correct category classification |
| `priority_score` | 0.3 | Correct priority (partial credit for off-by-one) |
| `response_quality_score` | 0.2 | Response relevance, clarity, and completeness |
| `customer_satisfaction_score` | 0.1 | Tone + resolution effectiveness + correctness bonus |

### Penalties (applied after summing)

| Penalty | Value | Trigger |
|---|---|---|
| Wrong category | −0.1 | Predicted category ≠ true category |
| Wrong priority | −0.1 | Predicted priority ≠ true (full miss) |
| Empty response | −0.2 | No response message provided |
| Repeated mistake | −0.1 | Same wrong category/priority as previous step |
| Unnecessarily long | −0.05 | Response > 1200 characters |

The final reward is **clamped to [0.0, 1.0]** regardless of penalties.

### Why Dense Rewards?

Sparse rewards (correct/incorrect only) produce slow, sample-inefficient learning. By rewarding partial correctness (off-by-one priority, keyword coverage in responses, tone), agents receive informative gradient signal at every step, enabling faster convergence.

---

## Task Descriptions

### Easy — `easy_billing`
**Ticket:** Customer was double-charged $29.99 for a subscription and wants a refund.  
**Category:** `billing` | **Priority:** `high`  
**Challenge:** Straightforward — a capable agent should score near-perfect on step 1.

### Medium — `medium_login`
**Ticket:** Customer cannot log in and is not receiving a password reset email. Has a meeting in an hour.  
**Category:** `account` | **Priority:** `high`  
**Challenge:** Ambiguous — could be classified as `technical` (email delivery) or `account` (login). The urgency requires `high` priority. Agents that ask a clarification question about spam folders will receive a helpful answer.

### Hard — `hard_multi_issue`
**Ticket:** Customer reports three issues in vague language: app crashes on export, an unexpected invoice charge, and a missing settings section.  
**Category:** `technical` | **Priority:** `high`  
**Challenge:** Multi-issue, vague language. The dominant issue is the crash (technical). Agents must identify all issues in their response, ask targeted clarifying questions, and avoid misclassifying as billing.

---

## Setup Instructions

### Local (no Docker)

```bash
# 1. Clone the repo
git clone https://github.com/yourname/SupportTicketEnv
cd SupportTicketEnv

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r support_ticket_env/requirements.txt

# 4. Set environment variables
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="hf_your_token_here"   # or OPENAI_API_KEY

# 5. Run inference
python support_ticket_env/inference.py
```

### Docker

```bash
# Build the image
docker build -t support-ticket-env .

# Run with environment variables
docker run --rm \
  -e API_BASE_URL=https://api.openai.com/v1 \
  -e MODEL_NAME=gpt-4o-mini \
  -e HF_TOKEN=hf_your_token_here \
  support-ticket-env
```

### HuggingFace Inference Endpoints

```bash
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.3"
export HF_TOKEN="hf_your_token_here"
python support_ticket_env/inference.py
```

---

## Example Usage (Python API)

```python
from support_ticket_env import SupportTicketEnv, Action

env = SupportTicketEnv()

# Start an episode
obs = env.reset("easy_billing")
print(obs.ticket_text)

# Step 1: Classify and respond
action = Action(
    predicted_category="billing",
    predicted_priority="high",
    response_message=(
        "I sincerely apologize for the duplicate charge. I can see two charges "
        "of $29.99 on your account. I'll process a full refund immediately — "
        "please allow 3-5 business days for it to appear."
    ),
    optional_followup_question=None,
)

obs, reward, done, info = env.step(action)
print(f"Reward: {reward.total:.4f}")
print(f"Breakdown: {reward.breakdown}")
print(f"Done: {done}")

# Inspect full state
print(env.state())
```

---

## Example Inference Output

```
[START] task=easy_billing env=SupportTicketEnv model=gpt-4o-mini
[STEP] step=1 action={"category": "billing", "priority": "high", "has_followup": false} reward=0.94 done=true error=null
[END] success=true steps=1 score=0.94 rewards=0.94

[START] task=medium_login env=SupportTicketEnv model=gpt-4o-mini
[STEP] step=1 action={"category": "account", "priority": "high", "has_followup": true} reward=0.71 done=false error=null
[STEP] step=2 action={"category": "account", "priority": "high", "has_followup": false} reward=0.89 done=true error=null
[END] success=true steps=2 score=0.80 rewards=0.71,0.89

[START] task=hard_multi_issue env=SupportTicketEnv model=gpt-4o-mini
[STEP] step=1 action={"category": "technical", "priority": "high", "has_followup": true} reward=0.65 done=false error=null
[STEP] step=2 action={"category": "technical", "priority": "high", "has_followup": false} reward=0.82 done=true error=null
[END] success=true steps=2 score=0.74 rewards=0.65,0.82
```

---

## Baseline Performance

| Task | Random Agent | GPT-4o-mini | GPT-4o |
|---|---|---|---|
| `easy_billing` | ~0.15 | ~0.90 | ~0.95 |
| `medium_login` | ~0.12 | ~0.75 | ~0.88 |
| `hard_multi_issue` | ~0.08 | ~0.65 | ~0.82 |

*Baselines are approximate. Environment is fully deterministic — same inputs always produce same outputs.*

---

## File Structure

```
support_ticket_env/
├── __init__.py          # Package exports
├── env.py               # SupportTicketEnv — main environment class
├── models.py            # Pydantic models: Observation, Action, Reward
├── tasks.py             # Deterministic task definitions (easy, medium, hard)
├── grader.py            # Deterministic reward computation
├── inference.py         # LLM inference script (OpenAI-compatible)
├── requirements.txt     # Python dependencies
├── Dockerfile           # Container build
├── openenv.yaml         # OpenEnv spec metadata
└── README.md            # This file
```

---

## License

MIT License — free to use, modify, and distribute.
