# New Chat Prompt — Agent Testing Framework

Use this prompt verbatim at the start of a new Claude session working on the agent testing framework:

---

You are helping maintain and extend an **automated end-to-end testing framework** for a dental clinic AI chat agent.

**First, read these two files in full before doing anything:**
1. `C:\Users\ayush.singh_adit\Desktop\Claude_Test\Agent_testing\project_context.md` — full architecture, how the framework works, all test case IDs, critical evaluator rules, and known agent behaviours
2. `C:\Users\ayush.singh_adit\Desktop\Claude_Test\Agent_testing\changes.md` — every change made so far and why

**After reading those, here is additional context:**

---

## What This Framework Tests

The framework tests a **Website Chat Agent** — an AI receptionist for dental practices built on Retell AI. The agent handles appointment booking, rescheduling, cancellation, and patient queries via a chat widget.

The live API being tested is:
```
POST https://betaaiwebsitechatwidget.adit.com/engage/forward-to-agent
Bearer token: 2f81fe55
```

The agent ID currently in use: `agent_218003e350c39e8dc095fb7534`

---

## The Test Patient

A registered patient is used for all identity-dependent tests:
- **Name:** ANT MAN
- **Phone:** 1112223334
- **DOB:** 07 April 2004

**Available providers:** Lucas Bennett, Ethan Brooks, Clinic2 Provider, clinic3 provider

---

## Key Design Decisions

**Patient simulation:** GPT-4o-mini simulates the patient. Each test case has a `patient_persona` string that gives the simulator detailed instructions. The simulator is sometimes imperfect — if it drifts from the persona, that is a simulation failure, not an agent failure.

**Empty reply handling:** The agent sometimes returns an empty `agent_response` when it is still resolving tool calls internally. The runner waits 3 seconds and retries up to 2 times before giving up.

**Agent closing detection:** Retell ends the chat after the agent sends a farewell. Sending another message after that causes HTTP 400 "Chat already ended". The runner detects closing phrases in the agent's reply and stops immediately.

**Evaluator rules:** The LLM evaluator has 14 critical rules baked into its system prompt. The most important: evaluate ONLY against the expected behaviours checklist — never invent extra criteria. The full list is in `project_context.md`.

**Service/provider change during rescheduling:** The agent can ONLY change the date/time of an existing booking. It cannot change the service type or provider. Offering a note or a new booking is the correct behaviour.

**Last-minute windows:** Rescheduling within 48 hours or cancellation within 24 hours → agent refers to team and creates a note. This is correct behaviour.

---

## Files to Know

| File | What it does |
|---|---|
| `run_tests.py` | CLI entry point |
| `config.py` | API URL, agent ID, credentials, test patient data, timing settings |
| `runner.py` | Drives conversations, GPT simulates patient, handles retries and closing detection |
| `evaluator.py` | LLM judge — 14-rule system prompt, returns structured JSON verdict |
| `reporter.py` | Markdown report generator — includes prior thread + transcript per test |
| `test_cases/cases.py` | 45 TestCase definitions |

---

Always check `project_context.md` for the "Known Agent Behaviours" section and `changes.md` for "What To Do Next" before starting any task.
