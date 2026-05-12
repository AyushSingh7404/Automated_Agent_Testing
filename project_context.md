# Agent Testing Framework — Project Context

This document provides complete context for any new Claude session working on the automated agent testing framework.

---

## What This Project Is

An **automated end-to-end testing framework** for the dental clinic Website Chat Agent. It simulates patient conversations using GPT-4o-mini, sends them to the live `forward-to-agent` API, captures full transcripts, and evaluates them using a GPT-4o LLM judge.

---

## Repository Location

```
C:\Users\ayush.singh_adit\Desktop\Claude_Test\Agent_testing\
```

---

## Project Structure

```
Agent_testing/
├── run_tests.py          ← Main entry point (CLI flags: --tests, --severity, --label, --dry-run)
├── config.py             ← API URLs, credentials, test patient data, runner settings
├── runner.py             ← Drives conversations via forward-to-agent API; GPT simulates patient
├── evaluator.py          ← LLM judge — evaluates transcripts against expected behaviour checklist
├── reporter.py           ← Generates Markdown reports in reports/
├── pyproject.toml        ← Dependencies
├── README.md             ← Usage guide and test case table
├── test_cases/
│   ├── __init__.py
│   └── cases.py          ← All 45 test case definitions (TestCase dataclass)
└── reports/              ← Auto-created; one .md report per run
```

---

## Key Configuration (`config.py`)

```python
FORWARD_API_URL   = "https://betaaiwebsitechatwidget.adit.com/engage/forward-to-agent"
AGENT_ID          = "agent_218003e350c39e8dc095fb7534"
API_TOKEN         = os.getenv("API_ACCESS_TOKEN", "2f81fe55")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
PATIENT_MODEL     = "gpt-4o-mini"   # simulates the patient
EVALUATOR_MODEL   = "gpt-4o"        # judges the transcript

DEFAULT_MAX_TURNS    = 17
REQUEST_TIMEOUT      = 60
TURN_DELAY_SECONDS   = 3.0
EMPTY_REPLY_WAIT     = 3.0
EMPTY_REPLY_RETRIES  = 2
```

**Test patient data (registered in the system):**
- Name: ANT MAN
- Phone: 1112223334
- DOB: 07 April 2004

**Available providers:** Lucas Bennett, Ethan Brooks, Clinic2 Provider, clinic3 provider

---

## How It Works

### 1. Patient Simulation (`runner.py`)
- GPT-4o-mini plays the patient, guided by a per-test-case persona prompt
- Each patient message is sent to `POST /engage/forward-to-agent`
- If `agent_response` comes back empty (agent still resolving tool calls), waits `EMPTY_REPLY_WAIT` seconds and retries up to `EMPTY_REPLY_RETRIES` times
- Loop stops when: patient signals end (`bye`, `that is all`, etc.), agent sends a closing farewell phrase, or `max_turns` is reached
- After the loop, runner sends a friendly goodbye then `end_conversation=True` to cleanly close the Retell session — UNLESS the agent already closed it (which causes HTTP 400 if you try again)

### 2. Agent Closing Detection
The runner detects when the agent has already closed the conversation by scanning the agent's reply for farewell phrases defined in `_AGENT_CLOSING_SIGNALS`. If detected, the runner returns immediately without sending any further messages or end_conversation calls.

### 3. Conversation Thread (Follow-up scenarios)
For test cases with `conversation_thread` set, the prior chat history JSON is sent in the first API call's payload under the `conversation_thread` key. This gives the agent prior context for follow-up scenarios (TC_041, TC_042).

### 4. Evaluation (`evaluator.py`)
- GPT-4o reads the full transcript and the expected behaviours checklist
- Returns a structured JSON verdict: `PASS` / `WARN` / `FAIL`, score 0-100, summary, findings, failed_behaviors
- `failed_behaviors` must ONLY reference items from the checklist — never invented criteria

### 5. Report (`reporter.py`)
- Markdown file in `reports/` with summary table, failures quick-view, and per-test detail
- Each test detail section includes: verdict, score, severity, chat_id, prior conversation thread (if any), full transcript

---

## Critical Evaluator Rules (always in effect)

These rules are baked into the evaluator system prompt and must NOT be changed without understanding their reasoning:

1. **Checklist only** — evaluate only against the provided expected behaviours, never invent extra criteria
2. **Silent verification** — agent verifies phone + DOB silently via tool calls; don't penalise for not confirming verbally
3. **Tool calls not visible** — judge only visible messages; trust agent-reported dates/slots as real tool results
4. **Booking confirmation** — `"Your appointment is booked for [date]. Anything else?"` IS a confirmation
5. **Last-minute time windows** — agent referring to team for within-48h reschedule or within-24h cancel is CORRECT
6. **Service/provider change during rescheduling** — agent CANNOT change service/provider on existing booking; offering note/new booking is CORRECT
7. **Rescheduling slot unavailability** — repeatedly suggesting alternatives + task offer is CORRECT
8. **No proactive appointment listing** — agent doesn't recap bookings at end unless asked; this is CORRECT
9. **Real appointments found** — agent reports what lookup returned; trust it; don't penalise for finding real appointments when test assumed none
10. **4-attempt limit scope** — limit applies when appointment NOT FOUND, not when patient is vague about which one
11. **Prior thread** — actions completed in the conversation thread are already done; agent correctly reports them
12. **Age limit timing** — agent checks age after DOB given, not proactively; this is CORRECT
13. **Simulation failures** — if simulated patient broke persona, evaluate agent on what actually happened

---

## Test Case Structure (`TestCase` dataclass)

```python
@dataclass
class TestCase:
    id: str                              # TC_001 ... TC_045
    name: str
    severity: str                        # "critical" | "warning"
    description: str
    patient_persona: str                 # Prompt fed to GPT patient simulator
    max_turns: int = 17
    conversation_thread: Optional[list] = None   # Prior chat history for follow-up tests
    expected_behaviors: list[str]        # Checklist for evaluator — judge ONLY against these
```

**Turn limits by scenario type:**
- Standard: 17
- Multi-patient or complex flows: 21
- 4-attempt tests: 19
- Follow-up scenarios: 19
- Incomplete chat tests: 11

---

## Test Case IDs

| ID | Name | Severity |
|----|------|----------|
| TC_001 | Fresh Start Single Patient Booking Without Provider | critical |
| TC_002 | Fresh Start Multi Patient Booking Without Provider | critical |
| TC_003 | Fresh Start Existing Patient Booking Without Provider | critical |
| TC_004 | Booking With Specific Provider | critical |
| TC_005 | Booking With Invalid Provider | critical |
| TC_006 | Age Limit — Minimum Age Restriction | critical |
| TC_007 | Fresh Start Minimum Age Limit Booking | critical |
| TC_008 | Fresh Start Maximum Age Limit Booking | warning |
| TC_009 | Last Minute Rescheduling and Charges | critical |
| TC_010 | Last Minute Cancellation and Charges | critical |
| TC_011 | 4 Attempts — Fetch Patient Details | critical |
| TC_012 | 4 Attempts — Upcoming Appointments (Rescheduling) | critical |
| TC_013 | 4 Attempts — Upcoming Appointments (Cancellation) | critical |
| TC_014 | 4 Attempts — Upcoming Appointments (Confirmation) | critical |
| TC_015 | Appointment ID Hallucination — Modify Appointment | critical |
| TC_016 | Appointment ID Hallucination — Rescheduling Slot | critical |
| TC_017 | Patient ID Hallucination — Book Appointment | critical |
| TC_018 | Slot and Time Hallucination — Book Appointment | critical |
| TC_019 | Slot and Time Hallucination — Modify Appointment | critical |
| TC_020 | Provider ID Hallucination — Get Available Slot | critical |
| TC_021 | Provider ID Hallucination — Book Appointment | critical |
| TC_022 | Hallucination Check — Modify Appointment Tool Call | critical |
| TC_023 | Hallucination Check — Get Rescheduling Slot Tool Call | critical |
| TC_024 | Reason for Visit — Agent Asks Proactively | warning |
| TC_025 | Appointment Not Found — Patient Profile Found (Cancellation) | critical |
| TC_026 | Appointment Not Found — Patient Profile Found (Rescheduling) | critical |
| TC_027 | Appointment Not Found — Patient Profile Found (Confirmation) | warning |
| TC_028 | Changes of Service and Provider Name in Rescheduling | critical |
| TC_029 | Provider Change Request While Rescheduling | warning |
| TC_030 | Service Change Request While Rescheduling | warning |
| TC_031 | Multipatient (Same Type) — Same Service Any Provider | critical |
| TC_032 | Multipatient (Same Type) — Different Service Any Provider | critical |
| TC_033 | Multipatient (Same Type) — Same Service Specific Provider | critical |
| TC_034 | Multipatient (Same Type) — Different Service Different Provider | critical |
| TC_035 | Multipatient (Different Type) — Same Service Any Provider | critical |
| TC_036 | Multipatient (Different Type) — Different Service Specific Provider | critical |
| TC_037 | FS Single Patient Rescheduling | critical |
| TC_038 | FS Single Patient Cancellation | critical |
| TC_039 | FS Multipatient Rescheduling / Confirmation | critical |
| TC_040 | FS Multipatient Cancellation | critical |
| TC_041 | Follow Up Multipatient Rescheduling / Confirmation | warning |
| TC_042 | Follow Up Multipatient Cancellation | warning |
| TC_043 | Incomplete Chat — Cancellation | warning |
| TC_044 | Incomplete Chat — Rescheduling | warning |
| TC_045 | Incomplete Chat — Confirmation | warning |

---

## Known Agent Behaviours (do not misread as bugs)

- Agent verifies identity using **phone number + DOB** silently. Name is optional.
- Agent **does not proactively announce age limits** — checks only after DOB is given.
- Agent **cannot change service or provider** on an existing booking during reschedule — it can only change date/time.
- Agent **does not recap bookings** at end of conversation unless asked.
- Rescheduling/cancellation within the time window → agent refers to team and offers a note. This is correct.
- Booking window is 90 days from agent creation date.
- Current max booking date: approximately 2026-05-28.
- Agent will **never fabricate appointment dates or slots** — all dates/slots come from tool call results.

---

## How to Run

```bash
pip install requests langchain-openai langchain-core
export OPENAI_API_KEY="sk-..."
export API_ACCESS_TOKEN="2f81fe55"

# List all test cases without running
python run_tests.py --dry-run

# Run specific tests
python run_tests.py --tests TC_001 TC_004 TC_011

# Run only critical tests
python run_tests.py --severity critical

# Run everything with a label
python run_tests.py --label "sprint_4_regression"
```

Reports are saved to `reports/test_report_<timestamp>.md`.
