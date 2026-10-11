# Agent Testing Framework — Project Context

This document provides complete context for any new Claude session working on the automated agent testing framework.

---

## What This Project Is

An **automated end-to-end testing framework** for the dental clinic Website Chat Agent. It simulates patient conversations using GPT-4o-mini, sends them to the live `forward-to-agent` API, captures full transcripts, and evaluates them using a GPT-4o LLM judge. It also supports evaluating an already-completed transcript directly (paste mode), skipping the live simulated conversation.

---

## Repository Location

```
C:\Users\ayush.singh_adit\Desktop\Claude_Test\Agent_testing\
```

The agent's own system prompt (ground truth for correct behavior) lives in a sibling repo:
```
C:\Users\ayush.singh_adit\Desktop\Claude_Test\Retellai_chatagent\base_prompt.md
```
Always check that file before writing or editing test cases — it is the authoritative source of what the agent is supposed to do, and it changes independently of this project.

---

## Project Structure

```
Agent_testing/
├── run_tests.py          ← Main entry point (CLI flags: --tests, --severity, --label, --dry-run, --paste, --test-id, --chat-id)
├── config.py             ← API URLs, credentials, test patient data, runner settings
├── runner.py             ← Drives conversations via forward-to-agent API; GPT simulates patient
├── evaluator.py          ← LLM judge — evaluates transcripts against expected behaviour checklist
├── reporter.py           ← Generates Markdown reports in reports/
├── paste_mode.py         ← Evaluate an already-completed transcript directly, skipping the live runner
├── pyproject.toml        ← Dependencies
├── README.md             ← Usage guide and test case table
├── test_cases/
│   ├── __init__.py
│   └── cases.py          ← All 58 test case definitions (TestCase dataclass)
└── reports/              ← Auto-created; one .md report per run
```

---

## Key Configuration (`config.py`)

The backend's `/engage/forward-to-agent` endpoint (see `app/models/forward_to_agent.py` in the sibling repo) takes `location_id` + `organization_id` + `conversation_id`, NOT the older `agent_id` shape:

```python
FORWARD_API_URL   = "http://localhost:8000/engage/forward-to-agent"   # point at beta URL for real runs
LOCATION_ID       = "e235fadd-af83-41bb-90cd-bc17da9293d6"
ORGANIZATION_ID   = "ed0b5b16-dc50-4600-901e-0eae1fc2b9d2"
CONVERSATION_ID   = "6c999e6e-5f7a-49cc-a8a0-307a4c17cb0d"  # static ID used for all test conversations
API_TOKEN         = os.getenv("API_ACCESS_TOKEN")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")
PATIENT_MODEL     = "gpt-4o-mini"   # simulates the patient
EVALUATOR_MODEL   = "gpt-4o"        # judges the transcript

DEFAULT_MAX_TURNS    = 17
REQUEST_TIMEOUT      = 60
TURN_DELAY_SECONDS   = 3.0
EMPTY_REPLY_WAIT     = 3.0
EMPTY_REPLY_RETRIES  = 2

RESCHEDULING_WINDOW_HOURS = 48
CANCELLATION_WINDOW_HOURS = 24
```

**Test patient data (registered in the system):**
- Name: ANT MAN (`PATIENT_FIRST`/`PATIENT_LAST` in config.py)
- Phone: 1112223334
- DOB: 07 April 2004

**Configured providers:** `PROVIDERS = ["Default Hygienist", "Default Provider"]` — these are placeholder names read from `LOCATION_ID`'s real seed data, which lives in a DB and cannot be verified from code. **Confirm these match the actual registered providers for that location before relying on provider-specific test results.**

---

## How It Works

### 1. Patient Simulation (`runner.py`)
- GPT-4o-mini plays the patient, guided by a per-test-case persona prompt
- Each patient message is sent to `POST /engage/forward-to-agent` with `location_id` + `organization_id` + `conversation_id` in the payload
- If `agent_response` comes back empty (agent still resolving tool calls), waits `EMPTY_REPLY_WAIT` seconds and retries up to `EMPTY_REPLY_RETRIES` times
- **Exception:** on turn 0 of a follow-up test (`conversation_thread` set), the agent may intentionally return a null `agent_response` while it loads continuity memory and waits for the patient's first real input — this is NOT retried as an empty-reply failure, the runner just proceeds to the next patient turn
- Loop stops when: patient signals end (`bye`, `that is all`, etc.) or `max_turns` is reached
- The agent never closes the chat itself — the runner ALWAYS explicitly ends the session afterward: a friendly goodbye message, then a separate `end_conversation=True` call (no `message` field) to cleanly close the Retell session

### 2. Conversation Thread (Follow-up scenarios)
For test cases with `conversation_thread` set, the prior chat history JSON is sent in the first API call's payload under the `conversation_thread` key. This gives the agent prior context for follow-up scenarios (TC_041, TC_042). Thread messages use `role: "patient" | "ai"` (matches the real conversation_thread format seen in production transcripts).

### 3. Evaluation (`evaluator.py`)
- GPT-4o reads the full transcript and the expected behaviours checklist
- Returns a structured JSON verdict: `PASS` / `WARN` / `FAIL` / `API_ERROR`, score 0-100, summary, findings, failed_behaviors
- `API_ERROR` (score 0, no agent evaluation performed) is returned immediately when the conversation was cut short by an infrastructure failure (API timeout/connection error), distinct from an agent-caused `FAIL`
- `failed_behaviors` must ONLY reference items from the checklist — never invented criteria

### 4. Report (`reporter.py`)
- Markdown file in `reports/` with summary table (including an API-error count/quick-view), failures quick-view, and per-test detail
- Each test detail section includes: verdict, score, severity, chat_id, prior conversation thread (if any), full transcript

### 5. Paste Mode (`paste_mode.py`, wired into `run_tests.py --paste`)
- Evaluates an already-completed transcript against a chosen test case's `expected_behaviors` checklist, entirely skipping `runner.py` (no live GPT-simulated conversation, no live API calls to the agent)
- Accepts either: a JSON array of `{role, content}` turns — including the exact wrapper shape returned by `GET /engage/chat-transcript/{chat_id}` — or plain text with `PATIENT:`/`AGENT:` line prefixes
- Builds a `ConversationResult` directly from the parsed transcript, then calls the same `Evaluator` and `reporter.generate_report()` a normal run uses, so the report format is identical
- Usage: `python run_tests.py --paste transcript.json --test-id TC_031` (or `--paste -` to read from stdin)

---

## Critical Evaluator Rules (always in effect)

These rules are baked into the evaluator system prompt (`evaluator.py`) and must NOT be changed without understanding their reasoning:

1. **Checklist only** — evaluate only against the provided expected behaviours, never invent extra criteria
2. **Silent verification + NEXT_STEPS tool calls** — agent verifies phone + DOB silently via tool calls; in follow-ups the agent's continuity memory may fire a tool call (e.g. fetch_patient_details) before any visible reply, so its first message may already reflect that result — don't penalise for skipping questions it already silently answered
3. **Identity collection** — phone + DOB required; name optional
4. **Tool calls not visible** — judge only visible messages; trust agent-reported dates/slots as real tool results
5. **Booking confirmation** — `"Your appointment is booked for [date]. Anything else?"` IS a confirmation
6. **Last-minute time windows** — agent referring to team for within-48h reschedule or within-24h cancel, INSTEAD of directly rescheduling/cancelling, is CORRECT and counts as task completion
7. **Service/provider change during rescheduling** — agent CANNOT change service/provider on an existing booking and must never call provider_list in this flow; it always creates a task instead (never "books a fresh appointment" as a workaround)
8. **Rescheduling slot unavailability** — repeatedly suggesting alternatives + task offer is CORRECT
9. **No proactive appointment listing** — agent doesn't recap bookings at end unless asked; this is CORRECT
10. **Real appointments found** — agent reports what lookup returned; trust it; don't penalise for finding real appointments when test assumed none
11. **4-attempt limit scope** — limit applies when appointment NOT FOUND, not when patient is vague about which one
12. **Prior thread** — actions completed in the conversation thread are already done; agent correctly reports them
13. **Age limit timing** — agent checks age after DOB given, not proactively; this is CORRECT
14. **Simulation failures** — if simulated patient broke persona (placeholder text, out-of-range dates, wrong patient type, forgot a second patient, pre-empted a warning), evaluate the agent on what actually happened, not what the test intended
15. **Infrastructure/API errors** — handled via the dedicated `API_ERROR` verdict (see above), not folded into WARN/FAIL
16. **Relative date verification** — trust the agent's calendar date for "next Monday"/"tomorrow"/etc.; don't independently recalculate and penalise a mismatch
17. **Patient explicitly selects appointment to confirm** — an explicit patient statement ("confirm the one on April 29") IS acknowledgement; the agent confirming right after is correct, not auto-confirmation

---

## Test Case Structure (`TestCase` dataclass)

```python
@dataclass
class TestCase:
    id: str                              # TC_001 ... TC_058
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
| TC_019 | Slot and Time Hallucination — Modify Appointment (Rescheduling) | critical |
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
| TC_046 | Fresh Start Existing Patient Booking With Specific Provider | critical |
| TC_047 | Emergency Case — Single Patient Appointment Request | critical |
| TC_048 | Emergency Case — Multi Patient Appointment Request | critical |
| TC_049 | Multi Patient Booking — Patients Request to Be Seen Together | critical |
| TC_050 | Multi Patient Booking — Patients Request to Be Seen Separately | critical |
| TC_051 | Multipatient — Phone Number Never Assumed Shared | critical |
| TC_052 | Reason for Visit Asked for Every Patient — Never Defaulted | critical |
| TC_053 | No Re-Asking Details Already Given in the Same Conversation | warning |
| TC_054 | Multi Patient Booking — Same Time Slot, Two Different Services, Booked Simultaneously | critical |
| TC_055 | Insurance Acceptance Flow (I1) — Recommends Booking After Task Created | warning |
| TC_056 | Multipatient — Patient Type Never Volunteered, Agent Must Still Ask | critical |
| TC_057 | Cross-Patient-Type Service Substitution Is Silent, Never Announced | warning |
| TC_058 | Multi-Patient New-Patient Registration — No Field Combined or Silently Omitted | critical |

TC_051–058 were added from a QA-sheet audit (two exported spreadsheets of real production chat transcripts) cross-referenced against real transcripts pulled via `GET /engage/chat-transcript/{chat_id}` and against `base_prompt.md`. TC_028/029/030 were rewritten in the same pass — they previously (incorrectly) expected the agent to offer "book a fresh appointment" as an alternative to a service/provider change during rescheduling; base_prompt.md now unconditionally requires `create_task` instead, and never calling `provider_list` in that flow. Multi-location/location-switching test cases are explicitly out of scope and were not added.

---

## Known Agent Behaviours (do not misread as bugs)

- Agent verifies identity using **phone number + DOB** silently. Name is optional.
- Agent **does not proactively announce age limits** — checks only after DOB is given.
- Agent **cannot change service or provider** on an existing booking during reschedule — it can only change date/time, and must always create a task/note for such a request rather than book a fresh appointment as a workaround.
- Agent **does not recap bookings** at end of conversation unless asked.
- Rescheduling/cancellation within the time window → agent refers to team and offers a note. This is correct, and creating the note (without ever completing the reschedule/cancellation directly) is an acceptable, not just a fallback, outcome.
- For multi-patient bookings, phone number, reason for visit, DOB, and insurance provider must each be collected **separately per patient** — never combined into one message across patients, never assumed shared, never silently omitted for one patient.
- Agent must **never re-ask** a detail (name, reason for visit, phone, DOB) already given earlier in the same conversation.
- Booking window is 90 days from agent creation date.
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

# Paste mode — evaluate an existing/fetched transcript against a test case's
# checklist, skipping the live simulated conversation
python run_tests.py --paste transcript.json --test-id TC_031
python run_tests.py --paste - --test-id TC_001   # read from stdin
```

Reports are saved to `reports/test_report_<timestamp>.md`.
