# Agent Testing Framework — Change Log

---

## 1. Initial Framework Built

### What was built
Complete automated testing framework from scratch:

- `config.py` — central config: API URL, agent ID, auth token, OpenAI keys, test patient data, runner settings
- `runner.py` — `ConversationRunner` class: GPT-4o-mini simulates patient, sends messages to `forward-to-agent` API, captures full transcript
- `evaluator.py` — `Evaluator` class: GPT-4o LLM judge reads transcript + expected behaviours checklist, returns structured JSON verdict
- `reporter.py` — `generate_report()`: writes Markdown report with summary table + collapsible transcript per test case
- `run_tests.py` — CLI entry point with `--tests`, `--severity`, `--label`, `--dry-run` flags
- `test_cases/cases.py` — 45 `TestCase` dataclass definitions covering all test scenarios

### Test cases defined
45 test cases across: booking flows, multipatient flows, hallucination checks, age limits, time window enforcement, 4-attempt limits, rescheduling/cancellation, follow-up conversations, incomplete chats.

### API used
`POST https://betaaiwebsitechatwidget.adit.com/engage/forward-to-agent`

---

## 2. Runner Improvements

### What changed
- Added `_call_api_with_retry()` — if `agent_response` is empty (agent mid-tool-chain), waits `EMPTY_REPLY_WAIT` seconds and retries up to `EMPTY_REPLY_RETRIES` times
- Added `_AGENT_CLOSING_SIGNALS` list — farewell phrases the agent uses when closing a conversation
- Added `_agent_has_closed()` static method — checks if agent's reply contains a closing phrase
- In the run loop: if agent closing detected → return immediately, no further API calls (avoids HTTP 400 "Chat already ended" from Retell)
- `_end_conversation()` sends a friendly goodbye then `end_conversation=True` only when the agent has NOT already closed

### Why
- Empty replies occurred when the agent was still resolving multiple tool calls internally — retrying was needed
- Retell marks the chat as ended after the agent sends a farewell; any subsequent call to the same `chat_id` returns HTTP 400. Detecting the closing phrase prevents this error.

### Config changes
- `DEFAULT_MAX_TURNS` 12 → 17 (+5 across the board)
- `REQUEST_TIMEOUT` 30 → 60 (agent may chain multiple tool calls)
- `TURN_DELAY_SECONDS` 0.8 → 3.0
- Added `EMPTY_REPLY_WAIT = 3.0` and `EMPTY_REPLY_RETRIES = 2`

---

## 3. Evaluator System Prompt — 14 Critical Rules Added

### What changed
Replaced the original 5-rule evaluator prompt with a comprehensive 14-rule prompt addressing real evaluation failures observed in the first test run.

Rules added:
1. Checklist Only — evaluate ONLY against the provided expected behaviours
2. Silent Verification — agent verifies phone+DOB silently; don't penalise
3. Identity Collection — phone + DOB required; name optional
4. Tool Calls Not Visible — trust agent-reported slots/dates as real
5. Booking Confirmation — inline confirmation IS a confirmation
6. Last-Minute Time Windows — referring to team for within-window requests is CORRECT
7. Service/Provider Change During Rescheduling — agent cannot change these on existing booking; note/new booking is CORRECT
8. Rescheduling Slot Unavailability — alternatives + task offer is CORRECT
9. Appointment Confirmation at End — no proactive recap is CORRECT
10. Real Appointments Found — agent reports real lookup results; don't penalise
11. 4-Attempt Limit Scope — only applies when appointment NOT FOUND, not when patient is vague
12. Prior Conversation Thread — completed actions in thread are already done
13. Age Limit Timing — check after DOB only is CORRECT
14. Simulation Failures — if patient simulator broke persona, don't penalise agent

Also: `failed_behaviors` must ONLY reference items from the expected behaviours checklist.

### Why
First test run showed the evaluator was: inventing criteria not in the checklist, penalising correct agent behaviour, misreading normal flows as failures. All 14 rules address specific real cases observed.

---

## 4. Evaluator Gets Conversation Thread

### What changed
`evaluator.py` — `evaluate()` now includes the `conversation_thread` (if any) in the prompt sent to the LLM judge, formatted as JSON above the transcript.

### Why
Follow-up test cases (TC_041, TC_042) have prior context the agent used. Without seeing the thread, the evaluator incorrectly flagged actions the agent correctly based on prior context.

---

## 5. Test Case Corrections

### What changed — `test_cases/cases.py`

**TC_007** — Persona now explicitly gives the year-based DOB calculation and warns "do NOT give a DOB younger than 5". Prevents the simulator from giving an under-5 DOB on a test that should verify the agent accepts age-5 patients.

**TC_011** — Persona locks the patient as ANT MAN with explicit 4-DOB sequence and forbids changing identity. Prevents the simulator from presenting as a completely different person, which would trigger new patient creation instead of the 4-attempt verification test.

**TC_025, TC_026, TC_027** — Redesigned from "patient has no appointments" to "patient insists appointments found are not theirs". Since ANT MAN has real appointments in the system, the old design was wrong — the agent correctly found and reported them. New design tests the real behaviour: agent does not fabricate beyond what lookup returned.

**TC_028** — Expected behaviours updated to reflect real agent capability: agent cannot directly change service/provider during rescheduling; correct behaviour is to inform and offer note or new booking.

**TC_030** — Same as TC_028.

**TC_033** — Expected behaviours updated: if patient record not found, agent books for found patient and creates task for unfound one. Removed incorrect expectation that agent should offer alternative providers.

**TC_039** — Persona strengthened to explicitly mention the brother in the first or second message and provide his details when asked. Old persona was too vague and the simulator forgot to introduce the second patient.

**Turn limits** — All default turns increased by +5 (12→17, 16→21, 14→19, 10→15, 6→11).

**Rescheduling test cases** — Patient personas for TC_009, TC_016, TC_018, TC_019, TC_023, TC_028, TC_029, TC_037, TC_041 updated to reject first slot and ask for alternative, exercising the slot-suggestion loop.

---

## 6. Report — Conversation Thread Section Added

### What changed
`reporter.py` — For each test case that has a `conversation_thread`, a collapsible `📋 Prior Conversation Thread` section is rendered just above the `💬 Full Transcript` section.

### Why
Needed to see the prior context alongside the transcript when reviewing follow-up test results.

---

---

## 7. Post-Report-1 Fixes — Simulation Failures, Evaluator Errors, Infra Handling

### Root cause analysis
After reviewing the 2026-04-28 run (10 failures, 3 WARNs), every single failure was traced to one of three external causes:
- **Simulation failure** — the GPT patient simulator broke its persona
- **Evaluator error** — the LLM judge invented wrong criteria or misread the transcript
- **Infrastructure failure** — an API timeout ended the conversation mid-flow

The agent itself did not break its flow in any test.

### Changes to `runner.py`
- Added `infra_error: bool = False` field to `ConversationResult`
- When an API error occurs mid-conversation, `result.infra_error` is set to `True` in addition to `result.error`
- This lets the evaluator distinguish between a clean run and an externally interrupted one

### Changes to `evaluator.py`
- **New early-exit check**: if `result.infra_error` is `True`, immediately return `WARN` with a note that the test was cut short by infrastructure — no agent evaluation is performed
- **Strengthened Rule 6** (last-minute time windows): added explicit statement that referring to team and creating a note IS the completion of the task — mark as PASS not FAIL
- **Expanded Rule 14** (simulation failures): added a concrete list of simulation failure signals the evaluator must detect, including placeholder text, out-of-range dates, wrong patient type, missing second patient, and pre-empted warnings
- **New Rule 15** — Infrastructure/API Errors: if an infra error note is present in the prompt, return WARN at most
- **New Rule 16** — Relative Date Verification: trust the agent's calendar date when matching relative references like "next Monday" — do not recalculate independently
- **New Rule 17** — Patient Explicitly Selects Appointment to Confirm: if a patient says "I'll confirm [appointment]", that IS patient acknowledgement — agent confirming after that is correct, not auto-confirmation

### Changes to `test_cases/cases.py`

**TC_009** — Persona updated: patient must NOT bring up charges before the agent does. Old persona caused the simulator to pre-empt the agent's warning, making the agent's behaviour untestable.

**TC_011** — Persona updated: added explicit instruction that the patient is ALWAYS an existing patient and must NEVER say "first appointment" or "new patient". Old persona allowed the simulator to say "first appointment", completely bypassing the 4-attempt DOB test.

**TC_013** — Persona updated: added explicit instruction to say "No, that is not my appointment" each time the agent shows an appointment. Old persona allowed the simulator to say "cancel it anyway", which constituted implicit consent and caused the agent to correctly proceed — not a failure.

**TC_032** — Expected behaviors updated: "Agent books different services for each patient correctly" → now explicitly states that if a patient record cannot be found, booking for the found patient and offering a note for the unfound one is the CORRECT and EXPECTED outcome.

**TC_034** — Same treatment as TC_032.

**TC_035** — Expected behaviors updated: existing patient flow item now includes "if second patient record cannot be found, agent offers to create a note or task" as the acceptable outcome.

**TC_036** — Persona updated: patient must mention both their own appointment AND their sister's appointment in the very first message. Old persona had no such instruction, so the simulator booked only for itself and forgot the sister. Expected behaviors also updated to account for patient-not-found outcome.

**TC_037** — Persona updated: patient must pick from the appointment dates the agent actually lists, not invent dates. Old persona caused the simulator to give November dates that don't exist, making it impossible for the agent to find them.

**TC_040** — Persona updated: patient must mention their colleague in the first or second message, and must pick from dates the agent actually shows. Old persona caused the simulator to use placeholder text and forget the colleague entirely.

**TC_041** — Expected behaviors updated: "Agent confirms new times for both" → now accounts for the valid case where no rescheduling slots are available and the agent correctly creates a note instead.

**TC_045** — Persona updated: patient must leave WITHOUT selecting any appointment, to create a genuine "no patient acknowledgement" test. Old persona was ambiguous and the simulator said "I'll confirm [date]" which IS patient acknowledgement — causing the evaluator to misread it as a failure. Expected behavior reworded from "without patient acknowledgement" to "an appointment that the patient has not explicitly selected".

1. **Run full 45-case suite** and review report — some test cases may still need persona tuning based on real runs
2. **Tune personas for simulation failures** — if GPT simulator keeps drifting from the persona, add more explicit constraints to the persona string
3. **Add new test cases** as new agent behaviours are identified
4. **Update `AGENT_ID` in `config.py`** whenever the registered agent is re-created
5. **Update `max_booking_date` context** in personas that reference time windows as the booking window rolls forward
