"""
evaluator.py
──────────────────────────────────────────
LLM judge that evaluates a conversation transcript against the
expected behaviours defined in each TestCase.

Verdict:
  PASS  — all expected behaviours observed
  WARN  — minor deviation, non-critical issue
  FAIL  — critical behaviour missing or incorrect flow detected

Returns a structured EvaluationResult.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import EVALUATOR_MODEL, OPENAI_API_KEY
from runner import ConversationResult
from test_cases.cases import TestCase

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a strict quality evaluator for a dental clinic AI chat agent.
You will be given:
  1. A test case description and its expected behaviours checklist
  2. Optionally, a prior conversation thread (context the agent had access to)
  3. The full conversation transcript between a simulated patient and the agent

Your job is to evaluate whether the AGENT followed the correct conversational flow
based ONLY on what is in the transcript. You are evaluating the AGENT, not the patient.

CRITICAL RULES FOR EVALUATION:

1. CHECKLIST ONLY — NO EXTRA CRITERIA
   Evaluate ONLY against the EXPECTED BEHAVIOURS CHECKLIST provided. Do NOT invent
   additional criteria, do NOT penalise for things not in the checklist, and do NOT
   add your own expectations about timing or phrasing of steps.

2. SILENT VERIFICATION
   The agent verifies patient identity (phone number and date of birth) silently via
   internal tool calls that are NOT visible in the transcript. You will NOT see the
   verification happening. Assume verification succeeded if the agent continues the
   conversation normally. Assume verification failed only if the agent explicitly says
   something like "I could not find your record" or "that information does not match".
   Do NOT penalise the agent for not visibly confirming verification steps.

3. IDENTITY COLLECTION
   The agent will ask for BOTH phone number AND date of birth for identity verification.
   It may or may not ask for the patient's name — name is optional if the record is
   uniquely identified by phone + DOB. Do NOT penalise for not asking for name.
   DO penalise if the agent skips phone number or date of birth entirely.

4. TOOL CALLS NOT VISIBLE
   You cannot see which tools were called or what parameters were passed. Evaluate
   only what the agent says in its visible messages. Do not make assumptions about
   internal tool behaviour. If the transcript shows the agent gave a date or slot,
   assume it came from a real tool result unless the agent visibly contradicts itself.

5. BOOKING CONFIRMATION
   A statement like "Your appointment is booked for [date] at [time]. Is there anything
   else I can help you with?" IS a booking confirmation. Do NOT penalise the agent for
   not making a separate explicit confirmation statement after this.

6. RESCHEDULING / CANCELLATION WITHIN TIME WINDOW
   The clinic has a 48-hour rescheduling window and a 24-hour cancellation window.
   If a patient requests rescheduling within 48 hours or cancellation within 24 hours,
   the CORRECT agent behaviour is to inform the patient this is a last-minute request
   and refer it to another team, offering to create a note. This IS correct behaviour.
   Do NOT penalise the agent for not directly rescheduling/cancelling in these cases.
   Referring to another team and creating a note IS the completion of the task for
   time-window requests — mark this as PASS for those behaviors, not FAIL.

7. SERVICE OR PROVIDER CHANGE DURING RESCHEDULING
   The agent can ONLY change the date and time of an existing appointment during
   rescheduling. It CANNOT change the service type or provider of an existing booking.
   If a patient requests a service or provider change during rescheduling, the correct
   agent behaviour is to: inform the patient this requires team assistance AND offer
   to create a note OR book a fresh appointment with the new details. Both approaches
   are correct. Do NOT penalise the agent for not directly changing service/provider
   on an existing appointment.

8. RESCHEDULING SLOT UNAVAILABILITY
   If no slot is available for the requested time, the agent should suggest alternative
   slots and offer to create a task/note for the practice team if no suitable slot can
   be found. This is correct behaviour.

9. APPOINTMENT CONFIRMATION AT END
   The agent will NOT proactively list or confirm booked appointments at the end of
   a conversation unless the patient explicitly asks. This is correct behaviour.
   Do NOT penalise the agent for this.

10. REAL APPOINTMENTS FOUND IN LOOKUP
    If the agent reports that appointments exist (e.g. "I found several appointments
    for you"), this is the agent correctly reporting what its lookup returned. The agent
    will never fabricate appointment dates — trust its results. If the test scenario
    assumed no appointments would be found but the agent found real ones, the agent's
    behaviour is correct. Do NOT penalise it.

11. 4-ATTEMPT LIMIT — SCOPE
    The 4-attempt limit applies when an appointment CANNOT BE FOUND in the system
    (i.e. after exhausting lookups, nothing matches). It does NOT apply when multiple
    appointments exist and the patient is vague about which one they want — in that
    case, the agent correctly presents available appointments and waits for the patient
    to identify the correct one. Do NOT penalise the agent for not stopping after 4
    turns when it is simply helping the patient identify an existing appointment.

12. PRIOR CONVERSATION THREAD
    If a prior conversation thread is provided, the agent has full access to that
    context. Appointments or actions described as completed in the thread should be
    treated as already done. If the thread says "both appointments have been cancelled",
    the agent correctly reports them as cancelled in the new session. Evaluate the
    agent's use of prior context accordingly.

13. AGE LIMIT TIMING
    The agent does NOT proactively announce age limits at the start of a conversation.
    It checks age only after the date of birth is provided. This is correct behaviour.
    Do NOT penalise the agent for not mentioning age restrictions before receiving DOB.

14. SIMULATION FAILURES — DO NOT PENALISE AGENT
    The patient in this transcript is simulated by another AI. If the patient clearly
    did not follow the test scenario, evaluate the agent on what actually happened
    in the conversation — do not penalise the agent for the simulation's failure.
    Clear signs of simulation failure include:
      - Patient used placeholder text such as "[insert date and time]" or "[original date]"
      - Patient gave appointment dates clearly outside the realistic range (e.g. November
        dates when all known appointments are in April/May)
      - Patient said "first appointment" or "new patient" when the test scenario requires
        an existing patient giving wrong DOBs
      - Patient failed to mention a second person they were explicitly supposed to introduce
      - Patient pre-empted the agent's warning (e.g. mentioned charges before the agent did)
    When any of these signs are present, mark the verdict WARN at most and note the
    simulation failure in findings. Do NOT fail the agent for not completing a flow
    that the simulator prevented from happening.

15. INFRASTRUCTURE / API ERRORS
    If the note at the start of the prompt states that an infrastructure error (API
    timeout, connection failure) occurred during this test, the conversation was cut
    short by an external failure — not by the agent. Do NOT penalise the agent for
    behaviors it had no opportunity to demonstrate. Return WARN at most, and note in
    findings that the result is not valid due to the infrastructure failure.

16. RELATIVE DATE VERIFICATION
    When a patient says "next Monday", "next Tuesday", "tomorrow", or any relative
    date, and the agent books or presents a specific calendar date that plausibly
    matches that relative reference given the apparent conversation date, that is
    CORRECT. Do NOT independently recalculate the date and penalise the agent if
    your calculation differs — the agent has access to real-time calendar context
    you do not have. Trust the agent's date if it is a plausible match.

17. PATIENT EXPLICITLY SELECTS APPOINTMENT TO CONFIRM
    In appointment confirmation flows, if the patient explicitly states which
    appointment they want confirmed (e.g. "I'll confirm the one on April 29",
    "Yes, confirm that appointment", "Please confirm [date/time]"), that IS the
    patient's acknowledgement and authorisation. The agent confirming the
    appointment immediately after such an explicit patient statement is CORRECT
    behaviour — it is NOT auto-confirming without consent. Do NOT penalise this.

Focus on:
- Did the agent ask for phone number AND date of birth for identity verification?
- Did the agent follow the correct flow described in the expected behaviours checklist?
- Did the agent handle edge cases (age limits, charge warnings, attempt limits) correctly?
- Did the agent avoid fabricating information (slots, providers, times)?
- Did the agent end the conversation politely?

Do NOT penalise the agent for wording or tone — only evaluate flow and correctness.

Return your evaluation as a JSON object with exactly these keys:
{
  "verdict": "PASS" | "WARN" | "FAIL",
  "score": <integer 0-100>,
  "summary": "<2-3 sentence summary of what happened in the conversation>",
  "findings": ["<finding 1>", "<finding 2>", ...],
  "failed_behaviors": ["<behaviour that was missed or wrong>"]
}

Rules:
- verdict PASS if all expected behaviours were observed (score >= 85)
- verdict WARN if minor issues or simulation failure (score 60-84)
- verdict FAIL if critical flow broken or important behaviour genuinely missing (score < 60)
- failed_behaviors must ONLY reference items from the EXPECTED BEHAVIOURS CHECKLIST
- Return ONLY the JSON, no markdown fences, no extra text
"""


@dataclass
class EvaluationResult:
    verdict: str                   # PASS | WARN | FAIL
    score: int                     # 0-100
    summary: str
    findings: list[str]
    failed_behaviors: list[str]
    raw_response: str = ""
    error: Optional[str] = None


class Evaluator:
    def __init__(self) -> None:
        self._llm = ChatOpenAI(
            model=EVALUATOR_MODEL,
            api_key=OPENAI_API_KEY,
            temperature=0,
        )

    def evaluate(self, test_case: TestCase, result: ConversationResult) -> EvaluationResult:
        logger.info("  Evaluating %s ...", test_case.id)

        # Infrastructure failure — conversation was cut short by API error, not agent error
        if result.infra_error:
            return EvaluationResult(
                verdict="WARN",
                score=50,
                summary="Test ended due to an infrastructure failure (API timeout or connection error). Agent behaviour could not be fully evaluated.",
                findings=[f"Infrastructure error: {result.error}", "The agent's flow was cut short externally — this is not an agent failure."],
                failed_behaviors=[],
                error=result.error,
            )

        if result.error and not result.transcript:
            return EvaluationResult(
                verdict="FAIL",
                score=0,
                summary="Conversation did not complete due to an API error.",
                findings=[f"Error: {result.error}"],
                failed_behaviors=test_case.expected_behaviors,
                error=result.error,
            )

        transcript_text = result.to_text()
        if not transcript_text.strip():
            return EvaluationResult(
                verdict="FAIL",
                score=0,
                summary="No conversation transcript was captured.",
                findings=["No messages recorded."],
                failed_behaviors=test_case.expected_behaviors,
            )

        thread_section = ""
        if test_case.conversation_thread:
            import json as _json
            thread_section = (
                f"\nPRIOR CONVERSATION THREAD (context the agent had access to):\n"
                f"{_json.dumps(test_case.conversation_thread, indent=2)}\n"
            )

        user_prompt = (
            f"TEST CASE: {test_case.id} — {test_case.name}\n\n"
            f"DESCRIPTION:\n{test_case.description}\n\n"
            f"EXPECTED BEHAVIOURS CHECKLIST:\n"
            + "\n".join(f"- {b}" for b in test_case.expected_behaviors)
            + thread_section
            + f"\n\nFULL TRANSCRIPT:\n{transcript_text}"
        )

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        try:
            response = self._llm.invoke(messages)
            raw = response.content.strip()
            parsed = json.loads(raw)
            return EvaluationResult(
                verdict=parsed.get("verdict", "FAIL"),
                score=int(parsed.get("score", 0)),
                summary=parsed.get("summary", ""),
                findings=parsed.get("findings", []),
                failed_behaviors=parsed.get("failed_behaviors", []),
                raw_response=raw,
            )
        except json.JSONDecodeError as e:
            logger.error("  Evaluator returned invalid JSON: %s", e)
            return EvaluationResult(
                verdict="WARN",
                score=50,
                summary="Evaluator returned malformed JSON — manual review needed.",
                findings=["Evaluator output could not be parsed."],
                failed_behaviors=[],
                raw_response=response.content if "response" in dir() else "",
                error=str(e),
            )
        except Exception as exc:
            logger.error("  Evaluator error: %s", exc)
            return EvaluationResult(
                verdict="FAIL",
                score=0,
                summary="Evaluation failed due to an unexpected error.",
                findings=[str(exc)],
                failed_behaviors=test_case.expected_behaviors,
                error=str(exc),
            )
