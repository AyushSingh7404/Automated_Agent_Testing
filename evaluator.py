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

from config import CANCELLATION_WINDOW_HOURS, EVALUATOR_MODEL, OPENAI_API_KEY, REQUEST_TIMEOUT, RESCHEDULING_WINDOW_HOURS
from runner import ConversationResult
from test_cases.cases import TestCase

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = f"""\
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

2. SILENT VERIFICATION AND NEXT_STEPS TOOL CALLS
   The agent verifies patient identity (phone number and date of birth) silently via
   internal tool calls that are NOT visible in the transcript. You will NOT see the
   verification happening. Assume verification succeeded if the agent continues the
   conversation normally. Assume verification failed only if the agent explicitly says
   something like "I could not find your record" or "that information does not match".
   Do NOT penalise the agent for not visibly confirming verification steps.
   ADDITIONALLY: In follow-up conversations the agent may have a NEXT_STEPS instruction
   in its continuity memory telling it to immediately fire a tool call (e.g.
   fetch_patient_details) before generating any visible reply. In this case the
   agent's FIRST visible message may already reflect the tool result — it may skip
   asking for details it already retrieved silently. This is CORRECT behaviour.
   Do NOT penalise the agent for not re-asking for information it already has from
   the continuity memory NEXT_STEPS tool call.

3. IDENTITY COLLECTION
   The agent will ask for BOTH phone number AND date of birth for identity verification.
   It may or may not ask for the patient's name — name is optional if the record is
   uniquely identified by phone + DOB. Do NOT penalise for not asking for name.
   DO penalise if the agent skips phone number or date of birth entirely.

4. TOOL CALLS — VERIFY THEM, THIS IS THE DEFAULT EXPECTED MODE
   Every transcript is fetched directly from Retell after the conversation ends,
   so it SHOULD contain TOOL lines (e.g. "TOOL   : CALLED fetch_patient_details(...)"
   and "TOOL   : RESULT for fetch_patient_details | successful=True | ..."),
   interleaved chronologically at the point they actually happened. Assume TOOL
   lines will be present and treat their absence as the rare exception (the
   enrichment fetch genuinely failed for that run), not the norm.

   When TOOL lines ARE present, actively cross-check them — this is the primary
   value of having them, not a nice-to-have:
     - The right tool was called for the situation (e.g. fetch_patient_details
       for an existing patient, create_new_patient for a new one, not the other
       way round).
     - Arguments are not fabricated: a patient_id, appointment_id, provider_id,
       or slot passed into a tool call must have actually appeared in a prior
       tool RESULT or been explicitly given by the patient earlier in the
       transcript — never accept it on faith just because the agent's later
       message sounds confident.
     - Prerequisite ordering is respected: e.g. a booking/slot-lookup tool is
       never called before the patient record was created or found, a
       reschedule/cancel tool is never called before the appointment was
       located via a lookup tool.
     - A failed tool result (successful=False) was actually handled by the
       agent (acknowledged, retried, or escalated) rather than silently ignored
       or treated as if it had succeeded.
   Cross-reference TOOL lines against the expected behaviours checklist
   wherever it references specific tool calls, IDs, or parameters.

   Bias toward trusting the TOOL evidence over guessing from conversational
   phrasing. Most false positives come from penalising text that merely
   *sounds* off (terse, oddly worded, skips a verbal confirmation) when the
   TOOL lines underneath show the correct tool was called with correct,
   non-fabricated arguments and a successful result. If the TOOL evidence
   supports the agent's action, do not fail the agent for phrasing alone.

   ONLY if NO tool lines are present anywhere in the transcript (enrichment
   failed for that run — the exception, not the expected case) fall back to
   evaluating the agent's visible messages alone: trust that a date/slot the
   agent gives came from a real tool result unless the agent visibly
   contradicts itself, since you have no other way to verify it in that case.

   STANDING RULE — APPLIES REGARDLESS OF CHECKLIST WORDING: the fabrication
   check above is a universal check, not something that only applies when the
   expected behaviours checklist happens to mention IDs or hallucination by
   name. Even when the checklist is about something entirely different (e.g.
   reason for visit, phone-number collection), still actively check every
   visible tool call's arguments against real prior tool RESULTS. A fabricated
   argument — e.g. passing a patient's NAME string where a patient_id,
   appointment_id, or provider_id is expected, or any ID that never appeared
   in an earlier tool RESULT — is a critical failure. It must pull the score
   down and lower the verdict (this alone should generally mean FAIL, not a
   high PASS) regardless of whether any checklist item explicitly names
   fabrication. Always reflect it in "findings"; also put it in
   "failed_behaviors" if any checklist item can reasonably be read to cover it
   (e.g. anything about correctness, booking the right patient, or not
   fabricating) — but the score and verdict must reflect it either way. Do NOT
   give a high score just because fabrication wasn't explicitly called out on
   the checklist for this particular test.

5. IDENTITY CROSS-CHECK AND DISAMBIGUATION
   When a tool RESULT contains a patient_name (or equivalent identifying field)
   alongside a patient_id, cross-check that any patient_id/appointment_id the
   agent subsequently uses in a booking/modify/cancel call actually belongs to
   the record matching who the transcript says is being served right now — the
   caller themselves, or the specific named second patient in a multi-patient
   booking. A patient_id being real (genuinely returned by a tool call) is NOT
   sufficient on its own — it must also be the RIGHT real patient's ID for the
   action being taken. FAIL the agent if it acts on (books/reschedules/cancels
   for) a different real patient's record than the one it is actually
   talking to/about at that point in the conversation, even though the ID
   itself is genuine and not fabricated.

   DISAMBIGUATION SCOPE: only require an explicit confirmation step (e.g.
   confirming by first name, "is this you?") when a SINGLE lookup call's tool
   RESULT returned 2+ candidate records (e.g. multiple patients sharing a
   phone number or DOB) — per base_prompt.md's rule to confirm by first name
   before proceeding whenever this happens. Do NOT penalise the agent for
   proceeding directly after a tool result that returned exactly ONE matching
   record, even without an explicit "is this you" question — a clean
   single-record match needs no separate confirmation step.

6. SHARED / COMBINED PHONE NUMBER AND COMBINED-DETAIL DETECTION
   base_prompt.md explicitly forbids two things in multipatient flows: (a)
   asking for or accepting one phone number to cover multiple patients
   ("NEVER ask them about a number that they would like to use for all
   patients... ALWAYS collect phone number separately for all patients"), and
   (b) collecting multiple patients' details together in a single message
   ("never collect details for multiple patient in one message... even if
   related"). Actively scan the transcript for either violation — e.g. the
   agent asking "what number should I use for both of you", silently reusing
   one patient's phone number for another, or asking for two patients'
   name/DOB/phone/reason for visit within the same message — and FAIL the
   relevant checklist item whenever it happens, even if the booking still
   completes successfully overall.

   NOT A VIOLATION — THE upcoming_appointments LOOKUP PATTERN: Section 9's
   appointment confirmation flow explicitly instructs the agent to collect
   each patient's first name and DOB one at a time in conversation, THEN
   internally prepare a first_names list and a dob_list and call
   upcoming_appointments ONCE with both lists aggregated. Seeing a "TOOL   :
   CALLED upcoming_appointments(first_names=[...], dob_list=[...])" line with
   more than one entry in those lists is the CORRECT, prescribed shape of
   that single tool call — it is NOT the same thing as asking the PATIENT a
   combined question, is NOT phone-number reuse, and is NOT a fabrication.
   Only flag this rule against what the agent actually ASKED the patient in
   conversation (visible messages), never against how an upcoming_appointments
   call batches multiple already-separately-collected patients internally.

7. BOOKING CONFIRMATION
   A statement like "Your appointment is booked for [date] at [time]. Is there anything
   else I can help you with?" IS a booking confirmation. Do NOT penalise the agent for
   not making a separate explicit confirmation statement after this.

8. RESCHEDULING / CANCELLATION WITHIN TIME WINDOW
   The clinic has a {RESCHEDULING_WINDOW_HOURS}-hour rescheduling window and a {CANCELLATION_WINDOW_HOURS}-hour cancellation window.
   If a patient requests rescheduling within {RESCHEDULING_WINDOW_HOURS} hours or cancellation within {CANCELLATION_WINDOW_HOURS} hours,
   the CORRECT agent behaviour is to inform the patient this is a last-minute request
   and refer it to another team, offering to create a note. This IS correct behaviour.
   Do NOT penalise the agent for not directly rescheduling/cancelling in these cases.
   Referring to another team and creating a note IS the completion of the task for
   time-window requests — mark this as PASS for those behaviors, not FAIL.

9. SERVICE OR PROVIDER CHANGE / SUBSTITUTION — UNPROMPTED ANNOUNCEMENT ONLY
   The agent can ONLY change the date and time of an existing appointment during
   rescheduling. It CANNOT change the service type or provider of an existing booking.
   If a patient requests a service or provider change during rescheduling, the correct
   agent behaviour is to inform the patient only date/time can change and ALWAYS create
   a task/note for the change request — never call provider_list in this flow, and never
   book a fresh appointment as a workaround. Do NOT penalise the agent for not directly
   changing service/provider on an existing appointment.
   This same "unprompted announcement" principle also applies to service substitution at
   booking time (e.g. silently defaulting a cross-patient-type request to the correct
   default service): flag the agent ONLY when it volunteers or announces a
   substituted/changed service UNPROMPTED (e.g. proactively saying "Invisalign isn't
   available for new patients, I'll book a general consultation instead"). Do NOT flag
   the agent if the patient explicitly asks what service they are booked for (or asks to
   confirm it) and the agent simply answers accurately — truthfully answering a direct
   question is never a violation; only proactively volunteering/announcing the
   substitution unprompted is.

10. RESCHEDULING SLOT UNAVAILABILITY
    If no slot is available for the requested time, the agent should suggest alternative
    slots and offer to create a task/note for the practice team if no suitable slot can
    be found. This is correct behaviour.

11. APPOINTMENT CONFIRMATION AT END
    The agent will NOT proactively list or confirm booked appointments at the end of
    a conversation unless the patient explicitly asks. This is correct behaviour.
    Do NOT penalise the agent for this.

12. REAL APPOINTMENTS FOUND IN LOOKUP
    If the agent reports that appointments exist (e.g. "I found several appointments
    for you"), this is the agent correctly reporting what its lookup returned. The agent
    will never fabricate appointment dates — trust its results. If the test scenario
    assumed no appointments would be found but the agent found real ones, the agent's
    behaviour is correct. Do NOT penalise it.

13. REASONABLE ATTEMPT LIMIT — SCOPE (NO FIXED COUNT)
    The agent limits repeated identity/appointment-lookup attempts to a reasonable
    number before stopping and escalating — creating a task rather than continuing
    indefinitely or proceeding to act against an unverified record. There is NO
    fixed required count (do not require exactly 4 attempts, or any other specific
    number) — judge whether the agent eventually stopped and escalated within a
    reasonable number of tries, not whether it hit a particular number. This does
    NOT apply when multiple appointments exist and the patient is vague about which
    one they want — in that case, the agent correctly presents available
    appointments and waits for the patient to identify the correct one. Do NOT
    penalise the agent for not stopping at any particular turn count when it is
    simply helping the patient identify an existing appointment.

14. PRIOR CONVERSATION THREAD
    If a prior conversation thread is provided, the agent has full access to that
    context. Appointments or actions described as completed in the thread should be
    treated as already done. If the thread says "both appointments have been cancelled",
    the agent correctly reports them as cancelled in the new session. Evaluate the
    agent's use of prior context accordingly.
    FAIL the agent when its first action CONTRADICTS or IGNORES facts already
    established as complete in the prior thread — e.g. calling create_new_patient with
    blank/empty fields for a patient the thread already identifies as an existing,
    registered patient, or re-attempting (rescheduling, cancelling, booking) an action
    the thread explicitly states is already done. A confirmed real example: the thread
    said "Both appointments have been cancelled," yet the agent's first move was a
    nonsensical create_new_patient({{first_name: '', last_name: '', ...}}) tool call —
    ignoring the thread entirely. This is a critical failure, not a stylistic quirk.

15. AGE LIMIT TIMING
    The agent does NOT proactively announce age limits at the start of a conversation.
    It checks age only after the date of birth is provided. This is correct behaviour.
    Do NOT penalise the agent for not mentioning age restrictions before receiving DOB.

16. SIMULATION FAILURES — DO NOT PENALISE AGENT
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

    ADDITIONALLY — CORE SCENARIO NEVER TRIGGERED: read the TEST CASE DESCRIPTION
    above and identify the specific condition or scenario it says should occur
    (e.g. "patient specifically requests Invisalign", "appointment is within the
    cancellation charge window", "patient asks for X"). Actively check whether
    that condition actually appears anywhere in the transcript — not just
    whether surface placeholder artifacts are present. If the transcript never
    actually exercises the condition the description says should happen (e.g.
    the patient never actually asked for the named service, or no appointment
    within the charge window was ever referenced or cancelled), downgrade the
    verdict to WARN at most and note in findings that the test's core scenario
    never triggered — regardless of whether any of the surface-artifact signs
    listed above are present. The agent cannot be credited or blamed for
    behaviour in a scenario that never actually happened.

17. INFRASTRUCTURE / API ERRORS
    If the note at the start of the prompt states that an infrastructure error (API
    timeout, connection failure) occurred during this test, the conversation was cut
    short by an external failure — not by the agent. Do NOT penalise the agent for
    behaviors it had no opportunity to demonstrate. Return WARN at most, and note in
    findings that the result is not valid due to the infrastructure failure.

18. RELATIVE DATE VERIFICATION
    When a patient says "next Monday", "next Tuesday", "tomorrow", or any relative
    date, and the agent books or presents a specific calendar date that plausibly
    matches that relative reference given the apparent conversation date, that is
    CORRECT. Do NOT independently recalculate the date and penalise the agent if
    your calculation differs — the agent has access to real-time calendar context
    you do not have. Trust the agent's date if it is a plausible match.

19. PATIENT EXPLICITLY SELECTS APPOINTMENT TO CONFIRM
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

═══════════════════════════════════════════════════════════════════════════
MANDATORY PRE-VERDICT CHECKS
═══════════════════════════════════════════════════════════════════════════
Perform these three checks explicitly, in order, BEFORE deciding your verdict,
and report the result of each in the corresponding JSON fields below. These are
not soft guidance — the fields you report here are enforced by code after you
respond and can override your "verdict"/"score" — so filling them in
accurately, based on real transcript evidence, is what actually determines the
final result.

STABILITY RULE FOR ALL THREE CHECKS: the same transcript pattern must produce
the same answer every time it is evaluated — these checks exist to catch
clear, nameable problems, not to make a close call. If you cannot point to a
SPECIFIC, concrete piece of transcript evidence for a violation, the correct
answer is false/not-detected, not a guess. Do not flag something as
fabricated/reused/not-triggered based on a vague impression that something
"seems off" — only based on a specific line you can quote in the note. When
genuinely torn between true and false, prefer false and say so plainly in the
note (e.g. "borderline — no specific contradicting evidence found") rather
than flipping a coin; a missed edge case is preferable to an unstable verdict
that would score a correct, identical agent response differently across runs.

CHECK 1 — DID THE TEST'S CORE SCENARIO ACTUALLY HAPPEN?
Read the TEST CASE DESCRIPTION. Identify the specific condition, action, or
tool call it says should occur (e.g. "agent should call get_rescheduling_slot",
"patient introduces a second person named X", "appointment falls within the
charge window", "patient specifically requests Invisalign"). Search the FULL
TRANSCRIPT, including every TOOL line, for direct, specific evidence that this
condition actually happened. Do not infer it happened just because the
conversation sounds fine overall. Common failure patterns to catch:
  - A specific tool call the test exists to check for is never present in any
    "TOOL   : CALLED ..." line (e.g. a rescheduling test where
    get_rescheduling_slot is never called at all).
  - The patient's message contains obvious unresolved placeholder text (e.g.
    "[Brother's Name]", "[insert date]", "[original time]").
  - A person, detail, or request the persona was supposed to introduce never
    actually appears anywhere in the transcript.
Set "scenario_triggered" to true ONLY if you can point to specific transcript
evidence. Otherwise false, with a specific explanation in
"scenario_check_note" of exactly what never happened.

CHECK 2 — IS ANY TOOL CALL ARGUMENT FABRICATED, OR AN ACTION CLAIMED WITH NO TOOL CALL AT ALL?
List every "TOOL   : CALLED ..." line in the transcript, one by one. For EACH
tool call, check every argument value (patient_id, appointment_id,
provider_id, slot time/date, phone number, etc.) against: (a) a value that
appeared in an earlier "TOOL   : RESULT ..." line, or (b) a value the patient
explicitly stated earlier in the transcript. An argument that traces to
neither is fabricated. Watch specifically for:
  - A patient's NAME string (e.g. 'Ruby MAN') passed where a real
    patient_id/appointment_id is expected.
  - A slot time or date passed to book_appointment/modify_appointment that no
    get_available_slot/get_rescheduling_slot RESULT ever returned.
  - A book_appointment (or similar committing) call with no matching prior
    lookup tool call at all.
  - The agent's visible message CLAIMING an action completed (e.g. "your
    appointment is booked") with NO matching tool CALL anywhere in the TOOL
    lines for that action — a claim with no underlying tool call is exactly as
    serious as a fabricated argument, and just as much a CHECK 2 failure.
Set "tool_call_fabrication_detected" to true if ANY call (or claimed action)
fails this check, naming the exact call/claim and argument in
"tool_call_fabrication_note". If the transcript has NO tool lines at all
(enrichment failed for this run), you cannot perform this check — set it to
false and say so in the note.

CHECK 3 — WAS A PHONE NUMBER SILENTLY REUSED ACROSS PATIENTS?
Only relevant when the transcript involves more than one patient, and only
for fetch_patient_details / create_new_patient calls. For every
fetch_patient_details or create_new_patient "TOOL   : CALLED ..." line that
passes a phone number for a specific named patient, check that phone number
argument against what THAT SPECIFIC patient was actually shown stating (or
being given) earlier in the transcript. Check the TOOL CALL argument
directly — do not infer this from how the surrounding dialogue reads, since
the dialogue can look fully compliant (agent appearing to ask separately)
while the tool call underneath still silently reuses patient 1's phone number
for patient 2. Set "phone_number_reuse_detected" to true if any patient's
phone number argument does not match what was actually stated for that
specific patient, naming which patient/call in "phone_number_reuse_note".
This check does NOT apply to upcoming_appointments calls — its aggregated
first_names/dob_list shape for multiple already-separately-collected patients
is the correct, prescribed pattern (see the note under Rule 6), not reuse.

Return your evaluation as a JSON object with exactly these keys:
{{
  "verdict": "PASS" | "WARN" | "FAIL",
  "score": <integer 0-100>,
  "summary": "<2-3 sentence summary of what happened in the conversation>",
  "findings": ["<finding 1>", "<finding 2>", ...],
  "failed_behaviors": ["<behaviour that was missed or wrong>"],
  "scenario_triggered": true | false,
  "scenario_check_note": "<what you checked and found for CHECK 1>",
  "tool_call_fabrication_detected": true | false,
  "tool_call_fabrication_note": "<what you checked and found for CHECK 2>",
  "phone_number_reuse_detected": true | false,
  "phone_number_reuse_note": "<what you checked and found for CHECK 3, or 'not applicable — single patient'>"
}}

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


# Lower number = worse. Used to enforce a verdict cap/floor deterministically
# rather than trusting the model's own "verdict" field in isolation — see the
# MANDATORY PRE-VERDICT CHECKS enforcement in Evaluator.evaluate() below.
_VERDICT_SEVERITY = {"FAIL": 0, "WARN": 1, "PASS": 2}


def _cap_verdict(verdict: str, max_verdict: str) -> str:
    """Return whichever of verdict/max_verdict is worse (lower severity)."""
    current = _VERDICT_SEVERITY.get(verdict, 1)
    cap = _VERDICT_SEVERITY.get(max_verdict, 1)
    return verdict if current <= cap else max_verdict


class Evaluator:
    def __init__(self) -> None:
        self._llm = ChatOpenAI(
            model=EVALUATOR_MODEL,
            api_key=OPENAI_API_KEY,
            temperature=0,
            timeout=REQUEST_TIMEOUT,   # without this, a stalled OpenAI call can hang for minutes with no error
            max_retries=2,
            seed=4242,   # best-effort determinism (OpenAI: same seed + same inputs -> mostly-same output);
                         # temperature=0 alone does not guarantee this on its own
        )

    def evaluate(self, test_case: TestCase, result: ConversationResult) -> EvaluationResult:
        logger.info("  Evaluating %s ...", test_case.id)

        # Infrastructure failure — conversation was cut short by API error, not agent error
        if result.infra_error:
            return EvaluationResult(
                verdict="API_ERROR",
                score=0,
                summary="Test ended due to an infrastructure failure (API timeout or connection error). Agent behaviour could not be evaluated.",
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

            verdict = parsed.get("verdict", "FAIL")
            score = int(parsed.get("score", 0))
            findings = list(parsed.get("findings", []))
            failed_behaviors = parsed.get("failed_behaviors", [])

            # ── Deterministic enforcement of the mandatory self-audit fields ──
            # An LLM judge doesn't reliably self-apply free-text rules buried in
            # a long list when the overall dialogue reads as compliant, so these
            # are enforced here in code rather than trusted to the model's own
            # "verdict"/"score" fields alone.
            if parsed.get("scenario_triggered", True) is False:
                note = parsed.get("scenario_check_note") or "Test's core scenario condition did not occur in the transcript."
                findings = [f"[ENFORCED] Core scenario never triggered: {note}"] + findings
                verdict = _cap_verdict(verdict, "WARN")
                score = min(score, 75)

            if parsed.get("tool_call_fabrication_detected", False) is True:
                note = parsed.get("tool_call_fabrication_note") or "A tool call argument could not be traced to a real prior tool result or patient statement."
                findings = [f"[ENFORCED] Fabricated tool call argument: {note}"] + findings
                verdict = "FAIL"
                score = min(score, 35)

            if parsed.get("phone_number_reuse_detected", False) is True:
                note = parsed.get("phone_number_reuse_note") or "A phone number tool-call argument did not match what that patient actually stated."
                findings = [f"[ENFORCED] Phone number silently reused across patients: {note}"] + findings
                verdict = "FAIL"
                score = min(score, 40)

            return EvaluationResult(
                verdict=verdict,
                score=score,
                summary=parsed.get("summary", ""),
                findings=findings,
                failed_behaviors=failed_behaviors,
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
