"""
config.py
──────────────────────────────────────────
Central configuration for the Agent Testing framework.
All API endpoints, credentials, and test data in one place.
"""
import os
import random

# ── API ───────────────────────────────────────────────────────────────────────
FORWARD_API_URL   = "https://betaaiwebsitechatwidget.adit.com/engage/forward-to-agent"
# FORWARD_API_URL   = "http://localhost:8000/engage/forward-to-agent"
LOCATION_ID       = "3ec74eb5-79df-49c5-818f-661b67272725"
ORGANIZATION_ID   = "ed0b5b16-dc50-4600-901e-0eae1fc2b9d2"
CONVERSATION_ID   = "6c999e6e-5f7a-49cc-a8a0-307a4c17cb0d"  # static ID used for all test conversations

API_TOKEN         = os.getenv("API_ACCESS_TOKEN",)

API_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_TOKEN}",
}

# ── OpenAI ────────────────────────────────────────────────────────────────────
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
PATIENT_MODEL   = "gpt-4o-mini"   # simulates the patient
EVALUATOR_MODEL = "gpt-4o"        # evaluates the transcript

# ── Retell (direct) ─────────────────────────────────────────────────────────────
# Our own backend's /engage/chat-transcript strips tool-call entries before
# returning the transcript, so tool-call enrichment goes straight to Retell's
# own Get Chat API instead, which returns message_with_tool_calls in full.
RETELL_API_KEY      = os.getenv("RETELL_API_KEY", "PASTE_KEY_HERE")
RETELL_GET_CHAT_URL = "https://api.retellai.com/get-chat"

# ── Test patient data ─────────────────────────────────────────────────────────
PATIENT_PHONE = "1112223334"
PATIENT_DOB   = "04 July 2004"   # Abhi Shek's real DOB — do not confuse with SECOND_PATIENT_DOB (2004-04-07), a different real patient on the same shared number
PATIENT_FIRST = "Abhi"
PATIENT_LAST  = "Shek"
PATIENT_NAME  = f"{PATIENT_FIRST} {PATIENT_LAST}"

# Second real registered patient confirmed via fetch_patient_details under the
# SAME phone number as PATIENT_PHONE (a genuine shared-number household) —
# used for multi-patient tests that need two real, disambiguatable records.
SECOND_PATIENT_PHONE = "1112223334"   # same household number as the primary patient
SECOND_PATIENT_DOB   = "07 April 2004"
SECOND_PATIENT_FIRST = "ayush"
SECOND_PATIENT_LAST  = "singh"

# ── Fresh fictional identities (unique per run) ────────────────────────────────
# PROBLEM: every fictional "new patient" / "second patient" persona used to
# hardcode the SAME phone number on every run. The backend never resets
# between test runs (no admin/cleanup endpoint exists — confirmed by checking
# the Retellai_chatagent backend), so once any persona actually completed a
# create_new_patient or book_appointment call, that phone number was no
# longer "new" on the next run. That produced what looked like a behavioural
# regression (agent treats a previously-fictional patient as already
# existing) but was really just accumulated leftover test data — invalidating
# any day-to-day or prompt A/B comparison.
#
# FIX (the part controllable from this test harness): every fictional phone
# number is generated fresh, randomly, at process start, so it has never
# been used by any prior run and cannot collide with leftover backend state.
# Real seeded patients (PATIENT_PHONE / SECOND_PATIENT_PHONE above) are NOT
# generated this way — those are real, stable records the "existing patient"
# tests intentionally depend on staying put across runs.
#
# REMAINING GAP (not fixable from this harness alone): the real seeded
# patient (Abhi Shek / PATIENT_PHONE) still accumulates real appointments
# across runs since nothing here can delete them — there is no backend
# reset/cleanup endpoint to call. Any true apples-to-apples prompt A/B
# comparison still needs an operational DB reset/cleanup for that record
# between comparison runs; this fix only solves the fictional-identity half
# of the problem.
_ISSUED_FICTIONAL_PHONES: set[str] = set()


def unique_fictional_phone() -> str:
    """
    A phone number guaranteed unique within this process run, using the
    reserved 555-prefix ("this is fake") convention plus 7 random digits,
    redrawing on the rare in-run collision. Call once per fictional
    patient and reuse the returned value everywhere that patient's number
    is referenced — do not call it more than once for the same persona.
    """
    while True:
        candidate = "555" + f"{random.randint(0, 9_999_999):07d}"
        if candidate not in _ISSUED_FICTIONAL_PHONES:
            _ISSUED_FICTIONAL_PHONES.add(candidate)
            return candidate


# Fictional identity for "brand-new, never-seen-before patient" personas.
# PATIENT_PHONE/PATIENT_DOB belong to a REAL registered record (Abhi Shek) —
# a "new patient" persona must NOT reuse them, or it describes a brand-new
# person with the exact phone+DOB of a real existing patient, which produces
# spurious "already exists"/wrong-record failures. This is a wholly separate,
# unregistered identity reserved for that scenario only, and now fresh every run.
NEW_PATIENT_PHONE = unique_fictional_phone()
NEW_PATIENT_DOB   = "18 March 1992"

# ── Available providers ───────────────────────────────────────────────────────
PROVIDERS = [
    "Dr. Mahfooz Alam",
    "Dr. Ayush Singh",
]

# ── Practice rules (used in test personas and evaluator) ──────────────────────
RESCHEDULING_WINDOW_HOURS = 16      # last-minute rescheduling threshold
CANCELLATION_WINDOW_HOURS  = 24      # last-minute cancellation threshold
RESCHEDULING_CHARGE: str   = "10"     # e.g. "$50" — leave empty if charges not configured
CANCELLATION_CHARGE: str   = "25"     # e.g. "$75" — leave empty if charges not configured
MIN_AGE_LIMIT = 2    # youngest age accepted for booking (in years) — TC_006/TC_007
MAX_AGE_LIMIT = 80   # oldest age accepted for booking (in years) — TC_008

# ── Runner settings ───────────────────────────────────────────────────────────
DEFAULT_MAX_TURNS    = 17     # max back-and-forth turns per test case (+5 from original 12)
REQUEST_TIMEOUT      = 60    # seconds per API call (longer — agent may chain multiple tools)
TURN_DELAY_SECONDS   = 3.0   # pause between turns
EMPTY_REPLY_WAIT     = 3.0   # wait before retrying on empty agent response
EMPTY_REPLY_RETRIES  = 2     # max retries on empty response before moving on

# After a test conversation ends, the runner fetches the real chat-transcript
# (tool calls included) from Retell to enrich the transcript before evaluation.
# Retell's post-chat pipeline (it also generates chat_analysis server-side) can
# take a while to finish attaching message_with_tool_calls, so the runner
# retries rather than trusting a single fixed wait: sleep this long, try, and
# if the data isn't ready yet, sleep and try again up to FETCH_RETRIES times.
CHAT_TRANSCRIPT_FETCH_DELAY   = 5.0   # seconds between each fetch attempt
CHAT_TRANSCRIPT_FETCH_RETRIES = 6     # up to ~30s total before giving up and keeping the locally-built transcript
