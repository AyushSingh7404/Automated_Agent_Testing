"""
config.py
──────────────────────────────────────────
Central configuration for the Agent Testing framework.
All API endpoints, credentials, and test data in one place.
"""
import os

# ── API ───────────────────────────────────────────────────────────────────────
# FORWARD_API_URL   = "https://betaaiwebsitechatwidget.adit.com/engage/forward-to-agent"
FORWARD_API_URL   = "http://localhost:8000/engage/forward-to-agent"
LOCATION_ID       = "e235fadd-af83-41bb-90cd-bc17da9293d6"
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

# ── Test patient data ─────────────────────────────────────────────────────────
PATIENT_PHONE = "1112223334"
PATIENT_DOB   = "07 April 2004"
PATIENT_FIRST = "ANT"
PATIENT_LAST  = "MAN"
PATIENT_NAME  = f"{PATIENT_FIRST} {PATIENT_LAST}"

# ── Available providers ───────────────────────────────────────────────────────
PROVIDERS = [
    "Default Hygienist",
    "Default Provider",
]

# ── Practice rules (used in test personas and evaluator) ──────────────────────
RESCHEDULING_WINDOW_HOURS = 48      # last-minute rescheduling threshold
CANCELLATION_WINDOW_HOURS  = 24      # last-minute cancellation threshold
RESCHEDULING_CHARGE: str   = ""     # e.g. "$50" — leave empty if charges not configured
CANCELLATION_CHARGE: str   = ""     # e.g. "$75" — leave empty if charges not configured

# ── Runner settings ───────────────────────────────────────────────────────────
DEFAULT_MAX_TURNS    = 17     # max back-and-forth turns per test case (+5 from original 12)
REQUEST_TIMEOUT      = 60    # seconds per API call (longer — agent may chain multiple tools)
TURN_DELAY_SECONDS   = 3.0   # pause between turns
EMPTY_REPLY_WAIT     = 3.0   # wait before retrying on empty agent response
EMPTY_REPLY_RETRIES  = 2     # max retries on empty response before moving on
