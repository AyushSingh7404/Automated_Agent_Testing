"""
config.py
──────────────────────────────────────────
Central configuration for the Agent Testing framework.
All API endpoints, credentials, and test data in one place.
"""
import os

# ── API ───────────────────────────────────────────────────────────────────────
# FORWARD_API_URL = "https://betaaiwebsitechatwidget.adit.com/engage/forward-to-agent"
# AGENT_ID        = "agent_218003e350c39e8dc095fb7534"
FORWARD_API_URL = "http://localhost:8001/engage/forward-to-agent"
AGENT_ID        = "agent_261392226ad29fbf09d6fdda3c"
API_TOKEN       = os.getenv("API_ACCESS_TOKEN",)

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
    "Lucas Bennett",
    "Ethan Brooks",
    "Clinic2 Provider",
    "clinic3 provider",
]

# ── Runner settings ───────────────────────────────────────────────────────────
DEFAULT_MAX_TURNS    = 17     # max back-and-forth turns per test case (+5 from original 12)
REQUEST_TIMEOUT      = 60    # seconds per API call (longer — agent may chain multiple tools)
TURN_DELAY_SECONDS   = 3.0   # pause between turns
EMPTY_REPLY_WAIT     = 3.0   # wait before retrying on empty agent response
EMPTY_REPLY_RETRIES  = 2     # max retries on empty response before moving on
