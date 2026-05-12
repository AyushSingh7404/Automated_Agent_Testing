"""
test_cases/cases.py
──────────────────────────────────────────
All test case definitions.

Each TestCase contains:
  - id                  : unique identifier (TC_001 ... TC_045)
  - name                : human-readable name
  - severity            : "critical" or "warning"
  - description         : what this test is checking
  - patient_persona     : prompt fed to GPT to simulate the patient
  - max_turns           : max conversation turns before force-ending
  - conversation_thread : prior chat history for follow-up scenarios (or None)
  - expected_behaviors  : checklist used by the LLM evaluator
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TestCase:
    id: str
    name: str
    severity: str                          # "critical" | "warning"
    description: str
    patient_persona: str                   # GPT patient simulation prompt
    max_turns: int = 17
    conversation_thread: Optional[list] = None
    expected_behaviors: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Shared patient snippets (reused across personas)
# ─────────────────────────────────────────────────────────────────────────────
_EXISTING_IDENTITY = (
    "Your name is ANT MAN. Your phone number is 1112223334. "
    "Your date of birth is 07 April 2004. You are an EXISTING patient at this clinic."
)
_NEW_IDENTITY = (
    "Your name is ANT MAN. Your phone number is 1112223334. "
    "Your date of birth is 07 April 2004. You are a NEW patient at this clinic."
)
_CHILD_IDENTITY = (
    "You are booking for a child named Tommy MAN, age 3 years old (DOB: 10 Jan 2022). "
    "You are the parent. Your own name is ANT MAN, phone 1112223334."
)
_MAX_AGE_IDENTITY = (
    "You are booking for a patient named ANT MAN, age 95 years old (DOB: 01 Jan 1930). "
    "Phone number 1112223334."
)
_BEHAVIOR = (
    "Respond naturally and briefly as a real patient would. "
    "Do not reveal your full persona upfront — answer only what is asked. "
    "If the agent asks for your phone number, give it. "
    "If the agent asks for your date of birth, give it. "
    "If the agent asks for your name, give it. "
    "Stay in character throughout. "
    "When you feel the task is done or the agent has confirmed everything, say 'bye' or 'that is all, thank you'."
)


# ─────────────────────────────────────────────────────────────────────────────
# Conversation threads for follow-up scenarios
# ─────────────────────────────────────────────────────────────────────────────
_FOLLOWUP_THREAD_MULTI_RESCHEDULE = [
    {
        "date": "2026-04-01",
        "messages": [
            {"role": "patient", "content": "Hi, I want to book appointments for me and my wife"},
            {"role": "practice", "content": "Of course! Are you both new or existing patients?"},
            {"role": "patient", "content": "We are both existing patients"},
            {"role": "practice", "content": "Great! I have booked a General Consultation for ANT MAN on April 5 at 10am and a Cleaning Exam for your wife on April 5 at 11am. Is there anything else?"},
            {"role": "patient", "content": "No that is all, thanks"},
            {"role": "practice", "content": "You're welcome! See you soon."},
        ],
    }
]

_FOLLOWUP_THREAD_MULTI_CANCEL = [
    {
        "date": "2026-04-01",
        "messages": [
            {"role": "patient", "content": "I need to cancel my appointment and my wife's appointment"},
            {"role": "practice", "content": "I can help with that. Can I have your date of birth and phone number?"},
            {"role": "patient", "content": "07 April 2004, 1112223334"},
            {"role": "practice", "content": "I found your profile. You have an upcoming General Consultation on April 10. And your wife's appointment?"},
            {"role": "patient", "content": "Yes those are the ones, cancel both"},
            {"role": "practice", "content": "Both appointments have been cancelled. Is there anything else?"},
            {"role": "patient", "content": "No thanks"},
        ],
    }
]

_FOLLOWUP_THREAD_SINGLE_RESCHEDULE = [
    {
        "date": "2026-04-01",
        "messages": [
            {"role": "patient", "content": "I want to reschedule my appointment"},
            {"role": "practice", "content": "Sure, can I have your phone number and date of birth?"},
            {"role": "patient", "content": "1112223334, 07 April 2004"},
            {"role": "practice", "content": "I found your appointment for General Consultation on April 8 at 9am. What date works better for you?"},
            {"role": "patient", "content": "I will call back later"},
            {"role": "practice", "content": "Of course, feel free to reach out anytime!"},
        ],
    }
]

_FOLLOWUP_THREAD_SINGLE_CANCEL = [
    {
        "date": "2026-04-01",
        "messages": [
            {"role": "patient", "content": "I want to cancel my upcoming appointment"},
            {"role": "practice", "content": "I can help. Can I get your phone number and date of birth?"},
            {"role": "patient", "content": "1112223334, 07 April 2004"},
            {"role": "practice", "content": "Found it — you have a General Consultation on April 9. Shall I cancel it?"},
            {"role": "patient", "content": "Actually let me think about it"},
            {"role": "practice", "content": "No problem, reach out when ready!"},
        ],
    }
]


# ─────────────────────────────────────────────────────────────────────────────
# All test cases
# ─────────────────────────────────────────────────────────────────────────────
ALL_TEST_CASES: list[TestCase] = [

    TestCase(
        id="TC_001",
        name="Fresh Start Single Patient Booking Without Provider",
        severity="critical",
        description="New chat — single existing patient wants to book without specifying a provider.",
        patient_persona=(
            f"{_EXISTING_IDENTITY} You want to book a General Consultation appointment. "
            "Do NOT mention a specific provider. If the agent asks for a provider preference, say any is fine. "
            f"{_BEHAVIOR}"
        ),
        max_turns=17,
        expected_behaviors=[
            "Agent asks for reason for visit or service type",
            "Agent asks if patient is new or existing",
            "Agent collects phone number and date of birth to verify identity",
            "Agent proceeds to booking without forcing a provider choice",
            "Agent confirms appointment details before finalising",
        ],
    ),

    TestCase(
        id="TC_002",
        name="Fresh Start Multi Patient Booking Without Provider",
        severity="critical",
        description="New chat — patient wants to book for multiple people without specifying a provider.",
        patient_persona=(
            f"{_EXISTING_IDENTITY} You want to book appointments for yourself and your friend Jake (also an existing patient). "
            "Jake's DOB is 15 March 2000, phone 9998887776. "
            "Neither of you has a provider preference. "
            f"{_BEHAVIOR}"
        ),
        max_turns=21,
        expected_behaviors=[
            "Agent handles multiple patients in one session",
            "Agent collects phone number and date of birth for each patient separately",
            "Agent books appointments for both",
            "Agent confirms all bookings before closing",
        ],
    ),

    TestCase(
        id="TC_003",
        name="Fresh Start Existing Patient Booking Without Provider",
        severity="critical",
        description="New chat — existing patient books without specifying provider.",
        patient_persona=(
            f"{_EXISTING_IDENTITY} You want to book a Cleaning Exam OPD appointment. "
            "No provider preference. "
            f"{_BEHAVIOR}"
        ),
        max_turns=17,
        expected_behaviors=[
            "Agent collects phone number and date of birth to verify existing patient identity",
            "Agent does not create a new patient record for an existing patient",
            "Agent books the correct service",
            "Agent confirms appointment details",
        ],
    ),

    TestCase(
        id="TC_004",
        name="Booking With Specific Provider",
        severity="critical",
        description="Patient requests a booking with a specific available provider (Lucas Bennett).",
        patient_persona=(
            f"{_EXISTING_IDENTITY} You want to book a General Consultation specifically with Dr. Lucas Bennett. "
            "Insist on this provider only. "
            f"{_BEHAVIOR}"
        ),
        max_turns=17,
        expected_behaviors=[
            "Agent attempts to book with the requested provider Lucas Bennett",
            "Agent does not substitute a different provider without asking",
            "Agent confirms the provider name in the booking summary",
        ],
    ),

    TestCase(
        id="TC_005",
        name="Booking With Invalid Provider",
        severity="critical",
        description="Patient requests a booking with a provider name that does not exist.",
        patient_persona=(
            f"{_EXISTING_IDENTITY} You want to book with a provider named Dr. James Wilson. "
            "Insist on this provider. If told unavailable, ask for alternatives. "
            f"{_BEHAVIOR}"
        ),
        max_turns=15,
        expected_behaviors=[
            "Agent informs patient that the requested provider is not available",
            "Agent offers available alternatives instead of fabricating the provider",
            "Agent does not hallucinate a Dr. James Wilson into the booking",
        ],
    ),

    TestCase(
        id="TC_006",
        name="Age Limit — Minimum Age Restriction",
        severity="critical",
        description="Patient tries to book for a child below the minimum age limit (set to 5 years).",
        patient_persona=(
            f"{_CHILD_IDENTITY} You want to book a Routine Cleaning for Tommy. "
            "When asked Tommy's age or DOB, state 10 Jan 2022 (age 3). "
            f"{_BEHAVIOR}"
        ),
        max_turns=15,
        expected_behaviors=[
            "Agent informs that the patient does not meet the minimum age requirement",
            "Agent does not proceed with booking for an under-age patient",
            "Agent handles the situation politely",
        ],
    ),

    TestCase(
        id="TC_007",
        name="Fresh Start Minimum Age Limit Booking",
        severity="critical",
        description="Booking attempt at exactly the minimum age boundary (age 5). Agent should accept and book normally.",
        patient_persona=(
            "You are booking for a child named Sam MAN who is EXACTLY 5 years old. "
            "Sam's date of birth is EXACTLY 5 years ago from today — use the year that makes the child precisely 5. "
            "For example if today is April 2026, Sam's DOB would be April 2021. "
            "Phone 1112223334. You are the parent. "
            "When asked the child's DOB, give the DOB that makes the child exactly 5 years old. "
            "Do NOT give a DOB that makes the child younger than 5. "
            f"{_BEHAVIOR}"
        ),
        max_turns=15,
        expected_behaviors=[
            "Agent accepts the patient at minimum age boundary",
            "Agent proceeds with booking normally",
        ],
    ),

    TestCase(
        id="TC_008",
        name="Fresh Start Maximum Age Limit Booking",
        severity="warning",
        description="Booking attempt for a patient above the maximum age limit.",
        patient_persona=(
            f"{_MAX_AGE_IDENTITY} You want to book a General Consultation. "
            "When asked DOB give 01 Jan 1930. "
            f"{_BEHAVIOR}"
        ),
        max_turns=15,
        expected_behaviors=[
            "Agent informs that the patient exceeds the maximum age limit",
            "Agent handles the situation politely without proceeding",
        ],
    ),

    TestCase(
        id="TC_009",
        name="Last Minute Rescheduling and Charges",
        severity="critical",
        description="Patient tries to reschedule within the rescheduling window (48 hours) — agent should warn about charges.",
        patient_persona=(
            f"{_EXISTING_IDENTITY} You have an appointment tomorrow morning and want to reschedule it to next week. "
            "Do NOT mention or bring up charges yourself before the agent does. "
            "Wait for the agent to inform you about any last-minute rescheduling policy or charges first. "
            "Only if the agent mentions charges or a policy, acknowledge it and confirm you still want to proceed. "
            "Reject the first slot offered and ask for an alternative. Accept the second slot. "
            f"{_BEHAVIOR}"
        ),
        max_turns=17,
        expected_behaviors=[
            "Agent warns patient about last-minute rescheduling charges",
            "Agent mentions the charges amount or policy before proceeding",
            "Agent proceeds only after patient confirms acceptance of charges",
            "Agent offers alternative slots when first slot is rejected",
        ],
    ),

    TestCase(
        id="TC_010",
        name="Last Minute Cancellation and Charges",
        severity="critical",
        description="Patient tries to cancel within the cancellation window (24 hours) — agent should warn about charges.",
        patient_persona=(
            f"{_EXISTING_IDENTITY} You have an appointment in a few hours and want to cancel it. "
            "If warned about last-minute cancellation charges, say you understand and want to cancel anyway. "
            f"{_BEHAVIOR}"
        ),
        max_turns=17,
        expected_behaviors=[
            "Agent warns patient about last-minute cancellation charges",
            "Agent proceeds only after patient confirms they accept the charges",
            "Agent confirms cancellation",
        ],
    ),

    TestCase(
        id="TC_011",
        name="4 Attempts — Fetch Patient Details",
        severity="critical",
        description="Patient ANT MAN provides wrong DOB repeatedly during identity verification — agent should stop after 4 failed attempts.",
        patient_persona=(
            "You are ANT MAN. Your phone number is 1112223334. "
            "You want to book a General Consultation. "
            "You are ALWAYS an EXISTING patient — you have been to this clinic before. "
            "NEVER say 'first appointment', NEVER say 'new patient', NEVER say 'I haven't visited before'. "
            "Always say you are an existing patient if the agent asks. "
            "When the agent asks for your date of birth, give WRONG dates in this exact sequence: "
            "1st time: 01 January 2000. 2nd time: 15 March 1995. 3rd time: 20 June 1988. 4th time: 11 November 2001. "
            "NEVER give the correct DOB (07 April 2004) regardless of how many times the agent asks. "
            "Do NOT change your name or pretend to be a different person. You are always ANT MAN. "
            "Do NOT give any other DOB not listed above. "
            f"{_BEHAVIOR}"
        ),
        max_turns=19,
        expected_behaviors=[
            "Agent allows up to 4 verification attempts",
            "Agent stops and does not proceed after 4 failed attempts",
            "Agent offers to create a task or escalate instead of continuing indefinitely",
        ],
    ),

    TestCase(
        id="TC_012",
        name="4 Attempts — Upcoming Appointments (Rescheduling)",
        severity="critical",
        description="Patient claims to have an appointment that cannot be found — agent tries up to 4 times.",
        patient_persona=(
            f"{_EXISTING_IDENTITY} You want to reschedule your appointment. "
            "When asked for appointment details keep saying you are not sure — give vague or wrong info each time. "
            "Do not confirm any specific appointment the agent mentions. "
            f"{_BEHAVIOR}"
        ),
        max_turns=19,
        expected_behaviors=[
            "Agent attempts to locate appointment up to 4 times",
            "Agent stops after 4 failed attempts and offers alternatives",
            "Agent does not hallucinate an appointment",
        ],
    ),

    TestCase(
        id="TC_013",
        name="4 Attempts — Upcoming Appointments (Cancellation)",
        severity="critical",
        description="Patient wants to cancel but appointment cannot be found after 4 attempts.",
        patient_persona=(
            f"{_EXISTING_IDENTITY} You want to cancel your appointment. "
            "Each time the agent asks for appointment details, give a different vague answer. "
            "Each time the agent shows you an appointment or suggests a specific date and time, "
            "explicitly say 'No, that is not my appointment, I don't recognise that one.' "
            "NEVER say anything like 'cancel it anyway' or 'just cancel it regardless' — "
            "always deny the appointment shown and refuse to confirm it as yours. "
            "Never confirm any appointment the agent presents. "
            f"{_BEHAVIOR}"
        ),
        max_turns=19,
        expected_behaviors=[
            "Agent attempts to locate appointment up to 4 times",
            "Agent stops after 4 failed attempts",
            "Agent offers to create a task for the team to follow up",
        ],
    ),

    TestCase(
        id="TC_014",
        name="4 Attempts — Upcoming Appointments (Confirmation)",
        severity="critical",
        description="Patient wants to confirm appointment but it cannot be located after 4 attempts.",
        patient_persona=(
            f"{_EXISTING_IDENTITY} You want to confirm your upcoming appointment. "
            "Each time the agent asks which appointment, give different/wrong info. "
            "Never successfully identify an appointment. "
            f"{_BEHAVIOR}"
        ),
        max_turns=19,
        expected_behaviors=[
            "Agent limits appointment lookup to 4 attempts",
            "Agent gracefully exits after exhausting attempts",
        ],
    ),

    TestCase(
        id="TC_015",
        name="Appointment ID Hallucination — Modify Appointment",
        severity="critical",
        description="Check that agent does not fabricate an appointment ID when calling modify_appointment.",
        patient_persona=(
            f"{_EXISTING_IDENTITY} You want to reschedule your General Consultation. "
            "Provide your correct identity details. Be cooperative throughout. "
            f"{_BEHAVIOR}"
        ),
        max_turns=17,
        expected_behaviors=[
            "Agent does not invent or guess appointment IDs",
            "Agent uses only appointment IDs returned by the tool lookup",
            "If no appointment found, agent does not fabricate one",
        ],
    ),

    TestCase(
        id="TC_016",
        name="Appointment ID Hallucination — Rescheduling Slot",
        severity="critical",
        description="Check that agent does not fabricate an appointment ID in the get_rescheduling_slot call.",
        patient_persona=(
            f"{_EXISTING_IDENTITY} You want to reschedule your Cleaning Exam to next Friday. "
            "Cooperate with all identity questions. "
            "Reject the first slot offered and ask for another. Accept the second. "
            f"{_BEHAVIOR}"
        ),
        max_turns=17,
        expected_behaviors=[
            "Agent uses only real appointment IDs returned from the lookup",
            "Agent does not fabricate appointment IDs for rescheduling slot calls",
            "Agent offers alternative slots when first is rejected",
        ],
    ),

    TestCase(
        id="TC_017",
        name="Patient ID Hallucination — Book Appointment",
        severity="critical",
        description="Check that agent does not fabricate a patient ID when calling book_appointment.",
        patient_persona=(
            f"{_EXISTING_IDENTITY} You want to book an Emergency Consult. "
            "Provide correct identity when asked. "
            f"{_BEHAVIOR}"
        ),
        max_turns=17,
        expected_behaviors=[
            "Agent uses only patient IDs returned by fetch_patient_details",
            "Agent does not invent a patient ID",
        ],
    ),

    TestCase(
        id="TC_018",
        name="Slot and Time Hallucination — Book Appointment",
        severity="critical",
        description="Check that agent does not fabricate slot or time when booking.",
        patient_persona=(
            f"{_EXISTING_IDENTITY} You want to book a General Consultation for next Monday afternoon. "
            "Be cooperative throughout. "
            "Reject the first slot offered and ask for another time. Accept the second slot. "
            f"{_BEHAVIOR}"
        ),
        max_turns=17,
        expected_behaviors=[
            "Agent presents only slots returned by get_available_slot",
            "Agent does not invent appointment times",
            "Agent offers alternative slots when first is rejected",
            "Agent confirms the slot with patient before booking",
        ],
    ),

    TestCase(
        id="TC_019",
        name="Slot and Time Hallucination — Modify Appointment (Rescheduling)",
        severity="critical",
        description="Check that agent does not fabricate slot or time when rescheduling.",
        patient_persona=(
            f"{_EXISTING_IDENTITY} You want to reschedule your appointment to next Wednesday morning. "
            "Reject the first slot offered and ask for a different time. Accept the second slot offered. "
            f"{_BEHAVIOR}"
        ),
        max_turns=17,
        expected_behaviors=[
            "Agent only offers slots returned by rescheduling slot lookup",
            "Agent does not fabricate available slots",
            "Agent offers alternative when first slot is rejected",
        ],
    ),

    TestCase(
        id="TC_020",
        name="Provider ID Hallucination — Get Available Slot",
        severity="critical",
        description="Check that agent does not pass a fabricated provider ID to get_available_slot.",
        patient_persona=(
            f"{_EXISTING_IDENTITY} You want to book with Ethan Brooks specifically. "
            "Be cooperative throughout. "
            f"{_BEHAVIOR}"
        ),
        max_turns=17,
        expected_behaviors=[
            "Agent uses the correct provider ID for Ethan Brooks",
            "Agent does not fabricate or guess provider IDs",
        ],
    ),

    TestCase(
        id="TC_021",
        name="Provider ID Hallucination — Book Appointment",
        severity="critical",
        description="Check that agent passes the correct provider ID from slot results into book_appointment.",
        patient_persona=(
            f"{_EXISTING_IDENTITY} You want to book a General Consultation with Lucas Bennett. "
            "Confirm the slot the agent offers. "
            f"{_BEHAVIOR}"
        ),
        max_turns=17,
        expected_behaviors=[
            "Agent passes only provider IDs obtained from slot lookup into book_appointment",
            "Agent does not fabricate provider IDs",
        ],
    ),

    TestCase(
        id="TC_022",
        name="Hallucination Check — Modify Appointment Tool Call",
        severity="critical",
        description="Comprehensive check that modify_appointment is called with only real values from prior tool results.",
        patient_persona=(
            f"{_EXISTING_IDENTITY} You want to cancel your upcoming appointment. "
            "Provide correct identity when asked. Confirm the appointment. "
            f"{_BEHAVIOR}"
        ),
        max_turns=17,
        expected_behaviors=[
            "Appointment ID used in cancellation comes from real lookup — not fabricated",
            "Agent confirms cancellation with patient before acting",
            "Agent does not use fabricated parameters",
        ],
    ),

    TestCase(
        id="TC_023",
        name="Hallucination Check — Get Rescheduling Slot Tool Call",
        severity="critical",
        description="Check all parameters passed to get_rescheduling_slot are real values from prior context.",
        patient_persona=(
            f"{_EXISTING_IDENTITY} You want to reschedule your appointment to next Thursday. "
            "Cooperate fully. "
            "Reject the first available slot and ask for another. Accept the second. "
            f"{_BEHAVIOR}"
        ),
        max_turns=17,
        expected_behaviors=[
            "Appointment ID used for rescheduling comes from real lookup",
            "Agent does not fabricate values when looking up rescheduling slots",
            "Agent offers alternative slots when first is rejected",
        ],
    ),

    TestCase(
        id="TC_024",
        name="Reason for Visit — Agent Asks Proactively",
        severity="warning",
        description="Patient does not mention reason for visit upfront — agent should ask before proceeding.",
        patient_persona=(
            f"{_EXISTING_IDENTITY} You just say 'I want to book an appointment' without specifying any service. "
            "Wait for the agent to ask what you need. Only then say General Consultation. "
            f"{_BEHAVIOR}"
        ),
        max_turns=15,
        expected_behaviors=[
            "Agent asks for reason for visit or service type before proceeding",
            "Agent does not assume a service without asking",
        ],
    ),

    TestCase(
        id="TC_025",
        name="Appointment Not Found — Patient Profile Found (Cancellation)",
        severity="critical",
        description="Patient profile exists and agent will find appointments. Patient insists the appointments shown are wrong and they don't recognise them — agent should handle gracefully without fabricating new ones.",
        patient_persona=(
            f"{_EXISTING_IDENTITY} You want to cancel your appointment. "
            "When the agent presents any appointments it found, insist that none of those are yours and you don't recognise them. "
            "Keep insisting across multiple attempts. "
            f"{_BEHAVIOR}"
        ),
        max_turns=15,
        expected_behaviors=[
            "Agent verifies patient identity via phone and DOB",
            "Agent does not fabricate appointments that were not returned by lookup",
            "Agent offers to create a task for the team when appointments cannot be confirmed by patient",
        ],
    ),

    TestCase(
        id="TC_026",
        name="Appointment Not Found — Patient Profile Found (Rescheduling)",
        severity="critical",
        description="Patient profile exists and agent will find appointments. Patient insists none of the found appointments are theirs — agent should handle gracefully.",
        patient_persona=(
            f"{_EXISTING_IDENTITY} You want to reschedule your appointment. "
            "When the agent presents any appointments it found, insist those are not your appointments and you don't recognise any of them. "
            "Ask the agent to double-check. Maintain this across multiple rounds. "
            f"{_BEHAVIOR}"
        ),
        max_turns=15,
        expected_behaviors=[
            "Agent verifies patient identity",
            "Agent does not fabricate appointments beyond what lookup returned",
            "Agent offers to create a task or book a new appointment when patient cannot confirm any appointment",
        ],
    ),

    TestCase(
        id="TC_027",
        name="Appointment Not Found — Patient Profile Found (Confirmation)",
        severity="warning",
        description="Patient profile exists and agent will find appointments. Patient insists none are theirs — agent should not fabricate a new one.",
        patient_persona=(
            f"{_EXISTING_IDENTITY} You want to confirm your upcoming appointment. "
            "When the agent shows any appointments it found, insist none of those are yours. "
            "Ask if they are sure. Do not confirm any appointment shown. "
            f"{_BEHAVIOR}"
        ),
        max_turns=15,
        expected_behaviors=[
            "Agent does not fabricate appointments beyond what lookup returned",
            "Agent offers to create a task when patient cannot confirm any appointment",
        ],
    ),

    TestCase(
        id="TC_028",
        name="Changes of Service and Provider Name in Rescheduling",
        severity="critical",
        description="Patient tries to change service and provider during rescheduling. Agent cannot change service/provider on an existing booking — correct behaviour is to inform patient and offer to create a note or book a new appointment with the new details.",
        patient_persona=(
            f"{_EXISTING_IDENTITY} You have a General Consultation booked and want to reschedule it but also "
            "change the service to Cleaning Exam OPD and switch the provider to Ethan Brooks. "
            "If the agent offers to create a note or book a new appointment with those changes, accept that. "
            f"{_BEHAVIOR}"
        ),
        max_turns=19,
        expected_behaviors=[
            "Agent informs patient that service or provider changes on existing bookings require team assistance or a new booking",
            "Agent offers to create a note or book a fresh appointment with the new service and provider",
            "Agent verifies patient identity via phone and DOB",
        ],
    ),

    TestCase(
        id="TC_029",
        name="Provider Change Request While Rescheduling",
        severity="warning",
        description="Patient specifically requests a different provider during rescheduling.",
        patient_persona=(
            f"{_EXISTING_IDENTITY} You have an appointment with Lucas Bennett and want to reschedule it "
            "but switch to Ethan Brooks instead. "
            "Reject the first slot offered and ask for another. Accept the second. "
            f"{_BEHAVIOR}"
        ),
        max_turns=17,
        expected_behaviors=[
            "Agent acknowledges provider change request",
            "Agent attempts to find slot with new provider",
            "Agent offers alternative slots when first is rejected",
            "Agent confirms new provider in booking summary",
        ],
    ),

    TestCase(
        id="TC_030",
        name="Service Change Request While Rescheduling",
        severity="warning",
        description="Patient wants to change the service type during rescheduling. Agent cannot change service on an existing booking — correct behaviour is to inform and offer a note or a new booking.",
        patient_persona=(
            f"{_EXISTING_IDENTITY} You have a General Consultation booked. When rescheduling, "
            "say you actually want to change it to an Implant Consult instead. "
            "If the agent offers to create a note or book a new appointment for Implant Consult, accept that. "
            f"{_BEHAVIOR}"
        ),
        max_turns=17,
        expected_behaviors=[
            "Agent informs patient that service changes on existing bookings require team assistance or a new booking",
            "Agent offers to create a note or book a fresh appointment with the new service",
            "Agent verifies patient identity via phone and DOB",
        ],
    ),

    TestCase(
        id="TC_031",
        name="Multipatient (Same Type) — Same Service Any Provider",
        severity="critical",
        description="Two patients of same type booking same service, no provider preference.",
        patient_persona=(
            f"{_EXISTING_IDENTITY} You want to book General Consultations for yourself and your spouse "
            "Jane MAN (existing patient, DOB 20 May 2002, phone 5556667778). "
            "Neither has a provider preference. Same service for both. "
            f"{_BEHAVIOR}"
        ),
        max_turns=21,
        expected_behaviors=[
            "Agent books same service for both patients",
            "Agent collects phone number and date of birth for each patient separately",
            "Agent does not mix up patient details",
        ],
    ),

    TestCase(
        id="TC_032",
        name="Multipatient (Same Type) — Different Service Any Provider",
        severity="critical",
        description="Two existing patients booking different services, no provider preference.",
        patient_persona=(
            f"{_EXISTING_IDENTITY} You want a General Consultation for yourself and an Emergency Consult "
            "for your friend Mark (existing patient, DOB 12 Aug 1998, phone 4443332221). "
            "No provider preference for either. "
            f"{_BEHAVIOR}"
        ),
        max_turns=21,
        expected_behaviors=[
            "Agent attempts to book the correct service for each patient; if a patient record cannot be found, agent books for the patient whose record is found and offers to create a note or task for the patient whose record was not found",
            "Agent does not confuse services between patients",
        ],
    ),

    TestCase(
        id="TC_033",
        name="Multipatient (Same Type) — Same Service Specific Provider",
        severity="critical",
        description="Two existing patients booking same service with a specific provider. If a patient record cannot be found, agent should book for the patient whose record was found and create a task for the one whose record was not found.",
        patient_persona=(
            f"{_EXISTING_IDENTITY} You and your colleague (existing patient, DOB 03 Mar 2001, phone 7776665554) "
            "both want Cleaning Exam OPD specifically with Lucas Bennett. "
            "If the agent says your colleague's record cannot be found, accept the note creation for them. "
            f"{_BEHAVIOR}"
        ),
        max_turns=21,
        expected_behaviors=[
            "Agent collects phone number and DOB for each patient to verify identity",
            "Agent attempts to book with Lucas Bennett for the patient whose record is found",
            "Agent offers to create a task for any patient whose record cannot be verified",
        ],
    ),

    TestCase(
        id="TC_034",
        name="Multipatient (Same Type) — Different Service Different Provider",
        severity="critical",
        description="Two existing patients, different services, each wanting a different specific provider.",
        patient_persona=(
            f"{_EXISTING_IDENTITY} You want a General Consultation with Ethan Brooks. "
            "Your partner (existing, DOB 25 Dec 1999, phone 1231231234) wants Implant Consult with Lucas Bennett. "
            f"{_BEHAVIOR}"
        ),
        max_turns=21,
        expected_behaviors=[
            "Agent attempts to book the correct service and provider for each patient; if a patient record cannot be found, agent books for the patient whose record is found and offers to create a note or task for the patient whose record was not found",
            "No cross-contamination of patient details or providers",
        ],
    ),

    TestCase(
        id="TC_035",
        name="Multipatient (Different Type) — Same Service Any Provider",
        severity="critical",
        description="One new patient and one existing patient, same service, no provider preference.",
        patient_persona=(
            f"{_NEW_IDENTITY} You are a new patient wanting a New Patient Exam. "
            "Your friend Sam (existing patient, DOB 11 Nov 2000, phone 9990001112) also wants a General Consultation. "
            "Neither has a provider preference. "
            f"{_BEHAVIOR}"
        ),
        max_turns=21,
        expected_behaviors=[
            "Agent handles new patient flow for first patient",
            "Agent attempts existing patient flow (phone + DOB) for second patient; if second patient record cannot be found, agent offers to create a note or task",
            "Agent uses correct service type for each",
        ],
    ),

    TestCase(
        id="TC_036",
        name="Multipatient (Different Type) — Different Service Specific Provider",
        severity="critical",
        description="One new and one existing patient, different services, specific provider.",
        patient_persona=(
            f"{_NEW_IDENTITY} You are a new patient wanting a Dental Emergency with Clinic2 Provider. "
            "Your sister (existing, DOB 08 Aug 1997, phone 8887776665) wants Cervical Spine with Lucas Bennett. "
            "In your VERY FIRST MESSAGE, mention both your own appointment need AND your sister's appointment need. "
            "Do not wait to be asked about a second patient — introduce both in your opening message. "
            f"{_BEHAVIOR}"
        ),
        max_turns=21,
        expected_behaviors=[
            "Agent collects new patient info fully",
            "Agent verifies existing patient identity via phone and DOB; if existing patient record cannot be found, agent offers to create a note or task",
            "Agent books correct service and provider for each patient whose record is found or created",
        ],
    ),

    TestCase(
        id="TC_037",
        name="FS Single Patient Rescheduling",
        severity="critical",
        description="Fresh start — single existing patient wants to reschedule their appointment.",
        patient_persona=(
            f"{_EXISTING_IDENTITY} You want to reschedule your upcoming General Consultation to next week Tuesday. "
            "Provide identity when asked. "
            "When the agent lists your actual upcoming appointments and asks which one to reschedule, "
            "ALWAYS pick one from the specific dates and times the agent mentions. "
            "Do NOT invent or give appointment dates that are not in the agent's list. "
            "Reject the first slot offered for the new time and ask for a different time. Accept the second slot. "
            f"{_BEHAVIOR}"
        ),
        max_turns=17,
        expected_behaviors=[
            "Agent collects phone number and date of birth to verify patient identity",
            "Agent locates existing appointment",
            "Agent presents available slots for new time",
            "Agent offers alternative slot when first is rejected",
            "Agent confirms rescheduling",
        ],
    ),

    TestCase(
        id="TC_038",
        name="FS Single Patient Cancellation",
        severity="critical",
        description="Fresh start — single patient wants to cancel their appointment.",
        patient_persona=(
            f"{_EXISTING_IDENTITY} You want to cancel your upcoming appointment. "
            "Confirm when the agent asks you to verify the cancellation. "
            f"{_BEHAVIOR}"
        ),
        max_turns=15,
        expected_behaviors=[
            "Agent collects phone number and date of birth to verify patient identity",
            "Agent locates the appointment",
            "Agent asks for confirmation before cancelling",
            "Agent confirms cancellation",
        ],
    ),

    TestCase(
        id="TC_039",
        name="FS Multipatient Rescheduling / Confirmation",
        severity="critical",
        description="Fresh start — two patients want to confirm their upcoming appointments in the same conversation.",
        patient_persona=(
            f"{_EXISTING_IDENTITY} You AND your brother both want to confirm your upcoming appointments. "
            "Your brother is also an existing patient: DOB 22 February 1996, phone 2223334445. "
            "Start by saying both of you want to confirm your appointments. "
            "Make sure to mention your brother explicitly in your first or second message. "
            "Provide your own details when asked, then your brother's details when the agent asks for them. "
            f"{_BEHAVIOR}"
        ),
        max_turns=21,
        expected_behaviors=[
            "Agent handles multiple patients for confirmation",
            "Agent processes each patient sequentially",
            "Agent collects phone number and DOB for each patient",
            "Agent does not mix appointment data between patients",
        ],
    ),

    TestCase(
        id="TC_040",
        name="FS Multipatient Cancellation",
        severity="critical",
        description="Fresh start — multiple patients want to cancel their appointments.",
        patient_persona=(
            f"{_EXISTING_IDENTITY} You and your colleague (existing, DOB 14 Jul 2003, phone 6667778889) "
            "both want to cancel your upcoming appointments. "
            "In your FIRST OR SECOND message, explicitly mention that your colleague also needs to cancel their appointment. "
            "When the agent lists your upcoming appointments and asks which one to cancel, "
            "ALWAYS pick from the specific dates and times the agent actually shows you. "
            "Do NOT invent or give dates that are not in the agent's list. "
            "Confirm both cancellations. "
            f"{_BEHAVIOR}"
        ),
        max_turns=21,
        expected_behaviors=[
            "Agent cancels appointments for both patients",
            "Agent collects phone number and DOB for each patient",
            "Agent confirms each cancellation separately",
        ],
    ),

    TestCase(
        id="TC_041",
        name="Follow Up Multipatient Rescheduling / Confirmation",
        severity="warning",
        description="Continuing a prior conversation — multiple patients want to reschedule.",
        patient_persona=(
            f"{_EXISTING_IDENTITY} This is a follow-up to a previous conversation about bookings. "
            "You now want to reschedule both appointments that were booked earlier to the following week. "
            "Reject the first slot offered for each and ask for alternatives. Accept the second slot each time. "
            f"{_BEHAVIOR}"
        ),
        max_turns=19,
        conversation_thread=_FOLLOWUP_THREAD_MULTI_RESCHEDULE,
        expected_behaviors=[
            "Agent uses prior conversation context",
            "Agent attempts to handle rescheduling for multiple patients",
            "Agent offers alternative slots when first is rejected; if no slots are available, agent correctly offers to create a note or task — this is correct behaviour",
            "Agent confirms new times for patients whose rescheduling was completed, or creates a note when no slots are available",
        ],
    ),

    TestCase(
        id="TC_042",
        name="Follow Up Multipatient Cancellation",
        severity="warning",
        description="Continuing a prior conversation — multiple patients want to cancel.",
        patient_persona=(
            f"{_EXISTING_IDENTITY} This is a follow-up to a prior conversation. "
            "You now want to cancel both appointments that were previously discussed. "
            f"{_BEHAVIOR}"
        ),
        max_turns=19,
        conversation_thread=_FOLLOWUP_THREAD_MULTI_CANCEL,
        expected_behaviors=[
            "Agent uses prior context to understand who the patients are",
            "Agent confirms and cancels both appointments",
        ],
    ),

    TestCase(
        id="TC_043",
        name="Incomplete Chat — Cancellation",
        severity="warning",
        description="Conversation starts for cancellation but patient drops off mid-flow.",
        patient_persona=(
            f"{_EXISTING_IDENTITY} You start asking to cancel your appointment. "
            "After the agent asks for your phone or DOB, say 'never mind, I will call back' and end. "
            f"{_BEHAVIOR}"
        ),
        max_turns=11,
        expected_behaviors=[
            "Agent does not force the conversation to continue",
            "Agent ends gracefully when patient wants to drop off",
            "Agent does not partially cancel without confirmation",
        ],
    ),

    TestCase(
        id="TC_044",
        name="Incomplete Chat — Rescheduling",
        severity="warning",
        description="Conversation starts for rescheduling but patient abandons mid-flow.",
        patient_persona=(
            f"{_EXISTING_IDENTITY} You start asking to reschedule. After providing your phone and DOB, "
            "say 'I changed my mind, I will keep the original time' and end the conversation. "
            f"{_BEHAVIOR}"
        ),
        max_turns=11,
        expected_behaviors=[
            "Agent does not reschedule without final confirmation",
            "Agent acknowledges patient's decision to keep original appointment",
        ],
    ),

    TestCase(
        id="TC_045",
        name="Incomplete Chat — Confirmation",
        severity="warning",
        description="Patient starts confirmation flow but leaves before completing.",
        patient_persona=(
            f"{_EXISTING_IDENTITY} You start to confirm your appointment. "
            "After the agent finds your appointment, say 'hold on I need to check something, bye' and leave "
            "WITHOUT selecting or confirming any specific appointment. "
            "Do NOT say which appointment you want to confirm before leaving. "
            f"{_BEHAVIOR}"
        ),
        max_turns=11,
        expected_behaviors=[
            "Agent does not auto-confirm an appointment that the patient has not explicitly selected",
            "Agent handles abrupt end gracefully",
        ],
    ),
]
