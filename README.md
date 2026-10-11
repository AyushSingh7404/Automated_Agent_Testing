# Agent Testing Framework

Automated end-to-end testing for the dental clinic AI chat agent.

## Structure

```
Agent_testing/
├── run_tests.py          # Main entry point
├── config.py             # API URLs, credentials, test data
├── runner.py             # Drives conversations via the forward-to-agent API
├── evaluator.py          # LLM judge — evaluates transcripts
├── reporter.py           # Generates Markdown reports
├── paste_mode.py         # Evaluate a pasted/fetched transcript directly, skipping the live runner
├── pyproject.toml
├── test_cases/
│   └── cases.py          # All 58 test case definitions
└── reports/              # Generated Markdown reports (auto-created)
```

## Setup

```bash
pip install requests langchain-openai langchain-core
# or
uv sync
```

Set environment variables:
```bash
export OPENAI_API_KEY="sk-..."
export API_ACCESS_TOKEN="2f81fe55"   # optional — default is already set
```

## Usage

```bash
# Run all test cases
python run_tests.py

# Run specific test cases
python run_tests.py --tests TC_001 TC_002 TC_015

# Run only critical severity tests
python run_tests.py --severity critical

# Run with a custom report label
python run_tests.py --label "sprint_3_regression"

# List all test cases without running
python run_tests.py --dry-run

# Paste mode — evaluate an already-completed transcript against a test case's
# expected_behaviors checklist, skipping the live GPT-simulated conversation.
# Accepts a JSON array of {role, content} (exactly what
# GET /engage/chat-transcript/{chat_id} returns, wrapper included) or
# PATIENT:/AGENT: labeled plain text.
python run_tests.py --paste transcript.json --test-id TC_031
python run_tests.py --paste - --test-id TC_001   # read from stdin
```

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

## Report Format

Each report is saved to `reports/test_report_<timestamp>.md` and contains:
- Overall pass rate and average score
- Summary table with verdict, score, severity, and chat ID per test
- Quick-view list of all failures
- Detailed section per test case with findings, failed behaviours, and full transcript
