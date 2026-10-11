"""
paste_mode.py
──────────────────────────────────────────
Evaluate an already-completed conversation transcript against a test case's
expected_behaviors checklist, WITHOUT driving a live GPT-simulated conversation
through runner.py.

Use case: you already have a real transcript (e.g. pulled via
GET /engage/chat-transcript/{chat_id}, or copy-pasted from Retell/support),
and you just want the same GPT-4o judge + Markdown report that a normal test
run produces, run directly against that transcript.

Accepted input formats (auto-detected):
  1. JSON array of turn objects, e.g. exactly what /engage/chat-transcript
     returns: [{"role": "user"|"agent"|"patient"|"ai"|"practice", "content": "..."}, ...]
  2. Plain text, one turn per line, prefixed with "PATIENT:" / "AGENT:"
     (case-insensitive) — the same format the reports/runner print in.
"""
from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Optional

from evaluator import Evaluator
from reporter import TestResult, generate_report
from retell_transcript import retell_items_to_turns, unwrap_transcript_payload
from runner import ConversationResult, TurnRecord
from test_cases.cases import ALL_TEST_CASES, TestCase

logger = logging.getLogger(__name__)

_LINE_PATTERN = re.compile(r"^\s*(PATIENT|AGENT)\s*:\s*(.*)$", re.IGNORECASE)


def find_test_case(test_id: str) -> Optional[TestCase]:
    for tc in ALL_TEST_CASES:
        if tc.id == test_id:
            return tc
    return None


def parse_transcript(raw: str) -> list[TurnRecord]:
    """Parse pasted transcript text into TurnRecords. Tries JSON first, falls
    back to PATIENT:/AGENT: line format. Raises ValueError if neither parses
    into at least one turn.
    """
    raw = raw.strip()
    if not raw:
        raise ValueError("Pasted transcript is empty.")

    turns = _try_parse_json(raw)
    if turns is None:
        turns = _parse_labeled_lines(raw)

    if not turns:
        raise ValueError(
            "Could not parse any turns from the pasted transcript. "
            "Expected either a JSON array of {role, content} objects, or lines "
            "prefixed with 'PATIENT:' / 'AGENT:'."
        )
    return turns


def _try_parse_json(raw: str) -> Optional[list[TurnRecord]]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None

    items = unwrap_transcript_payload(data)
    if items is None:
        return None

    turns = [TurnRecord(role=t["role"], content=t["content"]) for t in retell_items_to_turns(items)]
    return turns or None


def _parse_labeled_lines(raw: str) -> list[TurnRecord]:
    turns: list[TurnRecord] = []
    current_role: Optional[str] = None
    current_lines: list[str] = []

    def _flush() -> None:
        if current_role and current_lines:
            turns.append(TurnRecord(role=current_role, content=" ".join(current_lines).strip()))

    for line in raw.splitlines():
        match = _LINE_PATTERN.match(line)
        if match:
            _flush()
            label, content = match.groups()
            current_role = "patient" if label.upper() == "PATIENT" else "agent"
            current_lines = [content.strip()] if content.strip() else []
        elif current_role:
            current_lines.append(line.strip())
    _flush()

    return turns


def run_paste_mode(
    test_id: str,
    transcript_source: str,
    chat_id: Optional[str] = None,
    run_label: Optional[str] = None,
) -> Path:
    """
    Evaluate a pasted transcript against test_id's expected_behaviors and
    generate a report exactly like a normal run would.

    transcript_source: "-" to read from stdin, otherwise a file path.
    """
    test_case = find_test_case(test_id)
    if test_case is None:
        raise ValueError(f"Unknown test case id: {test_id}")

    if transcript_source == "-":
        logger.info("Reading pasted transcript from stdin (end with Ctrl+D / Ctrl+Z)...")
        raw = sys.stdin.read()
    else:
        raw = Path(transcript_source).read_text(encoding="utf-8")

    turns = parse_transcript(raw)

    conv_result = ConversationResult(
        test_case_id=test_case.id,
        chat_id=chat_id,
        transcript=turns,
        ended_gracefully=True,
        infra_error=False,
    )

    logger.info("Parsed %d turns. Evaluating %s against pasted transcript...", len(turns), test_case.id)

    evaluator = Evaluator()
    eval_result = evaluator.evaluate(test_case, conv_result)

    logger.info(
        "Result: %s | Score: %d/100",
        eval_result.verdict,
        eval_result.score,
    )

    result = TestResult(test_case=test_case, conversation=conv_result, evaluation=eval_result)
    report_path = generate_report([result], run_label=run_label or f"paste_{test_case.id}")
    logger.info("Report written to %s", report_path)
    return report_path
