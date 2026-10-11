"""
reporter.py
──────────────────────────────────────────
Generates a Markdown report from all test results.

Report structure:
  - Header with run metadata (timestamp, total counts, pass rate)
  - Summary table (one row per test case)
  - Detailed section per test case with:
      - Chat ID
      - Severity
      - Verdict + Score
      - Summary
      - Findings
      - Failed behaviours
      - Tool Calls (not collapsed — quick-scan summary of every tool call/result)
      - Full transcript (collapsible)
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from evaluator import EvaluationResult
from runner import ConversationResult, TurnRecord
from test_cases.cases import TestCase

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).parent / "reports"


@dataclass
class TestResult:
    test_case: TestCase
    conversation: ConversationResult
    evaluation: EvaluationResult


_VERDICT_EMOJI = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌", "API_ERROR": "🔌"}
_SEVERITY_EMOJI = {"critical": "🔴", "warning": "🟡"}

_CALL_PATTERN = re.compile(r"^CALLED (.+?)\((.*)\)$", re.DOTALL)
_RESULT_PATTERN = re.compile(r"^RESULT for (.+?) \| successful=(.+?) \| (.*)$", re.DOTALL)


def _extract_tool_calls(transcript: list[TurnRecord]) -> list[dict]:
    """
    Pair up the flattened "CALLED ..." / "RESULT for ..." TOOL turns (in
    transcript order) back into structured {tool_name, arguments, successful,
    result} records for the quick-scan Tool Calls section. Each RESULT is
    matched to the earliest still-unmatched CALL for the same tool name
    (tool calls execute in order, so this holds even with several calls to
    the same tool). A call with no result yet, or a result with no call
    recorded, is still shown rather than dropped.
    """
    pending: dict[str, list[dict]] = {}
    ordered: list[dict] = []

    for turn in transcript:
        if turn.role != "tool":
            continue

        call_match = _CALL_PATTERN.match(turn.content)
        if call_match:
            tool_name, arguments = call_match.groups()
            record = {"tool_name": tool_name, "arguments": arguments, "successful": None, "result": None}
            pending.setdefault(tool_name, []).append(record)
            ordered.append(record)
            continue

        result_match = _RESULT_PATTERN.match(turn.content)
        if result_match:
            tool_name, successful, result_content = result_match.groups()
            queue = pending.get(tool_name)
            if queue:
                record = queue.pop(0)
                record["successful"] = successful
                record["result"] = result_content
            else:
                # Result with no matching call recorded earlier in the transcript.
                ordered.append({
                    "tool_name": tool_name,
                    "arguments": None,
                    "successful": successful,
                    "result": result_content,
                })

    return ordered


def generate_report(results: list[TestResult], run_label: Optional[str] = None) -> Path:
    """Write the full markdown report and return the file path."""
    REPORTS_DIR.mkdir(exist_ok=True)

    timestamp = datetime.now()
    ts_str = timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    label = run_label or ts_str
    filename = REPORTS_DIR / f"test_report_{ts_str}.md"

    md = _build_report(results, timestamp, label)
    filename.write_text(md, encoding="utf-8")
    logger.info("Report written to %s", filename)
    return filename


# ─────────────────────────────────────────────────────────────────────────────
# Report builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_report(results: list[TestResult], timestamp: datetime, label: str) -> str:
    total      = len(results)
    passed     = sum(1 for r in results if r.evaluation.verdict == "PASS")
    warned     = sum(1 for r in results if r.evaluation.verdict == "WARN")
    failed     = sum(1 for r in results if r.evaluation.verdict == "FAIL")
    api_errors = sum(1 for r in results if r.evaluation.verdict == "API_ERROR")
    pass_rate  = round((passed / total) * 100, 1) if total else 0
    avg_score  = round(sum(r.evaluation.score for r in results) / total, 1) if total else 0

    lines: list[str] = []

    # ── Header ────────────────────────────────────────────────────────────────
    lines += [
        f"# 🧪 Agent Test Report",
        f"",
        f"**Run:** `{label}`  ",
        f"**Date:** {timestamp.strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Total Tests:** {total}  ",
        f"**Pass Rate:** {pass_rate}%  ",
        f"**Average Score:** {avg_score}/100  ",
        f"",
        "---",
        "",
    ]

    # ── Summary counts ────────────────────────────────────────────────────────
    lines += [
        "## 📊 Summary",
        "",
        f"| Verdict | Count |",
        f"|---------|-------|",
        f"| ✅ PASS | {passed} |",
        f"| ⚠️ WARN | {warned} |",
        f"| ❌ FAIL | {failed} |",
        f"| 🔌 API ERROR | {api_errors} |",
        f"| **Total** | **{total}** |",
        "",
        "---",
        "",
    ]

    # ── Summary table ─────────────────────────────────────────────────────────
    lines += [
        "## 📋 Results Table",
        "",
        "| # | Test Case | Severity | Verdict | Score | Chat ID |",
        "|---|-----------|----------|---------|-------|---------|",
    ]
    for r in results:
        tc   = r.test_case
        ev   = r.evaluation
        conv = r.conversation
        v_emoji = _VERDICT_EMOJI.get(ev.verdict, "❓")
        s_emoji = _SEVERITY_EMOJI.get(tc.severity, "⚪")
        chat_id = f"`{conv.chat_id}`" if conv.chat_id else "*n/a*"
        lines.append(
            f"| {tc.id} | {tc.name} | {s_emoji} {tc.severity} | {v_emoji} {ev.verdict} | {ev.score}/100 | {chat_id} |"
        )

    lines += ["", "---", ""]

    # ── Failures quick-list ────────────────────────────────────────────────────
    failures = [r for r in results if r.evaluation.verdict == "FAIL"]
    if failures:
        lines += [
            "## 🚨 Failed Tests (Quick View)",
            "",
        ]
        for r in failures:
            lines += [
                f"- **{r.test_case.id}** — {r.test_case.name}",
                f"  - Score: {r.evaluation.score}/100",
                f"  - Summary: {r.evaluation.summary}",
            ]
        lines += ["", "---", ""]

    # ── API errors quick-list ───────────────────────────────────────────────────
    api_error_tests = [r for r in results if r.evaluation.verdict == "API_ERROR"]
    if api_error_tests:
        lines += [
            "## 🔌 API Errors (Quick View)",
            "",
        ]
        for r in api_error_tests:
            lines += [
                f"- **{r.test_case.id}** — {r.test_case.name}",
                f"  - Error: {r.conversation.error or 'unknown'}",
            ]
        lines += ["", "---", ""]

    # ── Detailed sections ─────────────────────────────────────────────────────
    lines += ["## 🔍 Detailed Results", ""]

    for r in results:
        tc   = r.test_case
        ev   = r.evaluation
        conv = r.conversation
        v_emoji = _VERDICT_EMOJI.get(ev.verdict, "❓")
        s_emoji = _SEVERITY_EMOJI.get(tc.severity, "⚪")

        lines += [
            f"### {tc.id} — {tc.name}",
            "",
            f"| Field | Value |",
            f"|-------|-------|",
            f"| **Verdict** | {v_emoji} {ev.verdict} |",
            f"| **Score** | {ev.score}/100 |",
            f"| **Severity** | {s_emoji} {tc.severity} |",
            f"| **Chat ID** | `{conv.chat_id or 'n/a'}` |",
            f"| **Ended Gracefully** | {'Yes' if conv.ended_gracefully else 'No'} |",
            f"| **Turns** | {len([t for t in conv.transcript if t.role == 'patient'])} patient turns |",
            f"| **Tool Calls Visible** | {'Yes' if any(t.role == 'tool' for t in conv.transcript) else 'No'} |",
            "",
        ]

        if conv.error:
            lines += [f"> ⚠️ **API Error:** {conv.error}", ""]

        # Summary
        lines += [
            f"**Summary:**  ",
            f"{ev.summary}",
            "",
        ]

        # Findings
        if ev.findings:
            lines += ["**Findings:**", ""]
            for f_item in ev.findings:
                lines.append(f"- {f_item}")
            lines.append("")

        # Failed behaviours
        if ev.failed_behaviors:
            lines += ["**Failed Behaviours:**", ""]
            for fb in ev.failed_behaviors:
                lines.append(f"- ❌ {fb}")
            lines.append("")

        # Expected behaviours checklist
        lines += ["**Expected Behaviours Checklist:**", ""]
        for eb in tc.expected_behaviors:
            passed_eb = eb not in ev.failed_behaviors
            mark = "✅" if passed_eb else "❌"
            lines.append(f"- {mark} {eb}")
        lines.append("")

        # Tool Calls (not collapsed — quick-scan summary, in the order they happened)
        tool_calls = _extract_tool_calls(conv.transcript)
        lines += ["**🔧 Tool Calls:**", ""]
        if tool_calls:
            for call in tool_calls:
                if call["successful"] is None:
                    status = "⏳ no result recorded"
                elif call["successful"].strip().lower() == "true":
                    status = "✅ successful"
                else:
                    status = "❌ failed"
                lines.append(f"- `{call['tool_name']}({call['arguments'] or '?'})` — {status}")
                if call["result"] is not None:
                    lines.append(f"  - Result: {call['result']}")
                elif call["arguments"] is None:
                    lines.append("  - (no matching call recorded)")
        else:
            lines.append("*No tool calls visible for this test — enrichment may have failed for this run.*")
        lines.append("")

        # Conversation thread (if any)
        if tc.conversation_thread:
            import json as _json
            lines += [
                "<details>",
                "<summary>📋 Prior Conversation Thread (sent as context)</summary>",
                "",
                "```json",
                _json.dumps(tc.conversation_thread, indent=2),
                "```",
                "",
                "</details>",
                "",
            ]

        # Full transcript (collapsible)
        lines += [
            "<details>",
            f"<summary>💬 Full Transcript ({len(conv.transcript)} messages)</summary>",
            "",
            "```",
        ]
        for turn in conv.transcript:
            if turn.role == "patient":
                label = "PATIENT"
            elif turn.role == "tool":
                label = "TOOL   "
            else:
                label = "AGENT  "
            lines.append(f"{label}: {turn.content}")
        lines += [
            "```",
            "",
            "</details>",
            "",
            "---",
            "",
        ]

    return "\n".join(lines)
