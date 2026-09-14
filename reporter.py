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
      - Full transcript (collapsible)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from evaluator import EvaluationResult
from runner import ConversationResult
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
            label = "PATIENT" if turn.role == "patient" else "AGENT  "
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
