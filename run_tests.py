"""
run_tests.py
──────────────────────────────────────────
Main entry point for the Agent Testing framework.

Usage:
    # Run all test cases
    python run_tests.py

    # Run specific test cases by ID
    python run_tests.py --tests TC_001 TC_002 TC_015

    # Run only critical severity tests
    python run_tests.py --severity critical

    # Run a single test and print report path
    python run_tests.py --tests TC_001 --label "smoke_test"

    # Dry run — list test cases without running
    python run_tests.py --dry-run

    # Paste mode — evaluate an already-completed transcript against a test
    # case's expected_behaviors checklist, skipping the live GPT-simulated
    # conversation entirely. Accepts a JSON array of {role, content} (exactly
    # what GET /engage/chat-transcript/{chat_id} returns) or PATIENT:/AGENT:
    # labeled plain text.
    python run_tests.py --paste transcript.json --test-id TC_001
    python run_tests.py --paste - --test-id TC_031   # read from stdin
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from config import OPENAI_API_KEY
from evaluator import Evaluator
from paste_mode import run_paste_mode
from reporter import TestResult, generate_report
from runner import ConversationRunner
from test_cases.cases import ALL_TEST_CASES, TestCase

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Delay between test cases to avoid hammering the API
INTER_TEST_DELAY = 2.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agent automated test runner")
    parser.add_argument(
        "--tests",
        nargs="+",
        metavar="ID",
        help="Run only these test case IDs (e.g. TC_001 TC_002)",
    )
    parser.add_argument(
        "--severity",
        choices=["critical", "warning"],
        help="Run only test cases of this severity",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Optional label for the report filename",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List all test cases that would run without executing them",
    )
    parser.add_argument(
        "--paste",
        metavar="FILE",
        help="Path to a pasted transcript file to evaluate directly (skips the live runner). "
             "Pass '-' to read the transcript from stdin. Requires --test-id.",
    )
    parser.add_argument(
        "--test-id",
        metavar="ID",
        help="Test case ID whose expected_behaviors checklist the pasted transcript is evaluated against. "
             "Required when using --paste.",
    )
    parser.add_argument(
        "--chat-id",
        metavar="CHAT_ID",
        default=None,
        help="Optional chat_id to record in the report for a pasted transcript.",
    )
    args = parser.parse_args()

    if args.paste and not args.test_id:
        parser.error("--paste requires --test-id")
    if args.test_id and not args.paste:
        parser.error("--test-id has no effect without --paste")

    return args


def _select_cases(args: argparse.Namespace) -> list[TestCase]:
    cases = ALL_TEST_CASES

    if args.tests:
        requested = set(args.tests)
        cases = [c for c in cases if c.id in requested]
        found = {c.id for c in cases}
        missing = requested - found
        if missing:
            logger.warning("Unknown test IDs (will be skipped): %s", ", ".join(sorted(missing)))

    if args.severity:
        cases = [c for c in cases if c.severity == args.severity]

    return cases


def main() -> None:
    args = _parse_args()

    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY is not set. Export it before running tests.")
        sys.exit(1)

    # ── Paste mode — evaluate a pre-existing transcript, skip the live runner ──
    if args.paste:
        try:
            run_paste_mode(
                test_id=args.test_id,
                transcript_source=args.paste,
                chat_id=args.chat_id,
                run_label=args.label,
            )
        except (ValueError, FileNotFoundError) as exc:
            logger.error("Paste mode failed: %s", exc)
            sys.exit(1)
        return

    selected = _select_cases(args)

    if not selected:
        logger.error("No test cases matched the given filters.")
        sys.exit(1)

    # ── Dry run ───────────────────────────────────────────────────────────────
    if args.dry_run:
        print(f"\n{'─'*60}")
        print(f"  DRY RUN — {len(selected)} test case(s) would run:")
        print(f"{'─'*60}")
        for tc in selected:
            sev = f"[{tc.severity.upper()}]".ljust(12)
            print(f"  {tc.id}  {sev}  {tc.name}")
        print(f"{'─'*60}\n")
        return

    # ── Run ───────────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Starting agent test run — %d test case(s)", len(selected))
    logger.info("=" * 60)

    runner    = ConversationRunner()
    evaluator = Evaluator()
    results: list[TestResult] = []

    for i, tc in enumerate(selected, start=1):
        logger.info("")
        logger.info("─" * 60)
        logger.info("[%d/%d] %s — %s  [%s]", i, len(selected), tc.id, tc.name, tc.severity.upper())
        logger.info("─" * 60)

        # Run conversation
        conv_result = runner.run(tc)

        # Evaluate
        eval_result = evaluator.evaluate(tc, conv_result)

        logger.info(
            "  Result: %s | Score: %d/100 | Chat ID: %s",
            eval_result.verdict,
            eval_result.score,
            conv_result.chat_id or "n/a",
        )

        results.append(TestResult(
            test_case=tc,
            conversation=conv_result,
            evaluation=eval_result,
        ))

        if i < len(selected):
            time.sleep(INTER_TEST_DELAY)

    # ── Report ────────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("All tests complete. Generating report...")

    report_path = generate_report(results, run_label=args.label)

    # ── Final summary ─────────────────────────────────────────────────────────
    passed     = sum(1 for r in results if r.evaluation.verdict == "PASS")
    warned     = sum(1 for r in results if r.evaluation.verdict == "WARN")
    failed     = sum(1 for r in results if r.evaluation.verdict == "FAIL")
    api_errors = sum(1 for r in results if r.evaluation.verdict == "API_ERROR")
    total      = len(results)

    logger.info("=" * 60)
    logger.info("RESULTS:  ✅ %d PASS  |  ⚠️  %d WARN  |  ❌ %d FAIL  |  🔌 %d API ERROR  (total: %d)", passed, warned, failed, api_errors, total)
    logger.info("REPORT:   %s", report_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
