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
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from config import OPENAI_API_KEY
from evaluator import Evaluator
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
    return parser.parse_args()


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
    passed = sum(1 for r in results if r.evaluation.verdict == "PASS")
    warned = sum(1 for r in results if r.evaluation.verdict == "WARN")
    failed = sum(1 for r in results if r.evaluation.verdict == "FAIL")
    total  = len(results)

    logger.info("=" * 60)
    logger.info("RESULTS:  ✅ %d PASS  |  ⚠️  %d WARN  |  ❌ %d FAIL  (total: %d)", passed, warned, failed, total)
    logger.info("REPORT:   %s", report_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
