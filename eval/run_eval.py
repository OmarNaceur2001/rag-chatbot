"""Evaluate answer keywords and retrieved source pages against questions.jsonl."""

import json
import re
import sys
import unicodedata
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.gemini_provider import (  # noqa: E402
    GeminiAuthenticationError,
    GeminiConnectionError,
    GeminiRateLimitError,
)

from app.guardrails import InputRejected  # noqa: E402
from app.rag import generate  # noqa: E402

QUESTIONS_FILE = Path(__file__).with_name("questions.jsonl")
REFUSAL_MARKERS = (
    "i do not have this information",
    "i can't help with that request",
    "je ne dispose pas de cette information",
    "je ne peux pas aider avec cette demande",
    "لا أملك هذه المعلومات",
    "لا أستطيع المساعدة في هذا الطلب",
)


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    without_marks = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(re.findall(r"\w+", without_marks))


def _matches_keywords(answer: str, keyword_groups: list[list[str]]) -> tuple[bool, list[list[str]]]:
    normalized = _normalize(answer)
    missing = [
        group for group in keyword_groups
        if not any(_normalize(term) in normalized for term in group)
    ]
    return not missing, missing


def _matches_sources(
    actual: list[dict[str, Any]],
    expected: list[dict[str, Any]],
) -> tuple[bool, list[dict[str, Any]]]:
    actual_sources = {
        (_normalize(str(source.get("filename", ""))), source.get("page"))
        for source in actual
    }
    missing = [
        source for source in expected
        if (_normalize(source["filename"]), source.get("page")) not in actual_sources
    ]
    return not missing, missing


def _is_refusal(answer: str) -> bool:
    normalized = _normalize(answer)
    return any(_normalize(marker) in normalized for marker in REFUSAL_MARKERS)


def score_case(case: dict[str, Any], result: dict[str, Any] | None = None, rejected: bool = False) -> dict[str, Any]:
    should_refuse = case.get("should_refuse", False)
    if should_refuse:
        refused = rejected or bool(result and _is_refusal(result["answer"]))
        return {
            "passed": refused,
            "answer_passed": refused,
            "retrieval_passed": None,
            "missing_keywords": [],
            "missing_sources": [],
        }

    if rejected or result is None:
        return {
            "passed": False,
            "answer_passed": False,
            "retrieval_passed": False,
            "missing_keywords": case.get("expected_keywords", []),
            "missing_sources": case.get("expected_sources", []),
        }

    answer_passed, missing_keywords = _matches_keywords(
        result["answer"],
        case.get("expected_keywords", []),
    )
    expected_sources = case.get("expected_sources", [])
    retrieval_passed, missing_sources = _matches_sources(
        result.get("sources", []),
        expected_sources,
    )
    return {
        "passed": answer_passed and retrieval_passed,
        "answer_passed": answer_passed,
        "retrieval_passed": retrieval_passed if expected_sources else None,
        "missing_keywords": missing_keywords,
        "missing_sources": missing_sources,
    }


def load_cases(path: Path = QUESTIONS_FILE) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def run_evaluation() -> int:
    cases = load_cases()
    passed = 0
    retrieval_hits = 0
    retrieval_cases = 0
    failures = []
    latencies = []
    api_errors = 0
    started = time.perf_counter()

    for case in cases:
        try:
            case_started = time.perf_counter()
            result = generate(case["question"])
            latencies.append(time.perf_counter() - case_started)
            scored = score_case(case, result=result)
        except InputRejected:
            scored = score_case(case, rejected=True)
            result = None
        except (GeminiAuthenticationError, GeminiConnectionError, GeminiRateLimitError) as error:
            api_errors += 1
            print(
                f"Evaluation stopped at {case['id']}: {type(error).__name__}. "
                "Check the GEMINI_API_KEY/model/service and retry."
            )
            return 2

        passed += int(scored["passed"])
        if scored["retrieval_passed"] is not None:
            retrieval_cases += 1
            retrieval_hits += int(scored["retrieval_passed"])
        if not scored["passed"]:
            failures.append((case["id"], scored, result))

    print(f"Evaluation score: {passed}/{len(cases)} = {passed / len(cases):.0%}")
    if latencies:
        print(f"Average latency: {sum(latencies) / len(latencies):.2f}s")
        print(f"Total evaluation latency: {time.perf_counter() - started:.2f}s")
    print(f"API errors: {api_errors}; failure rate: {(len(cases) - passed) / len(cases):.1%}")
    if retrieval_cases:
        print(f"Retrieval hit rate: {retrieval_hits}/{retrieval_cases} = {retrieval_hits / retrieval_cases:.0%}")
    for case_id, scored, result in failures:
        answer = result["answer"][:180] if result else "(request was refused)"
        print(
            f"FAIL {case_id}: missing keywords={scored['missing_keywords']}; "
            f"missing sources={scored['missing_sources']}; answer={answer!r}"
        )
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(run_evaluation())
