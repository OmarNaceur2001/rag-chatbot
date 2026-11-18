import unittest
from collections import Counter

from eval.run_eval import load_cases, score_case


class EvaluationTests(unittest.TestCase):
    def test_benchmark_has_the_requested_30_case_mix(self):
        cases = load_cases()
        counts = Counter(case["category"] for case in cases)

        self.assertEqual(len(cases), 30)
        self.assertEqual(
            counts,
            {
                "direct": 10,
                "numbers_tables": 6,
                "cross_document": 4,
                "conditional": 3,
                "out_of_scope": 3,
                "multilingual": 2,
                "injection": 2,
            },
        )

    def test_answer_and_source_page_are_both_required_to_pass(self):
        case = {
            "expected_keywords": [["30 days"]],
            "expected_sources": [{"filename": "returns.pdf", "page": 2}],
        }
        result = {
            "answer": "You have 30 days.",
            "sources": [{"filename": "returns.pdf", "page": 1}],
        }

        score = score_case(case, result=result)

        self.assertFalse(score["passed"])
        self.assertTrue(score["answer_passed"])
        self.assertFalse(score["retrieval_passed"])

    def test_refusal_cases_accept_guardrail_rejections(self):
        score = score_case({"should_refuse": True}, rejected=True)

        self.assertTrue(score["passed"])
        self.assertIsNone(score["retrieval_passed"])


if __name__ == "__main__":
    unittest.main()
