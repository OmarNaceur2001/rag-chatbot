import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.rag import build_prompt, generate, retrieve, rewrite_search_query


class RagTests(unittest.TestCase):
    def test_retrieval_uses_ingestion_metadata_and_optional_page(self):
        expected = [{
            "text": "Relevant text",
            "source": "https://example.com/",
            "title": "Example",
            "page": None,
            "url": "https://example.com/",
            "distance": 0.1,
        }]
        with patch("app.rag.hybrid_search", return_value=expected):
            chunks = retrieve("question")

        self.assertEqual(chunks[0]["source"], "https://example.com/")
        self.assertIsNone(chunks[0]["page"])

    def test_prompt_keeps_followup_history_and_source_context(self):
        messages = build_prompt(
            "And the second one?",
            [{"source": "guide.txt", "title": "Guide", "page": None, "text": "Two options."}],
            [{"role": "user", "content": "What are the options?"}],
        )

        self.assertEqual(messages[1]["content"], "What are the options?")
        self.assertEqual(messages[2]["content"], "And the second one?")
        self.assertIn("Guide", messages[0]["content"])
        self.assertIn("<untrusted_document", messages[0]["content"])

    def test_followup_rewrite_uses_history_and_returns_standalone_query(self):
        provider = Mock()
        provider.generate.return_value = {
            "text": "NovaCloud plan options and their included users",
            "function_calls": [],
        }
        history = [
            {"role": "user", "content": "What are the available plans?"},
            {"role": "assistant", "content": "Basic and Pro."},
        ]
        rewritten = rewrite_search_query(
            "And the second one?",
            history,
            provider,
        )

        self.assertEqual(rewritten, "NovaCloud plan options and their included users")
        self.assertEqual(
            provider.generate.call_args.args[0][1]["content"],
            "What are the available plans?",
        )

    def test_generate_runs_calculator_tool_and_returns_safe_answer(self):
        provider = Mock()
        provider.generate.side_effect = [
            {"text": "calculate discount", "function_calls": []},
            {
                "text": "",
                "function_calls": [{
                    "id": "tool-call-1",
                    "name": "calculate",
                    "arguments": {"expression": "319 * (1 - 15 / 100)"},
                }],
            },
            {"text": "The discounted price is $271.15.", "function_calls": []},
        ]
        chunks = [{
            "text": "Home Security Bundle",
            "source": "pricing.pdf",
            "title": "Pricing",
            "page": 1,
            "url": None,
        }]

        with patch("app.rag.hybrid_search", return_value=chunks):
            result = generate("What is $319 minus 15%?", provider=provider)

        self.assertIn("$271.15", result["answer"])
        self.assertEqual(result["sources"][0]["filename"], "pricing.pdf")
        self.assertIsNone(result["pending_action"])

    def test_generate_stages_write_tools_without_executing_them(self):
        provider = Mock()
        provider.generate.side_effect = [
            {"text": "NovaCam connection support", "function_calls": []},
            {
                "text": "",
                "function_calls": [{
                    "id": "tool-call-2",
                    "name": "create_support_ticket",
                    "arguments": {
                        "subject": "NovaCam connection",
                        "description": "The camera cannot connect.",
                    },
                }],
            },
        ]

        with patch("app.rag.hybrid_search", return_value=[]), patch(
            "app.tools.execute_confirmed_action"
        ) as execute:
            result = generate("Open a support ticket about my NovaCam.", provider=provider)

        self.assertIsNotNone(result["pending_action"])
        self.assertEqual(result["pending_action"]["tool"], "create_support_ticket")
        execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
