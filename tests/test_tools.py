import unittest
from unittest.mock import patch

from app.tools import (
    _safe_calculate,
    dispatch_tool,
    execute_confirmed_action,
)


class ToolTests(unittest.TestCase):
    def test_calculator_supports_arithmetic_without_eval(self):
        self.assertAlmostEqual(
            _safe_calculate("319 * (1 - 15 / 100)"),
            271.15,
        )
        for expression in ("__import__('os')", "2 ** 100", "1 / 0"):
            with self.subTest(expression=expression), self.assertRaises(ValueError):
                _safe_calculate(expression)

    def test_unknown_tools_and_invalid_arguments_are_not_executed(self):
        with self.assertRaises(ValueError):
            dispatch_tool("run_shell", "{}", lambda _query: [])

        outcome = dispatch_tool(
            "create_support_ticket",
            '{"subject":"x","description":"too short"}',
            lambda _query: [],
        )
        self.assertIn("Invalid tool arguments", outcome.result["error"])
        self.assertIsNone(outcome.pending_action)

    def test_write_tools_prepare_pending_actions_until_confirmation(self):
        outcome = dispatch_tool(
            "create_support_ticket",
            '{"subject":"NovaCam connection","description":"The camera cannot connect to Wi-Fi."}',
            lambda _query: [],
        )
        self.assertTrue(outcome.result["confirmation_required"])
        self.assertEqual(outcome.pending_action["tool"], "create_support_ticket")

        with patch("app.tools._append_jsonl") as append:
            result = execute_confirmed_action(outcome.pending_action)

        append.assert_called_once()
        self.assertIn("Ticket TKT-", result["message"])

    def test_email_write_is_confirmed_and_recipient_is_validated(self):
        outcome = dispatch_tool(
            "send_email",
            '{"recipient":"support@example.com","subject":"Update","body":"Please call me."}',
            lambda _query: [],
        )
        with patch("app.tools._append_jsonl") as append:
            result = execute_confirmed_action(outcome.pending_action)

        append.assert_called_once()
        self.assertIn("support@example.com", result["message"])

        invalid_email = dispatch_tool(
            "send_email",
            '{"recipient":"invalid","subject":"Update","body":"Please call me."}',
            lambda _query: [],
        )
        self.assertIn("Invalid tool arguments", invalid_email.result["error"])
        self.assertIsNone(invalid_email.pending_action)

        invalid_action = {
            "tool": "send_email",
            "arguments": {
                "recipient": "not-an-email",
                "subject": "Update",
                "body": "Please call me.",
            },
        }
        with self.assertRaises(ValueError):
            execute_confirmed_action(invalid_action)


if __name__ == "__main__":
    unittest.main()
