import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.responses import Response

from app.main import (
    LOG_FILE,
    ChatRequest,
    _conversations,
    _log_chat_completed,
    _remember_turn,
    _require_api_key,
    app,
    chat,
)


class ChatBehaviorTests(unittest.TestCase):
    def setUp(self):
        _conversations.clear()

    def test_missing_api_key_is_rejected_with_401(self):
        with patch("app.main.has_gemini_api_key", return_value=False):
            with self.assertRaises(HTTPException) as raised:
                _require_api_key()

        self.assertEqual(raised.exception.status_code, 401)

    def test_chat_route_rejects_missing_api_key_before_generation(self):
        request = SimpleNamespace(
            cookies={},
            url=SimpleNamespace(scheme="http"),
            client=None,
        )
        with patch("app.main.has_gemini_api_key", return_value=False):
            route_call = chat(ChatRequest(question="hello"), request, Response())
            with self.assertRaises(HTTPException) as raised:
                route_call.send(None)

        self.assertEqual(raised.exception.status_code, 401)

    def test_history_is_limited_and_isolated_by_session_id(self):
        history = []
        for index in range(4):
            _remember_turn("session-one", history, f"question {index}", f"answer {index}")
            history = _conversations["session-one"]

        self.assertEqual(len(history), 5)
        self.assertEqual(history[-1], {"role": "assistant", "content": "answer 3"})
        self.assertNotIn("session-two", _conversations)

        _remember_turn("session-two", [], "new session", "new answer")
        self.assertEqual(len(_conversations["session-two"]), 2)

    def test_application_log_records_request_metadata(self):
        _log_chat_completed("test-request", 0)

        log = LOG_FILE.read_text(encoding="utf-8")
        self.assertIn("chat_completed request_id=test-request", log)

    def test_api_schema_exposes_advanced_routes(self):
        paths = app.openapi()["paths"]

        self.assertIn("/chat/confirm", paths)
        self.assertIn("/telegram/webhook", paths)
        self.assertIn("/health", paths)


if __name__ == "__main__":
    unittest.main()
