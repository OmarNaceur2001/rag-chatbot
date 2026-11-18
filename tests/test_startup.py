import unittest
from unittest.mock import patch

from app.startup import _build_index, ensure_index


class StartupTests(unittest.TestCase):
    def test_existing_nonempty_index_is_reused(self):
        class ExistingCollection:
            def count(self):
                return 3

        with patch("app.startup.get_collection", return_value=ExistingCollection()), patch(
            "app.startup.ingest"
        ) as ingest:
            ensure_index()

        ingest.assert_not_called()

    def test_invalid_key_error_is_explained_during_index_bootstrap(self):
        class TestAuthenticationError(Exception):
            pass

        with patch("app.startup.GeminiAuthenticationError", TestAuthenticationError), patch(
            "app.startup.ingest",
            side_effect=TestAuthenticationError("rejected"),
        ):
            with self.assertRaisesRegex(RuntimeError, "Gemini rejected GEMINI_API_KEY"):
                _build_index()


if __name__ == "__main__":
    unittest.main()
