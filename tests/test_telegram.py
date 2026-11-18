import unittest

from app.telegram_bot import split_telegram_message


class TelegramTests(unittest.TestCase):
    def test_long_messages_are_split_without_losing_content(self):
        message = "Line one with spaces\n" + ("x" * 19) + "\nlast line"

        parts = split_telegram_message(message, limit=12)

        self.assertGreater(len(parts), 1)
        self.assertTrue(all(len(part.encode("utf-16-le")) // 2 <= 12 for part in parts))
        self.assertEqual("".join(parts), message)

    def test_supplementary_unicode_stays_within_telegram_character_limit(self):
        message = "\U0001f680" * 3000
        parts = split_telegram_message(message, limit=120)

        self.assertTrue(all(len(part.encode("utf-16-le")) // 2 <= 120 for part in parts))
        self.assertEqual("".join(parts), message)

    def test_empty_messages_and_invalid_limits(self):
        self.assertEqual(split_telegram_message(""), [""])
        with self.assertRaises(ValueError):
            split_telegram_message("text", limit=0)


if __name__ == "__main__":
    unittest.main()
