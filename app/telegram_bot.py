"""Telegram handlers shared by webhook and local long-polling modes."""

import logging
import os
from typing import Any

from starlette.concurrency import run_in_threadpool
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from app.common import has_gemini_api_key
from app.gemini_provider import (
    GeminiAuthenticationError,
    GeminiConnectionError,
    GeminiRateLimitError,
)
from app.guardrails import (
    InputRejected,
    RateLimitExceeded,
    UnsafeOutput,
    check_rate_limit,
)
from app.rag import generate
from app.retrieval import RerankError
from app.tools import execute_confirmed_action

logger = logging.getLogger("app.telegram")
MAX_HISTORY_MESSAGES = 5
TELEGRAM_MESSAGE_LIMIT = 4096
_conversations: dict[int, list[dict[str, str]]] = {}
_pending_actions: dict[int, dict[str, Any]] = {}


def split_telegram_message(text: str, limit: int = TELEGRAM_MESSAGE_LIMIT) -> list[str]:
    if limit < 1:
        raise ValueError("Message limit must be positive.")
    if not text:
        return [""]
    parts = []
    remaining = text
    while _telegram_length(remaining) > limit:
        boundary = _max_prefix_by_utf16(remaining, limit)
        if boundary == 0:
            raise ValueError("Message limit is too small for a Unicode character.")
        split_at = remaining.rfind("\n", 0, boundary)
        if split_at <= 0:
            split_at = remaining.rfind(" ", 0, boundary)
        if split_at <= 0:
            parts.append(remaining[:boundary])
            remaining = remaining[boundary:]
        else:
            parts.append(remaining[:split_at + 1])
            remaining = remaining[split_at + 1:]
    if remaining:
        parts.append(remaining)
    return parts


def _telegram_length(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def _max_prefix_by_utf16(text: str, limit: int) -> int:
    length = 0
    for index, character in enumerate(text):
        length += 2 if ord(character) > 0xFFFF else 1
        if length > limit:
            return index
    return len(text)


async def _reply(message: Any, text: str) -> None:
    for part in split_telegram_message(text):
        await message.reply_text(part)


def _remember_turn(chat_id: int, question: str, answer: str) -> None:
    updated = _conversations.get(chat_id, []) + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    _conversations[chat_id] = updated[-MAX_HISTORY_MESSAGES:]


async def start_command(update: Update, _context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_message:
        await _reply(
            update.effective_message,
            "Welcome to TechNova Support. Ask a question about our products or policies.",
        )


async def reset_command(update: Update, _context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    if chat:
        _conversations.pop(chat.id, None)
        _pending_actions.pop(chat.id, None)
    if update.effective_message:
        await _reply(update.effective_message, "Conversation history and pending actions were cleared.")


async def confirm_command(update: Update, _context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    message = update.effective_message
    if not chat or not message:
        return
    pending = _pending_actions.get(chat.id)
    if not pending:
        await _reply(message, "There is no action waiting for confirmation.")
        return
    try:
        result = await run_in_threadpool(execute_confirmed_action, pending)
    except (OSError, ValueError) as error:
        logger.warning("telegram_action_failed tool=%s reason=%s", pending["tool"], type(error).__name__)
        await _reply(message, "The confirmed action could not be completed. You can try /confirm again.")
        return

    del _pending_actions[chat.id]
    answer = result["message"]
    _remember_turn(chat.id, "User confirmed the proposed action.", answer)
    await _reply(message, answer)


async def cancel_command(update: Update, _context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    message = update.effective_message
    if not chat or not message:
        return
    pending = _pending_actions.pop(chat.id, None)
    if pending:
        answer = "Action cancelled. No changes were made."
        _remember_turn(chat.id, "User cancelled the proposed action.", answer)
    else:
        answer = "There is no action waiting for cancellation."
    await _reply(message, answer)


async def handle_text(update: Update, _context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    message = update.effective_message
    if not chat or not message or not message.text:
        return
    if chat.id in _pending_actions:
        await _reply(message, "Please use /confirm or /cancel before sending another message.")
        return

    try:
        check_rate_limit(f"telegram:{chat.id}")
    except RateLimitExceeded:
        await _reply(message, "Too many requests. Please try again later.")
        return
    if not has_gemini_api_key():
        await _reply(message, "The AI service is not configured. Please contact the administrator.")
        return

    history = list(_conversations.get(chat.id, []))
    try:
        result = await run_in_threadpool(generate, message.text, history)
    except InputRejected:
        await _reply(message, "This request was refused by the input safety checks.")
        return
    except GeminiAuthenticationError:
        await _reply(message, "The AI service rejected its configured API key.")
        return
    except (GeminiRateLimitError, GeminiConnectionError):
        await _reply(message, "The AI service is temporarily unavailable.")
        return
    except (UnsafeOutput, RerankError, RuntimeError) as error:
        logger.warning("telegram_chat_failed reason=%s", type(error).__name__)
        await _reply(message, "The assistant could not safely complete this request.")
        return

    pending = result.get("pending_action")
    answer = result["answer"]
    if pending:
        _pending_actions[chat.id] = pending
        answer = f"{answer}\n\n{pending['summary']}\nReply with /confirm to proceed or /cancel to stop."
    _remember_turn(chat.id, message.text, answer)

    sources = result.get("sources", [])
    if sources:
        labels = [
            f"{source['filename']}"
            + (f", page {source['page']}" if source.get("page") is not None else "")
            for source in sources
        ]
        answer = f"{answer}\n\nSources: {'; '.join(labels)}"
    await _reply(message, answer)


def create_telegram_application(token: str) -> Application:
    application = Application.builder().token(token).build()
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("reset", reset_command))
    application.add_handler(CommandHandler("confirm", confirm_command))
    application.add_handler(CommandHandler("cancel", cancel_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    return application


def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Set TELEGRAM_BOT_TOKEN before starting Telegram polling.")
    create_telegram_application(token).run_polling()


if __name__ == "__main__":
    main()
