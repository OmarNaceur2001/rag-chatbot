"""FastAPI server for the TechNova RAG chat application."""

import asyncio
from contextlib import asynccontextmanager
import hmac
import json
import logging
import os
import secrets
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from telegram import Update

from app.common import BASE_DIR, has_gemini_api_key
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
    validate_input,
)
from app.rag import generate
from app.retrieval import RerankError
from app.telegram_bot import create_telegram_application
from app.tools import execute_confirmed_action

static_dir = BASE_DIR / "static"
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "app.log"
logger = logging.getLogger("app.chat")
logger.setLevel(logging.INFO)
if not any(
    isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == LOG_FILE
    for handler in logger.handlers
):
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(file_handler)


@asynccontextmanager
async def lifespan(application: FastAPI):
    mode = os.getenv("TELEGRAM_MODE", "off").strip().lower()
    if mode == "polling":
        logger.info("telegram_mode=polling start_with=python_-m_app.telegram_bot")
    elif mode == "webhook":
        token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        secret = os.getenv("TELEGRAM_WEBHOOK_SECRET", "").strip()
        public_base_url = os.getenv("PUBLIC_BASE_URL", "").strip().rstrip("/")
        if not token or not secret or not public_base_url:
            raise RuntimeError(
                "Webhook mode requires TELEGRAM_BOT_TOKEN, TELEGRAM_WEBHOOK_SECRET, and PUBLIC_BASE_URL."
            )

        telegram_application = create_telegram_application(token)
        await telegram_application.initialize()
        started = False
        try:
            await telegram_application.start()
            started = True
            await telegram_application.bot.set_webhook(
                url=f"{public_base_url}/telegram/webhook",
                secret_token=secret,
            )
            application.state.telegram_application = telegram_application
            logger.info("telegram_webhook_started")
        except Exception:
            if started:
                await telegram_application.stop()
            await telegram_application.shutdown()
            raise
    elif mode != "off":
        raise RuntimeError("TELEGRAM_MODE must be 'off', 'polling', or 'webhook'.")

    try:
        yield
    finally:
        telegram_application = getattr(application.state, "telegram_application", None)
        if telegram_application is not None:
            await telegram_application.stop()
            await telegram_application.shutdown()
            del application.state.telegram_application


app = FastAPI(title="TechNova RAG Chatbot", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

_conversations: dict[str, list[dict[str, str]]] = {}
_conversation_locks: dict[str, asyncio.Lock] = {}
_pending_actions: dict[str, dict] = {}
_lock_guard = asyncio.Lock()
_MAX_HISTORY_MESSAGES = 5


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


class Source(BaseModel):
    filename: str
    page: int | None = None
    url: str | None = None


class Confirmation(BaseModel):
    action_id: str
    summary: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    confirmation: Confirmation | None = None


class ConfirmationRequest(BaseModel):
    action_id: str = Field(min_length=8, max_length=128)
    approved: bool


async def _get_conversation_lock(session_id: str) -> asyncio.Lock:
    async with _lock_guard:
        return _conversation_locks.setdefault(session_id, asyncio.Lock())


def _require_api_key() -> None:
    if not has_gemini_api_key():
        raise HTTPException(status_code=401, detail="Gemini API key is not configured.")


def _remember_turn(session_id: str, history: list[dict[str, str]], question: str, answer: str) -> None:
    updated_history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    _conversations[session_id] = updated_history[-_MAX_HISTORY_MESSAGES:]


def _log_chat_completed(request_id: str, source_count: int) -> None:
    logger.info(
        "chat_completed request_id=%s source_count=%d",
        request_id,
        source_count,
    )


def _session_id_from_request(request: Request) -> str:
    return request.cookies.get("chat_session", "")


def _check_request_limits(request: Request, question: str) -> None:
    client_ip = request.client.host if request.client else "unknown"
    check_rate_limit(client_ip)
    validate_input(question)


@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = static_dir / "index.html"
    with open(index_path, encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request, response: Response):
    request_id = secrets.token_hex(8)
    try:
        _check_request_limits(request, req.question)
    except InputRejected as error:
        logger.warning("chat_rejected request_id=%s reason=input_guardrail", request_id)
        raise HTTPException(status_code=400, detail=str(error)) from None
    except RateLimitExceeded:
        logger.warning("chat_rejected request_id=%s reason=rate_limit", request_id)
        raise HTTPException(status_code=429, detail="Too many requests. Please try again later.") from None

    try:
        _require_api_key()
    except HTTPException:
        logger.warning("chat_rejected request_id=%s reason=missing_api_key", request_id)
        raise

    session_id = _session_id_from_request(request)
    if not session_id or len(session_id) > 128:
        session_id = secrets.token_urlsafe(32)
    lock = await _get_conversation_lock(session_id)

    async with lock:
        if session_id in _pending_actions:
            raise HTTPException(
                status_code=409,
                detail="Please confirm or cancel the pending action before sending another message.",
            )
        history = list(_conversations.get(session_id, []))
        try:
            result = await run_in_threadpool(generate, req.question, history)
        except GeminiAuthenticationError:
            logger.warning("chat_rejected request_id=%s reason=invalid_api_key", request_id)
            raise HTTPException(status_code=401, detail="Gemini API key was rejected.") from None
        except GeminiRateLimitError:
            logger.warning("chat_failed request_id=%s reason=upstream_rate_limit", request_id)
            raise HTTPException(status_code=503, detail="AI service is temporarily unavailable.") from None
        except GeminiConnectionError:
            logger.warning("chat_failed request_id=%s reason=upstream_unavailable", request_id)
            raise HTTPException(status_code=503, detail="AI service is temporarily unavailable.") from None
        except (UnsafeOutput, RerankError, RuntimeError) as error:
            logger.warning(
                "chat_failed request_id=%s reason=%s",
                request_id,
                type(error).__name__,
            )
            raise HTTPException(status_code=502, detail="The assistant could not safely complete this request.") from None

        pending = result.pop("pending_action", None)
        confirmation = None
        if pending:
            _pending_actions[session_id] = pending
            confirmation = Confirmation(
                action_id=pending["action_id"],
                summary=pending["summary"],
            )
        _remember_turn(session_id, history, req.question, result["answer"])

    response.set_cookie(
        key="chat_session",
        value=session_id,
        httponly=True,
        samesite="lax",
        secure=request.url.scheme == "https",
    )
    _log_chat_completed(request_id, len(result.get("sources", [])))
    return ChatResponse(**result, confirmation=confirmation)


@app.post("/chat/confirm", response_model=ChatResponse)
async def confirm_chat_action(
    req: ConfirmationRequest,
    request: Request,
):
    request_id = secrets.token_hex(8)
    session_id = _session_id_from_request(request)
    if not session_id:
        raise HTTPException(status_code=401, detail="Chat session is not available.")
    try:
        check_rate_limit(request.client.host if request.client else "unknown")
    except RateLimitExceeded:
        logger.warning("chat_rejected request_id=%s reason=rate_limit", request_id)
        raise HTTPException(status_code=429, detail="Too many requests. Please try again later.") from None

    lock = await _get_conversation_lock(session_id)
    async with lock:
        pending = _pending_actions.get(session_id)
        if not pending or pending["action_id"] != req.action_id:
            raise HTTPException(status_code=404, detail="Pending action was not found.")

        if req.approved:
            try:
                result = await run_in_threadpool(execute_confirmed_action, pending)
            except (OSError, ValueError) as error:
                logger.warning(
                    "action_failed request_id=%s tool=%s reason=%s",
                    request_id,
                    pending["tool"],
                    type(error).__name__,
                )
                raise HTTPException(status_code=500, detail="The confirmed action could not be completed.") from None
            answer = result["message"]
            status = "approved"
        else:
            answer = "Action cancelled. No changes were made."
            status = "cancelled"

        del _pending_actions[session_id]
        history = list(_conversations.get(session_id, []))
        _remember_turn(
            session_id,
            history,
            f"User {status} the proposed action.",
            answer,
        )

    logger.info("action_%s request_id=%s tool=%s", status, request_id, pending["tool"])
    return ChatResponse(answer=answer, sources=[])


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    secret = os.getenv("TELEGRAM_WEBHOOK_SECRET", "").strip()
    application = getattr(app.state, "telegram_application", None)
    if not secret or application is None:
        raise HTTPException(status_code=503, detail="Telegram webhook is not configured.")
    supplied_secret = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
    if not hmac.compare_digest(supplied_secret, secret):
        raise HTTPException(status_code=403, detail="Invalid webhook secret.")
    try:
        payload = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid Telegram update payload.") from None
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid Telegram update payload.")
    update = Update.de_json(payload, application.bot)
    await application.process_update(update)
    return {"ok": True}
