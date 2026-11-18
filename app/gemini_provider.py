"""Centralized Google Gemini SDK provider."""

import logging
import os
import time
from typing import Any, Callable

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError

from app.common import BASE_DIR

logger = logging.getLogger("app.gemini")


class GeminiError(RuntimeError):
    pass


class GeminiAuthenticationError(GeminiError):
    pass


class GeminiRateLimitError(GeminiError):
    pass


class GeminiConnectionError(GeminiError):
    pass


load_dotenv(BASE_DIR / ".env", override=True, encoding="utf-8-sig")


class GeminiProvider:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        embedding_model: str | None = None,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self.api_key = (api_key if api_key is not None else os.getenv("GEMINI_API_KEY", "")).strip()
        if not self.api_key:
            raise GeminiAuthenticationError("GEMINI_API_KEY is not configured.")
        self.model = (model or os.getenv("GEMINI_MODEL", "gemini-3.8-flash")).strip()
        self.embedding_model = (
            embedding_model
            or os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
        ).strip()
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(1, max_retries)
        self.client = genai.Client(
            api_key=self.api_key,
            http_options=types.HttpOptions(timeout=int(timeout_seconds * 1000)),
        )

    def _is_retryable(self, error: Exception) -> bool:
        code = getattr(error, "code", None)
        if code in {408, 429, 500, 502, 503, 504}:
            return True
        status = str(getattr(error, "status", "")).upper()
        return "UNAVAILABLE" in status or "RESOURCE_EXHAUSTED" in status

    def _call(self, operation: str, callback: Callable):
        for attempt in range(self.max_retries):
            try:
                return callback()
            except APIError as error:
                code = getattr(error, "code", None)
                if code in {401, 403}:
                    raise GeminiAuthenticationError("Gemini rejected the API key.") from error
                if code == 429 or self._is_retryable(error):
                    if attempt + 1 >= self.max_retries:
                        if code == 429:
                            raise GeminiRateLimitError("Gemini quota or rate limit was exceeded.") from error
                        raise GeminiConnectionError("Gemini is temporarily unavailable.") from error
                    delay = 2 ** attempt
                    logger.warning("gemini_retry operation=%s attempt=%d delay=%d", operation, attempt + 1, delay)
                    time.sleep(delay)
                    continue
                raise GeminiError("Gemini rejected the request.") from error
            except (TimeoutError, ConnectionError, OSError) as error:
                if attempt + 1 >= self.max_retries:
                    raise GeminiConnectionError("Could not reach Gemini.") from error
                delay = 2 ** attempt
                logger.warning("gemini_retry operation=%s attempt=%d delay=%d", operation, attempt + 1, delay)
                time.sleep(delay)
        raise GeminiConnectionError("Gemini request failed after retries.")

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        system_instruction: str | None = None,
        temperature: float = 0.2,
        max_output_tokens: int = 1024,
        tools: list[dict[str, Any]] | None = None,
        response_mime_type: str | None = None,
    ) -> dict[str, Any]:
        contents: list[types.Content] = []
        message_system_parts: list[str] = []
        for message in messages:
            if message.get("role") == "system":
                if message.get("content"):
                    message_system_parts.append(str(message["content"]))
                continue
            role = "model" if message.get("role") == "assistant" else "user"
            parts: list[types.Part] = []
            if message.get("content"):
                parts.append(types.Part(text=str(message["content"])))
            for tool_call in message.get("tool_calls", []) or []:
                parts.append(types.Part(
                    function_call=types.FunctionCall(
                        id=str(tool_call.get("id", "")),
                        name=str(tool_call["name"]),
                        args=tool_call.get("arguments", {}),
                    )
                ))
            if message.get("tool_call_id") and message.get("tool_name"):
                parts.append(types.Part(
                    function_response=types.FunctionResponse(
                        id=message["tool_call_id"],
                        name=message["tool_name"],
                        response=message.get("result", {}),
                    )
                ))
            if parts:
                contents.append(types.Content(role=role, parts=parts))

        config_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        elif message_system_parts:
            config_kwargs["system_instruction"] = "\n\n".join(message_system_parts)
        if response_mime_type:
            config_kwargs["response_mime_type"] = response_mime_type
        if tools:
            declarations = []
            for tool in tools:
                function = tool["function"]
                declarations.append(types.FunctionDeclaration(
                    name=function["name"],
                    description=function.get("description", ""),
                    parameters=function.get("parameters"),
                ))
            config_kwargs["tools"] = [types.Tool(function_declarations=declarations)]

        def operation():
            return self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(**config_kwargs),
            )

        response = self._call("generate", operation)
        text_parts: list[str] = []
        function_calls: list[dict[str, Any]] = []
        for candidate in getattr(response, "candidates", []) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", []) or []:
                text = getattr(part, "text", None)
                if text:
                    text_parts.append(text)
                call = getattr(part, "function_call", None)
                if call is not None:
                    function_calls.append({
                        "id": getattr(call, "id", "") or f"gemini-{len(function_calls) + 1}",
                        "name": call.name,
                        "arguments": dict(call.args or {}),
                    })
        return {"text": "".join(text_parts).strip(), "function_calls": function_calls}

    def embed_documents(self, texts: list[str], batch_size: int = 100) -> list[list[float]]:
        return self._embed(texts, "RETRIEVAL_DOCUMENT", batch_size)

    def embed_query(self, text: str) -> list[float]:
        vectors = self._embed([text], "RETRIEVAL_QUERY", 1)
        return vectors[0]

    def _embed(self, texts: list[str], task_type: str, batch_size: int) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        dimension: int | None = None
        for start in range(0, len(texts), max(1, batch_size)):
            batch = texts[start:start + max(1, batch_size)]

            def operation(batch=batch):
                return self.client.models.embed_content(
                    model=self.embedding_model,
                    contents=batch,
                    config=types.EmbedContentConfig(task_type=task_type),
                )

            response = self._call(f"embed_{task_type.lower()}", operation)
            embeddings = getattr(response, "embeddings", None) or []
            if len(embeddings) != len(batch):
                raise GeminiError("Gemini returned an incomplete embedding batch.")
            for item in embeddings:
                values = getattr(item, "values", None)
                if not values or not all(isinstance(value, (int, float)) for value in values):
                    raise GeminiError("Gemini returned an invalid embedding vector.")
                vector = [float(value) for value in values]
                if dimension is None:
                    dimension = len(vector)
                elif len(vector) != dimension:
                    raise GeminiError("Gemini returned inconsistent embedding dimensions.")
                vectors.append(vector)
        return vectors


_provider: GeminiProvider | None = None


def get_gemini_provider() -> GeminiProvider:
    global _provider
    if _provider is None:
        _provider = GeminiProvider()
    return _provider


def has_gemini_api_key() -> bool:
    return bool(os.getenv("GEMINI_API_KEY", "").strip())
