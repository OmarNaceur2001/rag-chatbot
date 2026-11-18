# TechNova RAG Chatbot

A FastAPI RAG chatbot for TechNova documents. It uses Google Gemini for answer
generation, reranking, query rewriting, and embeddings, with ChromaDB,
BM25/vector hybrid retrieval, reciprocal-rank fusion, guardrails, optional tools,
Telegram, Docker, and Render support.

## Requirements

- Python 3.13 recommended
- A Gemini API key
- Docker only for container execution
- A Telegram bot token only for Telegram

## Gemini setup

Copy `.env.example` to `.env`, then set at least:

```text
GEMINI_API_KEY=<your-key>
GEMINI_MODEL=gemini-3.8-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
CHROMA_COLLECTION_NAME=docs_gemini
```

Do not commit `.env` or paste credentials in source, logs, issues, or chat.
The SDK is `google-genai`. Free-tier availability and quotas can change; the
application retries bounded transient failures but cannot override provider
quota limits.

## Install and run

Windows:

```powershell
py -3.13 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python -m app.ingest
python -m uvicorn app.main:app --reload
```

macOS/Linux:

```bash
python3.13 -m venv venv
. venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
python -m app.ingest
python -m uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`.

## Index migration

The old OpenAI index is preserved as the `docs` collection. Gemini vectors are
stored in the separate `docs_gemini` collection and are never mixed with old
vectors.

```bash
python -m app.ingest
python -m app.ingest --search "How long is the warranty?"
```

Ingestion extracts PDF/DOCX/TXT or crawled text, preserves chunk metadata, embeds
documents with `RETRIEVAL_DOCUMENT`, and replaces only `docs_gemini` after all
vectors are created and dimensions are consistent. Query searches use
`RETRIEVAL_QUERY`. The old `docs` collection is not deleted automatically.

Validate retrieval before switching production traffic:

```bash
python -m eval.run_eval
```

Set `CHROMA_COLLECTION_NAME` only when intentionally selecting another versioned
Gemini collection.

## Web and Telegram

The web chat keeps five messages per browser session. Health is `GET /health`.
Telegram supports `/start`, `/reset`, `/confirm`, and `/cancel`; write tools
still require confirmation.

Polling:

```text
TELEGRAM_MODE=off
TELEGRAM_BOT_TOKEN=<BotFather token>
```

```bash
python -m app.telegram_bot
```

Webhook:

```text
TELEGRAM_MODE=webhook
TELEGRAM_BOT_TOKEN=<BotFather token>
TELEGRAM_WEBHOOK_SECRET=<random letters, digits, underscores, hyphens>
PUBLIC_BASE_URL=https://your-service.example
```

## Docker

```bash
docker build -t technova-rag-chatbot .
docker run --env-file .env -p 8000:8000 technova-rag-chatbot
```

The image contains application code and sample documents, not `.env` or a local
vector database. Startup builds the Gemini index if needed. Keep the old
`chroma_db/` volume/index out of containers unless it is a deliberate local
archive.

## Render

1. Connect the repository and deploy `render.yaml`.
2. Set `GEMINI_API_KEY`, `GEMINI_MODEL`, and `GEMINI_EMBEDDING_MODEL` as service
   environment variables.
3. Optionally configure `CHROMA_COLLECTION_NAME=docs_gemini`.
4. Configure Telegram variables only if using Telegram.
5. Use a persistent disk or always-on plan for a stable index.

The free Render filesystem is ephemeral, so its index may be rebuilt after a
restart. Rebuilding uses Gemini embedding quota.

## Tests

```powershell
python -m compileall -q app eval tests
python -m pip check
python -m unittest discover -s tests -v
```

Unit tests mock Gemini responses and do not require API quota. Real integration
tests should only be run after setting a valid key.

## Evaluation

The benchmark contains 30 reproducible cases. It reports retrieval hit rate,
answer/source pass results, failure rate, average latency, and total latency.
It does not fabricate scores and stops with a clear API error if Gemini rejects
authentication, connectivity, or quota.

## Troubleshooting

- Missing key: set `GEMINI_API_KEY` in `.env` or the deployment environment.
- Invalid key or model: verify both in Google AI Studio and confirm
  `GEMINI_MODEL=gemini-3.8-flash`.
- Embedding not found: verify `gemini-embedding-001` and re-run ingestion.
- Missing collection: run `python -m app.ingest`.
- 429/quota: wait for quota refresh or review free-tier limits.
- Telegram webhook: verify HTTPS, bot token, secret, public URL, and
  `X-Telegram-Bot-Api-Secret-Token`.
# rag-chatbot
