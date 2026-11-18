# Docker and Render deployment

## Docker

1. Copy `.env.example` to `.env` and set `GEMINI_API_KEY`.
2. Optionally set `GEMINI_MODEL=gemini-3.8-flash` and
   `GEMINI_EMBEDDING_MODEL=gemini-embedding-001`.
3. Build and run:

```powershell
docker build -t technova-rag-chatbot .
docker run --env-file .env -p 8000:8000 technova-rag-chatbot
```

The container does not contain `.env` or the existing Chroma database. Startup
creates or validates the Gemini collection. Rebuild the Gemini index after
deploying to a fresh volume.

## Render

Connect the repository and use `render.yaml`. Set these secrets/environment
values:

- `GEMINI_API_KEY`
- `GEMINI_MODEL=gemini-3.8-flash`
- `GEMINI_EMBEDDING_MODEL=gemini-embedding-001`
- `CHROMA_COLLECTION_NAME=docs_gemini`

Render free instances have ephemeral filesystems. A restart can rebuild the
index and consume embedding quota. Use persistent storage or an always-on plan
for stable client deployments.

## Telegram webhook

Set `TELEGRAM_MODE=webhook`, `TELEGRAM_BOT_TOKEN`,
`TELEGRAM_WEBHOOK_SECRET`, and `PUBLIC_BASE_URL`. The public origin must use
HTTPS. The webhook validates Telegram's secret-token header and uses the same
Gemini provider as the web chat.
