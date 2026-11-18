FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && useradd --create-home appuser

COPY app ./app
COPY static ./static
COPY data/docs ./data/docs

RUN mkdir -p /app/chroma_db /app/logs \
    && chown -R appuser:appuser /app

USER appuser
EXPOSE 8000

CMD ["sh", "-c", "python -m app.startup && exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
