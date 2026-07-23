FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install deps first so application edits don't bust the layer cache.
# `.[agent]` pulls langgraph, needed when USE_AGENT=true.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[agent]"

COPY corpus ./corpus
COPY scripts ./scripts

EXPOSE 8000
CMD ["uvicorn", "platform_copilot.main:app", "--host", "0.0.0.0", "--port", "8000"]
