FROM python:3.11-slim

LABEL org.opencontainers.image.title="email-triage-openenv"
LABEL openenv.tags="openenv,email-triage,nlp,agent"

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./

RUN pip install --no-cache-dir \
    fastapi>=0.110.0 \
    "uvicorn[standard]>=0.29.0" \
    "pydantic>=2.6.0" \
    "openai>=1.30.0" \
    "requests>=2.31.0" \
    pyyaml>=6.0 \
    httpx>=0.27.0

COPY . .

RUN touch data/__init__.py graders/__init__.py server/__init__.py

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
