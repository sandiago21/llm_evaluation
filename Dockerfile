FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# -------------------------
# System dependencies
# -------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# -------------------------
# Install Python runtime dependencies (pre-source COPY so the layer is cached).
# -------------------------
COPY requirements.txt ./
RUN pip install -r requirements.txt

# -------------------------
# Copy only what the runtime needs — keep the image free of host-side
# editor caches, .git, notebooks, dev requirements, and __pycache__ dirs.
# -------------------------
COPY src ./src
COPY configs ./configs
COPY data ./data
COPY main.py ./

EXPOSE 8000

# HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
#     CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

