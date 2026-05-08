FROM python:3.12-slim

# -------------------------
# System dependencies
# -------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# -------------------------
# Working directory
# -------------------------
WORKDIR /app

# -------------------------
# Install Python dependencies
# -------------------------
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# -------------------------
# Copy source code
# -------------------------
COPY . .

# -------------------------
# Expose FastAPI port
# -------------------------
EXPOSE 8000

# -------------------------
# Start API server
# -------------------------
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

