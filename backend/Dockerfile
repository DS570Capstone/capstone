FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/
COPY worker/ ./worker/

ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app

# Default overridden by docker-compose per service
CMD ["python", "-m", "api.server"]
