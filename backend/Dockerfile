FROM python:3.11-slim

WORKDIR /app

# Copy requirements first so this layer is cached unless requirements change.
# requirements.txt — inference-only (~380 MB installed, no torch/transformers).
# requirements-extras.txt — add only if depth.enabled or vlm.enabled in default.yaml.
COPY requirements.txt requirements-extras.txt* ./

# Install system dependencies, pip packages, then remove build tools — all in
# one RUN layer so Docker doesn't cache the build tools in a separate layer.
#
# Runtime libs that MUST stay (OpenCV + MediaPipe need them at runtime):
#   libgl1          — OpenCV libGL
#   libglib2.0-0    — OpenCV libglib
#   libsm6          — OpenCV libSM
#   libxext6        — OpenCV libXext
#   libxrender-dev  — OpenCV libXrender
#   curl            — used in healthchecks / model download scripts
#
# Build tools removed after install (only needed to compile C extensions):
#   build-essential, gcc, g++
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential gcc g++ \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev curl ffmpeg \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential gcc g++ \
    && rm -rf /var/lib/apt/lists/* /root/.cache/pip

COPY api/ ./api/
COPY worker/ ./worker/

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    # Prevent HuggingFace from re-downloading models on every container start
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    HF_HOME=/app/.cache/huggingface

EXPOSE 5050

CMD ["python", "-m", "api.server"]
