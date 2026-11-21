FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

LABEL org.opencontainers.image.title="vie-phim-cloud-worker" \
      org.opencontainers.image.description="Prebaked CUDA 12.4 AI audio/dubbing runtime (Zero weights, Zero secrets)" \
      org.opencontainers.image.source="https://github.com/phamductam24vn/viephim-cloud-image"

ENV DEBIAN_FRONTEND=noninteractive \
    VIEPHIM_PREBAKED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    UV_NO_CACHE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/.cache/huggingface \
    TORCH_HOME=/workspace/.cache/torch \
    XDG_CACHE_HOME=/workspace/.cache

WORKDIR /app

# Install system binaries (ffmpeg, cloudflared, uv) and clean apt cache in one layer
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        curl \
        ca-certificates \
        libsndfile1 \
    && curl -fsSL -o /usr/local/bin/cloudflared \
        https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
    && chmod +x /usr/local/bin/cloudflared \
    && curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /app /workspace/.cache/huggingface /workspace/.cache/torch

# Copy ONLY dependency manifest (never bake worker_server.py, secrets, or model weights)
COPY requirements-cloud.txt /app/requirements-cloud.txt

# Lock ONLY base PyTorch/CUDA core packages in /app/constraints.txt (allow upgrades for helper packages)
# Install in clean conflict-free groups: Step A (Audio/Separator) -> Step B (whisperx) -> Step C (vieneu)
RUN python -m pip freeze \
        | grep -E '^(torch(audio|vision)?|nvidia-|triton)(==|@)' \
        > /app/constraints.txt \
    && test -s /app/constraints.txt \
    && grep -vE '^(whisperx|vieneu)' /app/requirements-cloud.txt > /tmp/req-step-a.txt \
    && uv pip install --system --no-cache \
        -c /app/constraints.txt \
        -r /tmp/req-step-a.txt \
    && uv pip install --system --no-cache \
        -c /app/constraints.txt \
        "whisperx>=3.2.0" "numpy>=1.26.0,<2.0.0" \
    && uv pip install --system --no-cache \
        -c /app/constraints.txt \
        "vieneu>=3.8.5" "numpy>=1.26.0,<2.0.0" \
    && rm -f /tmp/req-step-a.txt \
    && python -c "import torch, fastapi, uvicorn, hf_transfer, soundfile, soxr, whisperx, audio_separator, faster_whisper, pyannote.audio"

EXPOSE 8000
