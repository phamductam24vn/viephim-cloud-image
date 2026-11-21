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

# Install system binaries (audio/TTS build & runtime tools, cloudflared, uv) and clean apt cache
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        curl \
        ca-certificates \
        libsndfile1 \
        espeak-ng \
        build-essential \
    && curl -fsSL -o /usr/local/bin/cloudflared \
        https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
    && chmod +x /usr/local/bin/cloudflared \
    && curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /app /workspace/.cache/huggingface /workspace/.cache/torch

# Copy ONLY dependency manifest (never bake worker_server.py, secrets, or model weights)
COPY requirements-cloud.txt /app/requirements-cloud.txt

# RUN 1: Extract clean version constraints (NO '@ file://' Conda URLs) for base PyTorch/CUDA core
RUN python -c 'import importlib.metadata as m; g = {d.metadata["Name"].lower(): d.version.split("+", 1)[0] for d in m.distributions() if d.metadata.get("Name")}; lines = [f"{p}=={g[p]}" for p in ("torch", "torchaudio", "torchvision", "triton") if p in g]; open("/app/constraints.txt", "w").write("\n".join(lines) + "\n"); print("=== /app/constraints.txt ===\n" + "\n".join(lines))' \
    && test -s /app/constraints.txt

# RUN 2: Web Core + Audio + BS-RoFormer Separator (<0.31.0 keeps numpy<2.0.0 & CUDA 12.4 compatibility)
RUN uv pip install --system --no-cache -c /app/constraints.txt \
        "fastapi>=0.115.0,<0.120" \
        "uvicorn[standard]>=0.32.0,<0.36" \
        "httpx>=0.27.0,<0.29" \
        "python-multipart>=0.0.12" \
        "pydantic>=2.7.0,<3.0" \
        "hf_transfer>=0.1.8" \
        "numpy>=1.26.0,<2.0.0" \
        "soundfile>=0.12.1" \
        "soxr>=0.5.0" \
        "scipy>=1.12.0" \
        "pyloudnorm>=0.1.1" \
        "audio-separator[gpu]>=0.22.0,<0.31.0"

# RUN 3: STT & Diarization (<3.3.3 pins PyTorch 2.5.1 + numpy<2.0.0 compatible whisperx/faster-whisper/pyannote)
RUN uv pip install --system --no-cache -c /app/constraints.txt \
        "whisperx>=3.2.0,<3.3.3" \
        "numpy>=1.26.0,<2.0.0"

# RUN 4: VieNeu TTS, restore onnxruntime-gpu native runtime over CPU onnxruntime, and verify all imports
RUN uv pip install --system --no-cache -c /app/constraints.txt \
        "vieneu>=3.8.5" \
        "numpy>=1.26.0,<2.0.0" \
    && uv pip install --system --no-cache --reinstall --no-deps "onnxruntime-gpu>=1.20.0" \
    && python -c 'import torch, torchaudio, torchvision, fastapi, uvicorn, httpx, hf_transfer, numpy, soundfile, soxr, scipy, pyloudnorm, audio_separator, whisperx, faster_whisper, pyannote.audio, vieneu, sea_g2p; assert numpy.__version__.startswith("1."), numpy.__version__; assert torch.__version__.startswith("2.5"), torch.__version__; print("IMPORT_VERIFY_OK:", torch.__version__, numpy.__version__)'

EXPOSE 8000
