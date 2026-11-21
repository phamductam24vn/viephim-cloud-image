FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

LABEL org.opencontainers.image.title="vie-phim-cloud-worker" \
      org.opencontainers.image.description="Prebaked CUDA 12.4 + cuDNN 9 AI audio/dubbing runtime (Zero weights, Zero secrets)" \
      org.opencontainers.image.source="https://github.com/phamductam24vn/viephim-cloud-image"

ENV DEBIAN_FRONTEND=noninteractive \
    VIEPHIM_PREBAKED=1 \
    HF_XET_HIGH_PERFORMANCE=1 \
    UV_NO_CACHE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/.cache/huggingface \
    TORCH_HOME=/workspace/.cache/torch \
    XDG_CACHE_HOME=/workspace/.cache \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cublas/lib:/opt/conda/lib/python3.11/site-packages/torch/lib:/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib

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

# RUN 1: Extract clean version constraints (NO '@ file://' Conda URLs) for base PyTorch/CUDA/cuDNN core
RUN python -c 'import importlib.metadata as m; g = {d.metadata["Name"].lower(): d.version.split("+", 1)[0] for d in m.distributions() if d.metadata.get("Name")}; core = [f"{p}=={g[p]}" for p in ("torch", "torchaudio", "torchvision", "triton") if p in g] + [f"{k}=={v}" for k, v in sorted(g.items()) if k.startswith("nvidia-")]; open("/app/constraints.txt", "w").write("\n".join(core) + "\n"); print("=== /app/constraints.txt ===\n" + "\n".join(core))' \
    && test -s /app/constraints.txt

# RUN 2: Web Core + Audio + BS-RoFormer Separator (<0.31.0 keeps numpy<2.0.0 & CUDA 12.4 compatibility)
RUN uv pip install --system --no-cache -c /app/constraints.txt \
        "fastapi>=0.115.0,<0.120" \
        "uvicorn[standard]>=0.32.0,<0.36" \
        "httpx>=0.27.0,<0.29" \
        "python-multipart>=0.0.12" \
        "pydantic>=2.7.0,<3.0" \
        "huggingface_hub[hf_xet]>=0.30.0" \
        "hf_transfer>=0.1.8" \
        "numpy>=1.26.0,<2.0.0" \
        "soundfile>=0.12.1" \
        "soxr>=0.5.0" \
        "scipy>=1.12.0" \
        "pyloudnorm>=0.1.1" \
        "audio-separator[gpu]>=0.22.0,<0.31.0"

# RUN 3: STT & Diarization — Pin transformers<5.0.0 (PyTorch 2.5.1 compat) & upgrade ctranslate2>=4.5.0 (cuDNN 9)
RUN uv pip install --system --no-cache -c /app/constraints.txt \
        "whisperx>=3.2.0,<3.3.3" \
        "transformers>=4.48.0,<5.0.0" \
        "numpy>=1.26.0,<2.0.0" \
    && uv pip install --system --no-cache --upgrade --no-deps \
        "ctranslate2>=4.5.0" \
        "faster-whisper>=1.1.0"

# RUN 4: VieNeu TTS, restore onnxruntime-gpu, register dynamic CUDA/cuDNN .so paths, and enforce 2 Hard Gates
RUN uv pip install --system --no-cache -c /app/constraints.txt \
        "vieneu>=3.8.5" \
        "transformers>=4.48.0,<5.0.0" \
        "numpy>=1.26.0,<2.0.0" \
    && uv pip install --system --no-cache --reinstall --no-deps "onnxruntime-gpu>=1.20.0" \
    && python -c 'import pathlib, site; dirs = [p for root in site.getsitepackages() for p in sorted(pathlib.Path(root).glob("*")) if p.is_dir() and any(x in p.name for x in ("torch", "nvidia", "ctranslate2"))]; sub = [str(d / "lib") for d in dirs if (d / "lib").is_dir()] + [str(d) for d in dirs if d.name.endswith(".libs")] + [str(p) for p in sorted(pathlib.Path("/opt/conda/lib/python3.11/site-packages/nvidia").glob("*/lib")) if p.is_dir()] + ["/usr/local/cuda/lib64"]; uniq = [x for i, x in enumerate(sub) if pathlib.Path(x).is_dir() and x not in sub[:i]]; pathlib.Path("/etc/ld.so.conf.d/viephim-cuda.conf").write_text("\n".join(uniq) + "\n")' \
    && ldconfig \
    && python -c 'import pathlib, subprocess, torch, torchaudio, torchvision, fastapi, uvicorn, httpx, huggingface_hub, numpy, soundfile, soxr, scipy, pyloudnorm, audio_separator, transformers, whisperx, faster_whisper, ctranslate2, pyannote.audio, vieneu, sea_g2p; ossl_out = subprocess.check_output(["/usr/bin/openssl", "version"], text=True, stderr=subprocess.STDOUT); assert "mismatch" not in ossl_out.lower(), f"OpenSSL mismatch: {ossl_out}"; curl_ldd = subprocess.check_output(["ldd", "/usr/bin/curl"], text=True, stderr=subprocess.STDOUT); assert "/opt/conda/lib/lib" not in curl_ldd, f"System binary links Conda OpenSSL: {curl_ldd}"; assert numpy.__version__.startswith("1."), numpy.__version__; assert torch.__version__.startswith("2.5"), torch.__version__; assert transformers.is_torch_available(), "Transformers disabled PyTorch!"; ct2_ver = tuple(int(x) for x in ctranslate2.__version__.split(".")[:3]); assert ct2_ver >= (4, 5, 0), f"ctranslate2<4.5.0 uses cuDNN 8: {ctranslate2.__version__}"; scan_dirs = [pathlib.Path(ctranslate2.__file__).resolve().parent]; libs_dir = scan_dirs[0].parent / "ctranslate2.libs"; scan_dirs.extend([libs_dir] if libs_dir.is_dir() else []); sos = [p for d in scan_dirs for p in d.rglob("*.so*") if p.is_file()]; assert sos, "No ctranslate2 .so found"; bad = [f"{p}:{ln}" for p in sos for raw in [p.read_bytes()] if b"libcudnn_ops_infer.so.8" in raw for ln in ["binary_ref"]] + [f"{p}:{ln}" for p in sos for out in [subprocess.check_output(["ldd", str(p)], text=True, stderr=subprocess.STDOUT)] for ln in out.splitlines() if "libcudnn_ops_infer.so.8" in ln or ("not found" in ln.lower() and any(k in ln.lower() for k in ("cudnn", "cublas", "cudart", "ctranslate2")))]; assert not bad, f"cuDNN/ldd check failed: {bad}"; print(f"IMPORT_VERIFY_OK: openssl={ossl_out.strip()} torch={torch.__version__} transformers={transformers.__version__} (torch_avail={transformers.is_torch_available()}) ctranslate2={ctranslate2.__version__} faster_whisper={faster_whisper.__version__} numpy={numpy.__version__}")'

EXPOSE 8000
