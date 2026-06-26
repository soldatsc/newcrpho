# syntax=docker/dockerfile:1
FROM runpod/worker-comfyui:5.7.1-base

# ============================================================
# insightface 0.7.3 (prebuilt wheel, numpy 1.x) — mirror of worker_swap
# ============================================================
RUN pip install --no-cache-dir "numpy==1.26.4" onnxruntime \
    https://huggingface.co/AlienMachineAI/insightface-0.7.3-cp312-cp312-linux_x86_64.whl/resolve/main/insightface-0.7.3-cp312-cp312-linux_x86_64.whl

RUN pip install --no-cache-dir ultralytics segment-anything scikit-image piexif accelerate "transformers>=4.47" sentencepiece einops timm

# ============================================================
# LoRAs from Civitai (token via BuildKit secret). Placed EARLY so a bad/empty
# token fails the build in seconds instead of after the ~18GB model download.
# wget -q keeps the token out of logs; on failure we dump the error body
# (never the token) so the cause (Unauthorized / Not Found / ...) is visible.
# ============================================================
RUN --mount=type=secret,id=civitai bash -c 'set -e; \
  T=$(tr -d "\n\r \t" < /run/secrets/civitai); \
  if [ -z "$T" ]; then echo "ERR: CIVITAI_TOKEN secret is EMPTY (set it in GitHub Secrets)"; exit 1; fi; \
  echo "civitai token length: ${#T} chars"; \
  mkdir -p /comfyui/models/loras; \
  dl(){ \
    wget -q -O "/comfyui/models/loras/$1" "https://civitai.com/api/download/models/$2?token=$T" || true; \
    sz=$(wc -c < "/comfyui/models/loras/$1" 2>/dev/null || echo 0); \
    echo "[$1] version=$2 size=${sz}B"; \
    if [ "$sz" -lt 1000000 ]; then echo "  FAILED - response head (no token):"; head -c 400 "/comfyui/models/loras/$1" 2>/dev/null; echo; exit 1; fi; \
  }; \
  dl realism_engine_krea2_v2.safetensors 3070702; \
  dl MysticXXX_KREA2_v1.safetensors 3067313; \
  ls -la /comfyui/models/loras/'

# ============================================================
# Custom nodes (only what the Krea workflow needs)
# ============================================================
RUN comfy-node-install rgthree-comfy
# ReActor — Boostedforce NSFW fork (filter removed at source, no patch needed)
RUN git clone --depth 1 https://github.com/Boostedforce/ComfyUI-ReActor-NSFW /comfyui/custom_nodes/ComfyUI-ReActor-NSFW && \
    pip install --no-cache-dir -r /comfyui/custom_nodes/ComfyUI-ReActor-NSFW/requirements.txt || true
RUN comfy-node-install comfyui-impact-pack
RUN comfy-node-install comfyui-impact-subpack

RUN D=$(find /comfyui/custom_nodes -maxdepth 1 -iname "*impact-pack*" -type d|head -1); [ -f "$D/install.py" ] && cd "$D" && python install.py || true
RUN D=$(find /comfyui/custom_nodes -maxdepth 1 -iname "*impact-subpack*" -type d|head -1); [ -f "$D/install.py" ] && cd "$D" && python install.py || true

# (No NSFW patch needed — Boostedforce fork ships with the filter removed.)

# ============================================================
# Force-reinstall insightface + pin numpy (nodes may overwrite)
# ============================================================
RUN pip install --no-cache-dir --no-deps --force-reinstall \
    https://huggingface.co/AlienMachineAI/insightface-0.7.3-cp312-cp312-linux_x86_64.whl/resolve/main/insightface-0.7.3-cp312-cp312-linux_x86_64.whl
RUN pip install --no-cache-dir "numpy==1.26.4"

# ============================================================
# Force CPU onnxruntime. A node's requirements pulled onnxruntime-gpu built
# for CUDA 13, but this image is CUDA 12.6 -> "libcudart.so.13: cannot open
# shared object file" -> ReActor's `import onnxruntime` dies -> node never
# registers ("ReActorFaceSwap does not exist"). CPU build = no CUDA dep.
# ============================================================
RUN pip uninstall -y onnxruntime onnxruntime-gpu || true && pip install --no-cache-dir onnxruntime
RUN python3 -c "import onnxruntime; print('onnxruntime', onnxruntime.__version__, 'import OK')"
RUN python3 -c "import insightface; print('insightface', insightface.__version__, 'import OK')"

# ============================================================
# Models — KREA 2
# ============================================================
RUN comfy model download --url https://huggingface.co/Comfy-Org/Krea-2/resolve/main/diffusion_models/krea2_turbo_fp8_scaled.safetensors --relative-path models/diffusion_models --filename krea2_turbo_fp8_scaled.safetensors
RUN comfy model download --url https://huggingface.co/Comfy-Org/Krea-2/resolve/main/text_encoders/qwen3vl_4b_fp8_scaled.safetensors --relative-path models/text_encoders --filename qwen3vl_4b_fp8_scaled.safetensors
RUN comfy model download --url https://huggingface.co/Comfy-Org/Krea-2/resolve/main/vae/qwen_image_vae.safetensors --relative-path models/vae --filename qwen_image_vae.safetensors

# Swap models (same sources as worker_swap)
RUN comfy model download --url https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/inswapper_128.onnx --relative-path models/insightface --filename inswapper_128.onnx
RUN comfy model download --url https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GPEN-BFR-512.onnx --relative-path models/facerestore_models --filename GPEN-BFR-512.onnx
RUN comfy model download --url https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/detection/bbox/face_yolov8m.pt --relative-path models/ultralytics/bbox --filename face_yolov8m.pt

# buffalo_l face analysis models
RUN mkdir -p /comfyui/models/insightface/models/buffalo_l && cd /tmp && \
    wget -q https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip && \
    python3 -c "import zipfile; z=zipfile.ZipFile('buffalo_l.zip'); z.extractall('b'); z.close()" && \
    find b -name "*.onnx" -exec cp {} /comfyui/models/insightface/models/buffalo_l/ \; && \
    rm -rf buffalo_l.zip b

# ============================================================
# Verification (fail build loud if anything missing)
# ============================================================
RUN echo "=== custom_nodes ===" && ls /comfyui/custom_nodes/ && \
    test -f /comfyui/models/diffusion_models/krea2_turbo_fp8_scaled.safetensors && echo "Krea UNET OK" && \
    test -f /comfyui/models/text_encoders/qwen3vl_4b_fp8_scaled.safetensors && echo "Encoder OK" && \
    test -f /comfyui/models/vae/qwen_image_vae.safetensors && echo "VAE OK" && \
    test -f /comfyui/models/insightface/inswapper_128.onnx && echo "inswapper OK" && \
    test -f /comfyui/models/facerestore_models/GPEN-BFR-512.onnx && echo "GPEN OK" && \
    test -f /comfyui/models/ultralytics/bbox/face_yolov8m.pt && echo "YOLO OK" && \
    ls /comfyui/models/insightface/models/buffalo_l/*.onnx > /dev/null && echo "buffalo_l OK" && \
    test -d "$(find /comfyui/custom_nodes -maxdepth 1 -iname '*reactor*' -type d|head -1)" && echo "ReActor dir OK" && \
    echo "=== ALL OK ==="
