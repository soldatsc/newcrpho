# syntax=docker/dockerfile:1
FROM runpod/worker-comfyui:5.7.1-base

# ============================================================
# insightface 0.7.3 (prebuilt wheel, numpy 1.x) — mirror of worker_swap
# ============================================================
RUN pip install --no-cache-dir "numpy==1.26.4" onnxruntime \
    https://huggingface.co/AlienMachineAI/insightface-0.7.3-cp312-cp312-linux_x86_64.whl/resolve/main/insightface-0.7.3-cp312-cp312-linux_x86_64.whl

RUN pip install --no-cache-dir ultralytics segment-anything scikit-image piexif accelerate "transformers>=4.47" sentencepiece einops timm

# ============================================================
# Custom nodes (only what the Krea workflow needs)
# ============================================================
RUN comfy-node-install rgthree-comfy
RUN comfy-node-install comfyui-reactor
RUN comfy-node-install comfyui-impact-pack
RUN comfy-node-install comfyui-impact-subpack

RUN D=$(find /comfyui/custom_nodes -maxdepth 1 -iname "*impact-pack*" -type d|head -1); [ -f "$D/install.py" ] && cd "$D" && python install.py || true
RUN D=$(find /comfyui/custom_nodes -maxdepth 1 -iname "*impact-subpack*" -type d|head -1); [ -f "$D/install.py" ] && cd "$D" && python install.py || true

# ============================================================
# Patch ReActor — disable NSFW filter (same patch as worker_swap)
# ============================================================
RUN R=$(find /comfyui/custom_nodes -maxdepth 1 -iname "*reactor*" -type d|head -1); F="$R/scripts/reactor_sfw.py"; \
    sed -i 's/SCORE = [0-9][0-9.]*/SCORE = 1.1/g' "$F"; \
    sed -i 's/score > SCORE/score > 999/g' "$F"; \
    sed -i 's/nsfw_score > /nsfw_score > 999 and nsfw_score > /g' "$F"; \
    echo "NSFW patch done"

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
# LoRAs from Civitai (token via BuildKit secret — NOT baked into a layer)
# ============================================================
RUN --mount=type=secret,id=civitai mkdir -p /comfyui/models/loras && T=$(cat /run/secrets/civitai) && \
    wget -q -O /comfyui/models/loras/realism_engine_krea2_v2.safetensors "https://civitai.com/api/download/models/3070702?token=$T" && \
    wget -q -O /comfyui/models/loras/MysticXXX_KREA2_v1.safetensors "https://civitai.com/api/download/models/3067313?token=$T" && \
    ls -la /comfyui/models/loras/ && \
    [ $(wc -c < /comfyui/models/loras/realism_engine_krea2_v2.safetensors) -gt 1000000 ] || (echo "ERR: realism lora download failed (check CIVITAI_TOKEN)" && exit 1) && \
    [ $(wc -c < /comfyui/models/loras/MysticXXX_KREA2_v1.safetensors) -gt 1000000 ] || (echo "ERR: mystic lora download failed" && exit 1)

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
    R=$(find /comfyui/custom_nodes -maxdepth 1 -iname '*reactor*' -type d|head -1) && grep -rq "SCORE = 1.1" "$R" && echo "NSFW patch OK" && \
    echo "=== ALL OK ==="
