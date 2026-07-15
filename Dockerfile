# syntax=docker/dockerfile:1
FROM runpod/worker-comfyui:5.8.6-base

# ============================================================
# Update ComfyUI (comfyanonymous) to master for the krea2 CLIP type.
# The 5.8.6 base ships ComfyUI ~0.25.0 (17 Jun) which predates Krea 2 support
# -> CLIPLoader rejects type 'krea2'. master has CLIPType.KREA2.
# comfy-kitchen / comfy-aimdo are separate pip packages (loaded as backends),
# not source patches, so a clean checkout of master leaves them intact.
# ============================================================
# PINNED to the exact master commit the 2026-06-27 build shipped (last commit
# before that build). Floating master was a drift source: every rebuild silently
# changed ComfyUI under BOTH consumers of this image (krea worker + prompt-lab).
# Bump deliberately by editing this sha.
ARG COMFYUI_COMMIT=603d891eaf045d726d9c23276b4428daf2977624
RUN cd /comfyui && git fetch --depth 1 origin ${COMFYUI_COMMIT} && git reset --hard ${COMFYUI_COMMIT} && \
    pip install --no-cache-dir -r requirements.txt && echo "ComfyUI -> $(git rev-parse --short HEAD)"

# ============================================================
# insightface 0.7.3 (prebuilt wheel, numpy 1.x) — mirror of worker_swap
# ============================================================
RUN pip install --no-cache-dir "numpy==1.26.4" onnxruntime \
    https://huggingface.co/AlienMachineAI/insightface-0.7.3-cp312-cp312-linux_x86_64.whl/resolve/main/insightface-0.7.3-cp312-cp312-linux_x86_64.whl

# ultralytics pinned to the version current at the 2026-06-27 build (>=8.3
# needed for YOLO11 NSFW detector below); floating pip here = same drift class
# as floating ComfyUI master.
RUN pip install --no-cache-dir "ultralytics==8.4.65" segment-anything scikit-image piexif numba dill blend-modes accelerate "transformers>=4.47" sentencepiece einops timm

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

# ============================================================
# ADDITIVE v2 (2026-07-15): occlusion-aware faceswap toolkit.
# Nothing above is removed/replaced — the second consumer of this image
# (txt2img prompt-lab endpoint) keeps working byte-identically.
# ============================================================
# NSFW body-part detectors for occluder subtraction (bbox -> SAM -> subtract
# from the face mask so the swap never repaints genitals/objects at the face).
# PRIMARY: EraX-NSFW-V1.0 (Apache-2.0, YOLO11m). Classes:
#   anus, make_love, nipple, penis, vagina  ("make_love" = the act zone —
#   covers blowjob-at-face even with no hands in frame).
# (NudeNet backup was dropped: its GitHub release .pt arrives corrupted —
#  CI smoke-load caught an UnpicklingError; EraX alone covers the need.)
RUN comfy model download --url https://huggingface.co/erax-ai/EraX-NSFW-V1.0/resolve/main/erax_nsfw_yolo11m.pt --relative-path models/ultralytics/bbox --filename erax_nsfw_yolo11m.pt

# CodeFormer restore — was NEVER baked; ReActor used to runtime-download it on
# warm workers, so fresh cold-started workers lost it (the 2026-07-13 "Krea
# перестала работать" incident: validation rejects 'codeformer-v0.1.0.pth').
RUN comfy model download --url https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/codeformer-v0.1.0.pth --relative-path models/facerestore_models --filename codeformer-v0.1.0.pth

# SAM vit_l — finer segmentation near faces (vit_b stays; opt-in by name in
# the workflow's SAMLoader).
RUN mkdir -p /comfyui/models/sams && \
    wget -q -O /comfyui/models/sams/sam_vit_l_0b3195.pth \
      https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth

# 256px swap models — the ReActor NSFW fork selects the swapper by filename
# substring (inswapper/reswapper/hyperswap) and looks in these exact dirs.
# Baked = available; production graphs keep sending inswapper_128.onnx until
# switched explicitly, so nothing changes until we A/B.
RUN mkdir -p /comfyui/models/hyperswap /comfyui/models/reswapper && \
    wget -q -O /comfyui/models/hyperswap/hyperswap_1a_256.onnx \
      https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/hyperswap_1a_256.onnx && \
    wget -q -O /comfyui/models/reswapper/reswapper_256.onnx \
      https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/reswapper_256.onnx

# Impact Subpack whitelist for the new .pt detector (torch weights_only guard).
RUN mkdir -p /comfyui/user/default/ComfyUI-Impact-Subpack && \
    printf 'erax_nsfw_yolo11m.pt\n' >> /comfyui/user/default/ComfyUI-Impact-Subpack/model-whitelist.txt

# Smoke-load the detector at build time (catches weights_only/pickle issues)
# and print the exact class-label strings for the BboxDetectorSEGS `labels` filter.
RUN python3 -c "from ultralytics import YOLO; \
print('EraX:', YOLO('/comfyui/models/ultralytics/bbox/erax_nsfw_yolo11m.pt').names)"

# buffalo_l face analysis models
RUN mkdir -p /comfyui/models/insightface/models/buffalo_l && cd /tmp && \
    wget -q https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip && \
    python3 -c "import zipfile; z=zipfile.ZipFile('buffalo_l.zip'); z.extractall('b'); z.close()" && \
    find b -name "*.onnx" -exec cp {} /comfyui/models/insightface/models/buffalo_l/ \; && \
    rm -rf buffalo_l.zip b

# ============================================================
# Boot ComfyUI (CPU) and verify EVERY workflow custom-node registers.
# Fails the build with the missing list instead of finding it in prod.
# ============================================================
RUN <<'EOF'
set -e
cd /comfyui
python main.py --cpu > /tmp/boot.log 2>&1 &
SRV=$!
python3 -c '
import time, urllib.request, json, sys
o = None
for _ in range(90):
    try:
        o = json.load(urllib.request.urlopen("http://127.0.0.1:8188/object_info", timeout=5)); break
    except Exception: time.sleep(2)
if o is None: print("ComfyUI did not start"); sys.exit(1)
need = ["ReActorFaceSwap", "FaceDetailer", "UltralyticsDetectorProvider", "Power Lora Loader (rgthree)"]
miss = [n for n in need if n not in o]
try: ct = o["CLIPLoader"]["input"]["required"]["type"][0]
except Exception: ct = []
if "krea2" not in ct: miss.append("CLIPLoader.krea2 (only %d types)" % len(ct))
print("NODE CHECK:", "ALL OK" if not miss else "MISSING: " + str(miss))
sys.exit(1 if miss else 0)
' || { echo "=== BOOT LOG (tail) ==="; tail -80 /tmp/boot.log; kill $SRV 2>/dev/null || true; exit 1; }
kill $SRV 2>/dev/null || true
EOF

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
    test -s /comfyui/models/ultralytics/bbox/erax_nsfw_yolo11m.pt && echo "EraX NSFW OK" && \
\
    test -s /comfyui/models/facerestore_models/codeformer-v0.1.0.pth && echo "CodeFormer OK" && \
    test -s /comfyui/models/sams/sam_vit_l_0b3195.pth && echo "SAM vit_l OK" && \
    test -s /comfyui/models/hyperswap/hyperswap_1a_256.onnx && echo "HyperSwap OK" && \
    test -s /comfyui/models/reswapper/reswapper_256.onnx && echo "ReSwapper OK" && \
    test -s /comfyui/models/ultralytics/bbox/hand_yolov8s.pt && echo "hand YOLO OK (implicit via Subpack)" && \
    test -s /comfyui/models/sams/sam_vit_b_01ec64.pth && echo "SAM vit_b OK (implicit via Impact)" && \
    echo "=== ALL OK ==="
