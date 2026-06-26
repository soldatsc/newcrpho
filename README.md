# newcrpho — Krea 2 + ReActor RunPod Serverless Worker

ComfyUI worker for RunPod Serverless: **Krea 2 Turbo** generation + **ReActor** face swap (NSFW filter patched out) + **FaceDetailer**. Built on `runpod/worker-comfyui:5.7.1-base`, mirrors the `worker_swap` pattern.

**Image:** `soldatsc/worker-krea:latest`

## Required GitHub Secrets
`Settings → Secrets and variables → Actions`
- `DOCKERHUB_USERNAME` — Docker Hub login
- `DOCKERHUB_TOKEN` — Docker Hub access token (Account → Security → New Access Token)
- `CIVITAI_TOKEN` — Civitai API key (Account → API Keys) — used to download LoRAs at build time

## Baked models
- **Krea 2:** `krea2_turbo_fp8_scaled`, `qwen3vl_4b_fp8_scaled`, `qwen_image_vae`
- **LoRA:** `realism_engine_krea2_v2`, `MysticXXX_KREA2_v1` (from Civitai)
- **Swap:** `inswapper_128`, `GPEN-BFR-512`, `face_yolov8m`, `buffalo_l`

## Custom nodes
`rgthree-comfy`, `comfyui-reactor` (NSFW-patched), `comfyui-impact-pack`, `comfyui-impact-subpack`, `comfyui-tooling-nodes`

## Backend / API template notes
- Send the workflow as **API format**.
- Face swap by gender: set `detect_gender_input: "female"` on the ReActor node.
- Set `width`/`height` directly on `EmptyLatentImage` (FluxResolutionNode is NOT installed).
- LoRA filename in the template: `realism_engine_krea2_v2.safetensors`.

## Deploy
Push to `main` → GitHub Action builds and pushes the image to Docker Hub.
Point a RunPod Serverless endpoint at `soldatsc/worker-krea:latest` (24 GB GPU).
