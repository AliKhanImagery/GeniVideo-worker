import os
import sys
import uuid
import base64
import tempfile
from pathlib import Path

import runpod
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT, AUDIO_SAMPLE_RATE

MODEL_ROOT  = Path("/runpod-volume/models")
CHECKPOINT  = MODEL_ROOT / "ltx-2.3-22b-distilled-fp8.safetensors"
UPSAMPLER   = MODEL_ROOT / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
GEMMA_ROOT  = MODEL_ROOT / "gemma-3-12b-it-qat-q4_0-unquantized"

def validate_assets():
    missing = [str(p) for p in [CHECKPOINT, UPSAMPLER, GEMMA_ROOT] if not p.exists()]
    if missing:
        print("FATAL: Missing model assets:", missing, file=sys.stderr)
        sys.exit(1)

validate_assets()

print("Loading LTX Distilled pipeline (FP8)...")
pipeline = DistilledPipeline(
    checkpoint_path=str(CHECKPOINT),
    spatial_upsampler_path=str(UPSAMPLER),
    gemma_root=str(GEMMA_ROOT),
    fp8transformer=True,
)
print("Pipeline ready.")

def handler(job):
    job_input = job["input"]
    prompt         = job_input.get("prompt", "")
    negative_prompt = job_input.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)
    height         = int(job_input.get("height", 512))
    width          = int(job_input.get("width", 768))
    num_frames     = int(job_input.get("num_frames", 49))
    num_steps      = int(job_input.get("num_inference_steps", 8))
    frame_rate     = int(job_input.get("frame_rate", 24))
    seed           = int(job_input.get("seed", 42))
    modality_scale = float(job_input.get("modality_scale", 3.0))

    if not prompt:
        return {"error": "prompt is required"}

    video, audio = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_steps,
        frame_rate=frame_rate,
        seed=seed,
        modality_scale=modality_scale,
    )

    filename = f"{uuid.uuid4()}.mp4"
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    encode_video(video, audio, tmp_path, fps=frame_rate, audio_sample_rate=AUDIO_SAMPLE_RATE)

    with open(tmp_path, "rb") as f:
        video_b64 = base64.b64encode(f.read()).decode("utf-8")

    os.unlink(tmp_path)
    return {"video_base64": video_b64, "filename": filename}

runpod.serverless.start({"handler": handler})
