import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

MODEL_ROOT  = Path("/runpod-volume/models")
LTX_FP8_REPO = "Lightricks/LTX-2.3-fp8"
GEMMA_REPO   = "google/gemma-3-12b-it-qat-q4_0-unquantized"

ASSETS = [
    (LTX_FP8_REPO, "ltx-2.3-22b-distilled-fp8.safetensors"),
    ("Lightricks/LTX-2.3", "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"),
]

GEMMA_DIR = MODEL_ROOT / "gemma-3-12b-it-qat-q4_0-unquantized"

def get_token():
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        print("ERROR: HF_TOKEN not set.", file=sys.stderr)
        sys.exit(1)
    return token

def download_asset(repo_id, filename, token):
    dest = MODEL_ROOT / filename
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"✅ Already cached: {filename}")
        return dest
    print(f"⬇️  Downloading {filename} from {repo_id} ...")
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(MODEL_ROOT), token=token)
    print(f"✅ Done: {filename}")
    return Path(path)

def download_gemma(token):
    if GEMMA_DIR.exists() and any(GEMMA_DIR.iterdir()):
        print("✅ Already cached: gemma encoder")
        return GEMMA_DIR
    print("⬇️  Downloading Gemma encoder ...")
    GEMMA_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=GEMMA_REPO, local_dir=str(GEMMA_DIR), token=token)
    print("✅ Done: Gemma encoder")
    return GEMMA_DIR

def verify_assets():
    missing = []
    for _, filename in ASSETS:
        path = MODEL_ROOT / filename
        if not path.exists() or path.stat().st_size < 1_000_000:
            missing.append(filename)
    if not GEMMA_DIR.exists() or not any(GEMMA_DIR.iterdir()):
        missing.append("gemma-3-12b-it-qat-q4_0-unquantized/")
    if missing:
        print("ERROR: Missing assets:", missing, file=sys.stderr)
        sys.exit(1)
    print("✅ All assets verified.")

if __name__ == "__main__":
    token = get_token()
    for repo_id, filename in ASSETS:
        download_asset(repo_id, filename, token)
    download_gemma(token)
    verify_assets()
    print("✅ Bootstrap complete. Worker is ready.")
