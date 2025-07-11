from huggingface_hub import HfApi, hf_hub_download
import numpy as np
from pathlib import Path
from typing import Optional
import os
import dotenv

REPO_ENTITY = "suchun"
REPO_NAME = "VectorMerge"  # your dataset repo name

def _get_token() -> str:
    """Load Hugging Face token from environment or .env file."""
    dotenv.load_dotenv()
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN or HUGGINGFACE_TOKEN not found in environment or .env"
        )
    return token

def upload_embedding(
    local_file: Path,
    upload_file_name: str,
    repo_entity: str = REPO_ENTITY,
    repo_name: str = REPO_NAME,
    commit_message: Optional[str] = None,
) -> None:
    """
    Upload a local .npy file to Hugging Face under `embeddings/<upload_file_name>`.
    """
    token = _get_token()
    api = HfApi(token=token)
    path_in_repo = f"embeddings/{upload_file_name}"
    api.upload_file(
        path_or_fileobj=str(local_file),
        path_in_repo=path_in_repo,
        repo_id=f"{repo_entity}/{repo_name}",
        repo_type="dataset",
        commit_message=commit_message or f"Upload embedding: {upload_file_name}",
    )
    print(f"✅ Uploaded '{local_file.name}' → '{path_in_repo}' in {repo_entity}/{repo_name}")

def download_embedding(
    download_file_name: str,
    target_dir: Path,
    repo_entity: str = REPO_ENTITY,
    repo_name: str = REPO_NAME,
    revision: Optional[str] = None,
) -> Path:
    """
    Download a specific embedding file from HF under `embeddings/<filename>`,
    caching locally in target_dir. Returns the local file path.
    For now, only support use of HF_TOKEN in .env file.
    """
    token = _get_token()
    target_dir.mkdir(parents=True, exist_ok=True)
    local_path = hf_hub_download(
        repo_id=f"{repo_entity}/{repo_name}",
        filename=f"embeddings/{download_file_name}",
        repo_type="dataset",
        cache_dir=str(target_dir),
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        revision=revision,
        token=token,
    )
    print(f"✅ Downloaded: {local_path}")
    return Path(local_path)