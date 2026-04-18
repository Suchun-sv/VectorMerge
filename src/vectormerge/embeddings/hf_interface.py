from huggingface_hub import HfApi, hf_hub_download
import numpy as np
from pathlib import Path
from typing import Optional
import os
import dotenv
import shutil

REPO_ENTITY = os.getenv("VM_HF_REPO_ENTITY", "DB-Edinburgh")
REPO_NAME = os.getenv("VM_HF_REPO_NAME", "VectorBenchmark")


def _get_token(required: bool = True) -> Optional[str]:
    """Load Hugging Face token from environment or .env file.

    Public dataset downloads can work without auth; uploads require auth.
    """
    dotenv.load_dotenv()
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if required and not token:
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
    token = _get_token(required=True)
    api = HfApi(token=token)
    path_in_repo = f"embeddings/{upload_file_name}"
    api.upload_file(
        path_or_fileobj=str(local_file),
        path_in_repo=path_in_repo,
        repo_id=f"{repo_entity}/{repo_name}",
        repo_type="dataset",
        commit_message=commit_message or f"Upload embedding: {upload_file_name}",
    )
    print(
        f"✅ Uploaded '{local_file.name}' → '{path_in_repo}' in {repo_entity}/{repo_name}"
    )


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
    Uses HF token if available; falls back to anonymous download for public repos.
    """
    token = _get_token(required=False)
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
    source_path = Path(local_path)
    destination_path = target_dir / download_file_name
    if source_path.resolve() != destination_path.resolve():
        shutil.move(str(source_path), str(destination_path))
        if source_path.parent.exists() and not os.listdir(source_path.parent):
            shutil.rmtree(source_path.parent)
    print(f"✅ Downloaded: {destination_path}")
    return destination_path
