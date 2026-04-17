"""Pydantic settings models for ml-dev-rules migration."""

from typing import Any, Dict

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SettingsSection(BaseModel):
    """Generic section model used as migration bridge."""

    model_config = ConfigDict(extra="allow")


class Settings(BaseSettings):
    """Root settings model loaded from settings.yaml + env + CLI overrides."""

    model_config = SettingsConfigDict(
        env_prefix="VECTORMERGE_",
        env_nested_delimiter="__",
        extra="allow",
    )

    data_path: str = "./data/raw/beir/"
    embedding_path: str = "./data/processed/embeddings/"
    embedding_cache_path: str = "./data/processed/embeddings/.cache/"
    reference_path: str = "./data/processed/references/"
    mapping_path: str = "./data/processed/mappings/"
    mapping_param_path: str = "./output/mapping_models/"
    mapping_embedding_path: str = "./output/mapping_embeddings/"
    cluster_path: str = "./data/processed/clusters/"

    wandb_project: str = "vector-merge"
    verbose: bool = False

    mapping_config: Dict[str, Any] = Field(default_factory=dict)
    clustering_config: Dict[str, Any] = Field(default_factory=dict)
    embedding_models: Dict[str, Any] = Field(default_factory=dict)
