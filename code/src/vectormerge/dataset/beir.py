"""BEIR dataset loader for VectorMerge."""

from typing import Union, List, Optional
from pathlib import Path
import os

from beir.datasets.data_loader import GenericDataLoader
from beir import util as beir_util
from loguru import logger

from .base import Dataset, DatasetLoader


class BEIRDatasetLoader(DatasetLoader):
    """Loader for BEIR (Benchmarking IR) datasets."""

    SUPPORTED_DATASETS = {
        "scifact": "test",
        "nfcorpus": "test",
        "nq": "test",
        "cqadupstack": "test",
        "arguana": "test",
        "scidocs": "test",
        "fiqa": "test",
        "signal1m": "test",
        "fever": "test",
        "hotpotqa": "test",
        "msmarco": "test",
        "trec-covid": "test",
        "bioasq": "test",
        "touche-2020": "test",
        "dbpedia-entity": "test",
        "webis-touche2020": "test",
        "climate-fever": "test",
        "robust04": "test",
    }

    def __init__(self, data_path: Union[str, Path]):
        """Initialize BEIR dataset loader."""
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.base_url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"

        logger.info(f"BEIRDatasetLoader initialized with data_path: {self.data_path}")

    def load(self, dataset_name: str, split: str = "test", **kwargs) -> Dataset:
        """Load a BEIR dataset."""
        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(
                f"Dataset '{dataset_name}' not supported. "
                f"Available BEIR datasets: {list(self.SUPPORTED_DATASETS.keys())}"
            )

        logger.info(f"Loading BEIR dataset: {dataset_name} (split: {split})")

        dataset_path = self._prepare_dataset(dataset_name)

        try:
            data_loader = GenericDataLoader(data_folder=str(dataset_path))
            corpus, queries, qrels = data_loader.load(split=split)
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            raise

        dataset = Dataset(
            name=dataset_name,
            corpus=corpus,
            queries=queries,
            qrels=qrels,
            split=split,
            source="beir",
            metadata={
                "dataset_path": str(dataset_path),
                "base_url": self.base_url,
                "loader": "BEIRDatasetLoader",
                "beir_version": "latest",
            }
        )

        stats = dataset.get_stats()
        logger.info(f"BEIR dataset loaded successfully: {stats}")

        return dataset

    def _prepare_dataset(self, dataset_name: str) -> Path:
        """Download and prepare BEIR dataset if needed."""
        dataset_path = self.data_path / dataset_name

        if not dataset_path.exists():
            logger.info(f"Downloading BEIR dataset: {dataset_name}")
            try:
                url = f"{self.base_url}/{dataset_name}.zip"
                beir_util.download_and_unzip(url, str(self.data_path))
                logger.info(f"Successfully downloaded {dataset_name}")
            except Exception as e:
                logger.error(f"Failed to download dataset {dataset_name}: {e}")
                raise

        final_path = self._handle_special_cases(dataset_name, dataset_path)

        if not final_path.exists():
            raise ValueError(f"Dataset path does not exist after download: {final_path}")

        return final_path

    def _handle_special_cases(self, dataset_name: str, dataset_path: Path) -> Path:
        """Handle special dataset directory structures."""
        if dataset_name == "cqadupstack":
            english_path = dataset_path / "english"
            if english_path.exists():
                return english_path

        return dataset_path

    def list_available_datasets(self) -> List[str]:
        """List available BEIR datasets."""
        return list(self.SUPPORTED_DATASETS.keys())

    def get_dataset_info(self, dataset_name: str) -> dict:
        """Get information about a specific BEIR dataset."""
        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(f"Dataset '{dataset_name}' not supported")

        dataset_info = {
            "scifact": {
                "description": "Scientific claim verification",
                "domain": "Scientific",
                "size": "small",
                "queries": "~300",
                "corpus": "~5K"
            },
            "nfcorpus": {
                "description": "Nutrition facts corpus",
                "domain": "Medical/Nutrition",
                "size": "small",
                "queries": "~3K",
                "corpus": "~3K"
            },
            "nq": {
                "description": "Natural Questions",
                "domain": "General Knowledge",
                "size": "large",
                "queries": "~3K",
                "corpus": "~2.6M"
            },
            "cqadupstack": {
                "description": "Community question answering",
                "domain": "Technical/Programming",
                "size": "medium",
                "queries": "~13K",
                "corpus": "~457K"
            },
            "arguana": {
                "description": "Argument mining",
                "domain": "Argumentative",
                "size": "small",
                "queries": "~1K",
                "corpus": "~8K"
            },
            "scidocs": {
                "description": "Scientific document classification",
                "domain": "Scientific",
                "size": "medium",
                "queries": "~1K",
                "corpus": "~25K"
            },
            "fiqa": {
                "description": "Financial question answering",
                "domain": "Financial",
                "size": "small",
                "queries": "~648",
                "corpus": "~57K"
            },
            "fever": {
                "description": "Fact extraction and verification",
                "domain": "General Knowledge",
                "size": "large",
                "queries": "~6K",
                "corpus": "~5.4M"
            },
        }

        return dataset_info.get(dataset_name, {
            "description": "BEIR dataset",
            "domain": "Unknown",
            "size": "unknown",
            "queries": "unknown",
            "corpus": "unknown"
        })

    def download_all(self, datasets: Optional[List[str]] = None) -> None:
        """Download multiple BEIR datasets."""
        if datasets is None:
            datasets = self.list_available_datasets()

        logger.info(f"Downloading {len(datasets)} BEIR datasets: {datasets}")

        for dataset_name in datasets:
            try:
                self._prepare_dataset(dataset_name)
                logger.info(f"✓ Downloaded: {dataset_name}")
            except Exception as e:
                logger.error(f"✗ Failed to download {dataset_name}: {e}")
                continue

        logger.info("Batch download completed")


def load_beir_dataset(dataset_name: str, split: str = "test",
                     data_path: Union[str, Path] = "./data/raw/beir/") -> Dataset:
    """Convenience function to load a BEIR dataset."""
    loader = BEIRDatasetLoader(data_path)
    return loader.load(dataset_name, split)