"""Custom dataset loader for VectorMerge."""

from typing import Union, List, Optional, Dict, Any
from pathlib import Path
import json
import os

from loguru import logger

from .base import Dataset, DatasetLoader


class CustomDatasetLoader(DatasetLoader):
    """Loader for custom dataset formats."""

    def __init__(self, data_path: Union[str, Path]):
        """Initialize custom dataset loader."""
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"CustomDatasetLoader initialized with data_path: {self.data_path}")

    def load(self, dataset_name: str, split: str = "test", **kwargs) -> Dataset:
        """Load a custom dataset."""
        dataset_format = kwargs.get('format', 'json')

        if dataset_format == 'json':
            return self._load_json_dataset(dataset_name, split, **kwargs)
        elif dataset_format == 'tsv':
            return self._load_tsv_dataset(dataset_name, split, **kwargs)
        else:
            raise ValueError(f"Unsupported custom dataset format: {dataset_format}")

    def _load_json_dataset(self, dataset_name: str, split: str = "test", **kwargs) -> Dataset:
        """Load a dataset from JSON format."""
        dataset_path = self.data_path / dataset_name
        json_file = dataset_path / f"{split}.json"

        if not json_file.exists():
            raise FileNotFoundError(f"Custom dataset file not found: {json_file}")

        logger.info(f"Loading custom JSON dataset: {dataset_name} from {json_file}")

        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            corpus = data.get('corpus', {})
            queries = data.get('queries', {})
            qrels = data.get('qrels', {})

            dataset = Dataset(
                name=dataset_name,
                corpus=corpus,
                queries=queries,
                qrels=qrels,
                split=split,
                source="custom",
                metadata={
                    "format": "json",
                    "file_path": str(json_file),
                    "loader": "CustomDatasetLoader",
                }
            )

            stats = dataset.get_stats()
            logger.info(f"Custom JSON dataset loaded: {stats}")

            return dataset

        except Exception as e:
            logger.error(f"Failed to load custom JSON dataset {dataset_name}: {e}")
            raise

    def _load_tsv_dataset(self, dataset_name: str, split: str = "test", **kwargs) -> Dataset:
        """Load a dataset from TSV format."""
        dataset_path = self.data_path / dataset_name

        corpus_file = dataset_path / f"corpus_{split}.tsv"
        queries_file = dataset_path / f"queries_{split}.tsv"
        qrels_file = dataset_path / f"qrels_{split}.tsv"

        for file_path in [corpus_file, queries_file, qrels_file]:
            if not file_path.exists():
                raise FileNotFoundError(f"Custom dataset file not found: {file_path}")

        logger.info(f"Loading custom TSV dataset: {dataset_name} from {dataset_path}")

        try:
            corpus = {}
            with open(corpus_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        doc_id, title, text = parts[0], parts[1], '\t'.join(parts[2:])
                        corpus[doc_id] = {"title": title, "text": text}

            queries = {}
            with open(queries_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        query_id, query_text = parts[0], '\t'.join(parts[1:])
                        queries[query_id] = query_text

            qrels = {}
            with open(qrels_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        query_id, doc_id, score = parts[0], parts[1], parts[2]
                        if query_id not in qrels:
                            qrels[query_id] = {}
                        qrels[query_id][doc_id] = int(float(score))

            dataset = Dataset(
                name=dataset_name,
                corpus=corpus,
                queries=queries,
                qrels=qrels,
                split=split,
                source="custom",
                metadata={
                    "format": "tsv",
                    "dataset_path": str(dataset_path),
                    "loader": "CustomDatasetLoader",
                }
            )

            stats = dataset.get_stats()
            logger.info(f"Custom TSV dataset loaded: {stats}")

            return dataset

        except Exception as e:
            logger.error(f"Failed to load custom TSV dataset {dataset_name}: {e}")
            raise

    def list_available_datasets(self) -> List[str]:
        """List available custom datasets."""
        if not self.data_path.exists():
            return []

        datasets = []
        for item in self.data_path.iterdir():
            if item.is_dir():
                has_json = any(f.suffix == '.json' for f in item.iterdir())
                has_tsv = any(f.name.startswith(('corpus_', 'queries_', 'qrels_')) for f in item.iterdir())

                if has_json or has_tsv:
                    datasets.append(item.name)

        return sorted(datasets)

    def create_dataset_template(self, dataset_name: str, format_type: str = "json") -> None:
        """Create a template for a new custom dataset."""
        dataset_path = self.data_path / dataset_name
        dataset_path.mkdir(parents=True, exist_ok=True)

        if format_type == "json":
            template = {
                "corpus": {
                    "doc_1": {"title": "Sample Document 1", "text": "This is the content of document 1."},
                    "doc_2": {"title": "Sample Document 2", "text": "This is the content of document 2."}
                },
                "queries": {
                    "query_1": "What is the content of the first document?",
                    "query_2": "Tell me about document 2"
                },
                "qrels": {
                    "query_1": {"doc_1": 1, "doc_2": 0},
                    "query_2": {"doc_1": 0, "doc_2": 1}
                }
            }

            template_file = dataset_path / "test.json"
            with open(template_file, 'w', encoding='utf-8') as f:
                json.dump(template, f, indent=2, ensure_ascii=False)

            logger.info(f"Created JSON template at: {template_file}")

        elif format_type == "tsv":
            corpus_file = dataset_path / "corpus_test.tsv"
            with open(corpus_file, 'w', encoding='utf-8') as f:
                f.write("doc_1\tSample Document 1\tThis is the content of document 1.\n")
                f.write("doc_2\tSample Document 2\tThis is the content of document 2.\n")

            queries_file = dataset_path / "queries_test.tsv"
            with open(queries_file, 'w', encoding='utf-8') as f:
                f.write("query_1\tWhat is the content of the first document?\n")
                f.write("query_2\tTell me about document 2\n")

            qrels_file = dataset_path / "qrels_test.tsv"
            with open(qrels_file, 'w', encoding='utf-8') as f:
                f.write("query_1\tdoc_1\t1\n")
                f.write("query_1\tdoc_2\t0\n")
                f.write("query_2\tdoc_1\t0\n")
                f.write("query_2\tdoc_2\t1\n")

            logger.info(f"Created TSV templates at: {dataset_path}")
        else:
            raise ValueError(f"Unsupported format type: {format_type}")


def load_custom_dataset(dataset_name: str, split: str = "test", format_type: str = "json",
                       data_path: Union[str, Path] = "./data/custom/") -> Dataset:
    """Convenience function to load a custom dataset."""
    loader = CustomDatasetLoader(data_path)
    return loader.load(dataset_name, split, format=format_type)