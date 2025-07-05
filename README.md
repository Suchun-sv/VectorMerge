# VectorMerge 🚀

[![PyPI version](https://badge.fury.io/py/vectormerge.svg)](https://badge.fury.io/py/vectormerge)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**VectorMerge** is a modern, extensible Python library for embedding evaluation and vector space mapping. It provides sophisticated tools for comparing, mapping, and evaluating embeddings from different models with state-of-the-art techniques.

## ✨ Features

- **🔬 Advanced Vector Space Mapping**: Procrustes analysis, non-linear mapping, and novel clustering-based approaches
- **🎯 Comprehensive Evaluation**: Support for BEIR datasets with multiple metrics (Recall, NDCG, MAP)
- **🚀 Multi-Model Support**: BERT, RoBERTa, BGE, OpenAI, Mistral, FastText, Word2Vec, GloVe, and more
- **⚡ Efficient Implementation**: Optimized with FAISS for fast similarity search and batched processing
- **🔧 Highly Configurable**: YAML-based configuration with sensible defaults
- **📊 Rich Evaluation Metrics**: Extended recall, rank recall, soundness, and completeness measures
- **🎨 Modern CLI Interface**: Beautiful command-line interface with progress bars and rich output
- **🔍 Extensible Architecture**: Easy to add new models, metrics, and mapping techniques

## 📦 Installation

### Basic Installation

```bash
pip install vectormerge
```

### Development Installation

```bash
git clone https://github.com/your-username/vectormerge.git
cd vectormerge
pip install -e .
```

## 🚀 Quick Start

### Command Line Interface

```bash
# Basic usage with default settings
vectormerge evaluate \
    --model-1 bert-base-uncased \
    --model-2 roberta-base \
    --dataset scifact

# Advanced usage with custom configuration
vectormerge evaluate \
    --config configs/advanced_config.yaml \
    --output results/experiment_1

# List available models and datasets
vectormerge list-models
vectormerge list-datasets

# Generate configuration file
vectormerge generate-config --interactive
```

### Python API

```python
import vectormerge as vm

# Create configuration
config = vm.VectorMergeConfig(
    model_1=vm.ModelConfig(name="bert-base-uncased"),
    model_2=vm.ModelConfig(name="roberta-base"),
    dataset=vm.DatasetConfig(name="scifact"),
    mapping=vm.MappingConfig(method="procrustes"),
    evaluation=vm.EvaluationConfig(k_list=[10, 100, 1000])
)

# Run evaluation
evaluator = vm.EmbeddingEvaluator(config)
results = evaluator.run()

print(f"Recall@10: {results.recall[10]:.4f}")
print(f"NDCG@100: {results.ndcg[100]:.4f}")
```

### Advanced Usage

```python
# Custom vector space mapping
mapper = vm.VectorSpaceMapper(
    strategy=vm.ProcrustesMappingStrategy(
        config=vm.MappingConfig(num_clusters=50)
    )
)

# Evaluate with custom metrics
calculator = vm.MetricsCalculator(
    config=vm.EvaluationConfig(
        metrics=["recall", "ndcg", "map", "extended_recall"]
    )
)

# Generate embeddings from multiple models
models = ["bert-base-uncased", "roberta-base", "bge"]
embeddings = {}
for model_name in models:
    generator = vm.get_embedding_generator(model_name)
    embeddings[model_name] = generator.generate_embeddings(texts)
```

## 📊 Supported Models

### Transformer Models
- **BERT**: bert-base-uncased, bert-large-uncased
- **RoBERTa**: roberta-base, roberta-large
- **BGE**: BAAI/bge-base-en, BAAI/bge-large-en
- **NV-Embed**: nvidia/NV-Embed-v2
- **GTE**: Alibaba-NLP/gte-Qwen2-7B-instruct

### Traditional Models
- **FastText**: Pre-trained and custom models
- **Word2Vec**: Google News vectors and custom models
- **GloVe**: Multiple dimensions (50, 100, 200, 300)

### API-Based Models
- **OpenAI**: text-embedding-3-small, text-embedding-3-large
- **Mistral**: mistral-embed

## 🎯 Supported Datasets

VectorMerge supports all major BEIR datasets:

- **SciFact**: Scientific claim verification
- **NFCorpus**: Nutrition facts corpus
- **Natural Questions**: Real questions from Google search
- **CQADupStack**: Community question answering
- **ArguAna**: Argument mining
- **SciDocs**: Scientific document classification
- **FiQA**: Financial question answering

## 🔧 Configuration

Create a YAML configuration file for complex experiments:

```yaml
# config.yaml
model_1:
  name: "bert-base-uncased"
  batch_size: 32
  device: "cuda"

model_2:
  name: "roberta-base"
  batch_size: 32
  device: "cuda"

dataset:
  name: "scifact"
  split: "test"

mapping:
  method: "procrustes"
  num_clusters: 50
  reference_creation_method: "hierarchical_kmeans"
  d0_ratio: 0.33

evaluation:
  k_list: [10, 100, 1000]
  metrics: ["recall", "ndcg", "map", "extended_recall"]
  cal_self_metric: true

logging:
  level: "INFO"
  use_wandb: true
  wandb_project: "vectormerge-experiments"
```

## 📈 Evaluation Metrics

VectorMerge provides comprehensive evaluation metrics:

- **Standard Metrics**: Recall@k, NDCG@k, MAP@k, Precision@k
- **Extended Metrics**: Extended Recall (novel approach for cross-model evaluation)
- **Rank Metrics**: Rank Recall for position-aware evaluation
- **Geometric Metrics**: Soundness and Completeness for mapping quality
- **Distance Metrics**: Maximum distances for similarity threshold analysis

## 🛠️ Advanced Features

### Vector Space Mapping Techniques

1. **Procrustes Analysis**: Orthogonal transformation between vector spaces
2. **Non-linear Mapping**: Neural network-based mapping for complex transformations
3. **Clustering-based Mapping**: Novel approach using hierarchical clustering and reference points

### Reference Creation Methods

1. **Hierarchical K-means**: Multi-level clustering for robust reference selection
2. **Farthest Point Sampling**: Diverse reference point selection
3. **Random Sampling**: Baseline reference selection method

### Evaluation Strategies

1. **Standard Evaluation**: Traditional IR metrics on individual models
2. **Cross-Model Evaluation**: Compare embeddings across different models
3. **Mapped Evaluation**: Evaluate quality of vector space mappings
4. **Extended Evaluation**: Novel metrics for comprehensive assessment

## 🎨 CLI Examples

```bash
# Basic evaluation
vectormerge evaluate --model-1 bert-base-uncased --model-2 roberta-base --dataset scifact

# List available models and datasets
vectormerge list-models
vectormerge list-datasets

# Generate embeddings only
vectormerge generate-embeddings --model bert-base-uncased --dataset scifact --output embeddings/

# Run mapping experiment
vectormerge map --source-model bert-base-uncased --target-model roberta-base --method procrustes

# System diagnostics
vectormerge doctor
```

## 🔬 Research Applications

VectorMerge is designed for research in:

- **Cross-Model Embedding Analysis**: Understanding relationships between different embedding models
- **Vector Space Geometry**: Analyzing geometric properties of embedding spaces
- **Information Retrieval**: Evaluating embedding quality on retrieval tasks
- **Model Comparison**: Systematic comparison of embedding models
- **Transfer Learning**: Understanding how embeddings transfer across domains

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/your-username/vectormerge.git
cd vectormerge

# Install development dependencies
pip install -e .

# Run tests
pytest

# Format code
black src/ tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on top of the excellent [BEIR](https://github.com/beir-cellar/beir) framework
- Inspired by research in vector space alignment and embedding evaluation
- Uses [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search
- Leverages [Hugging Face](https://huggingface.co/) transformers ecosystem

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-username/vectormerge/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/vectormerge/discussions)

## 🔗 Citation

If you use VectorMerge in your research, please cite:

```bibtex
@software{vectormerge,
  title={VectorMerge: A Modern Library for Embedding Evaluation and Vector Space Mapping},
  author={Your Name},
  year={2025},
  url={https://github.com/your-username/vectormerge}
}
```

---

<div align="center">
  <strong>Made with ❤️ for the research community</strong>
</div>
