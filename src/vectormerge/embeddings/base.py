from dataclasses import dataclass


@dataclass
class MistralConfig:
    model_name: str = "mistral-embed"
    batch_size: int = 32
    device: str = "cpu"
    max_tokens: int = 8192
    sleep_time: int = 2

@dataclass
class OpenaiConfig:
    model_name: str = "text-embedding-3-small"
    batch_size: int = 16
    device: str = "cpu"
    max_tokens: int = 8192
    sleep_time: int = 2

@dataclass
class NvEmbedConfig:
    model_name: str = "nv-embed"
    batch_size: int = 32
    device: str = "cuda"
    model_path: str = "~/.cache/embedding_models/nv-embed/"
    max_tokens: int = 8192

@dataclass
class GTEConfig:
    model_name: str = "Alibaba-NLP/gte-Qwen2-7B-instruct"
    batch_size: int = 32
    device: str = "cuda"
    model_path: str = "~/.cache/embedding_models/gte/"
    max_tokens: int = 8192

@dataclass
class FastTextConfig:
    model_name: str = "fast-text"
    batch_size: int = 64
    device: str = "cpu"
    model_path: str = "~/.cache/embedding_models/fast-text/"

@dataclass
class Word2VecConfig:
    model_name: str = "word2vec"
    batch_size: int = 64
    device: str = "cpu"
    model_path: str = "~/.cache/embedding_models/word2vec/"

@dataclass
class GloveConfig:
    model_name: str = "glove"
    batch_size: int = 64
    device: str = "cpu"
    model_path: str = "~/.cache/embedding_models/glove.6B/"

@dataclass
class EmbeddingModelConfig:
    model: str = "mistral"
    dataset: str = "scifact"
    type_: str = "corpus"
    mistral_config: MistralConfig = MistralConfig()
    openai_config: OpenaiConfig = OpenaiConfig()
    nv_embed_config: NvEmbedConfig = NvEmbedConfig()
    gte_config: GTEConfig = GTEConfig()
    fast_text_config: FastTextConfig = FastTextConfig()
    word2vec_config: Word2VecConfig = Word2VecConfig()
    glove_config: GloveConfig = GloveConfig()