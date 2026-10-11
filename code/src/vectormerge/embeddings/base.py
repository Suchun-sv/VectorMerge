from dataclasses import dataclass
from typing import Dict, Any
import dataclasses
from dacite import from_dict

@dataclass
class MistralConfig:
    model_name: str = "mistral-embed"
    batch_size: int = 32
    device: str = "cpu"
    max_tokens: int = 8192
    sleep_time: int = 2

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MistralConfig":
        return from_dict(cls, data)

@dataclass
class OpenaiConfig:
    model_name: str = "text-embedding-3-small"
    batch_size: int = 16
    device: str = "cpu"
    max_tokens: int = 8192
    sleep_time: int = 2

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OpenaiConfig":
        return from_dict(cls, data)

@dataclass
class NvEmbedConfig:
    model_name: str = "nv-embed"
    batch_size: int = 32
    device: str = "cuda"
    model_path: str = "~/.cache/embedding_models/nv-embed/"
    max_tokens: int = 8192

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NvEmbedConfig":
        return from_dict(cls, data)

@dataclass
class GTEConfig:
    model_name: str = "Alibaba-NLP/gte-Qwen2-7B-instruct"
    batch_size: int = 32
    device: str = "cuda"
    model_path: str = "~/.cache/embedding_models/gte/"
    max_tokens: int = 8192

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GTEConfig":
        return from_dict(cls, data)

@dataclass
class FastTextConfig:
    model_name: str = "fast-text"
    batch_size: int = 64
    device: str = "cpu"
    model_path: str = "~/.cache/embedding_models/fast-text/"

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FastTextConfig":
        return from_dict(cls, data)

@dataclass
class Word2VecConfig:
    model_name: str = "word2vec"
    batch_size: int = 64
    device: str = "cpu"
    model_path: str = "~/.cache/embedding_models/word2vec/"

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Word2VecConfig":
        return from_dict(cls, data)

@dataclass
class GloveConfig:
    model_name: str = "glove"
    batch_size: int = 64
    device: str = "cpu"
    model_path: str = "~/.cache/embedding_models/glove.6B/"

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GloveConfig":
        return from_dict(cls, data)

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

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)
    
    def load_model_settings(self, model_name: str) -> Dict[str, Any]:
        if model_name == "mistral":
            return self.mistral_config.to_dict()
        elif model_name == "openai":
            return self.openai_config.to_dict()
        elif model_name == "nv-embed":
            return self.nv_embed_config.to_dict()
        elif model_name == "gte":
            return self.gte_config.to_dict()
        elif model_name == "fast-text":
            return self.fast_text_config.to_dict()
        elif model_name == "word2vec":
            return self.word2vec_config.to_dict()
        elif model_name == "glove":
            return self.glove_config.to_dict()
        else:
            raise ValueError(f"Unknown model name: {model_name}")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmbeddingModelConfig":
        return from_dict(cls, data)