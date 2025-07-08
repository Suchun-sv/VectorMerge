from dataclasses import dataclass

@dataclass
class EmbeddingConfig:
    model: str = "mistral"
    dataset: str = "scifact"
    type_: str = "corpus"

class 