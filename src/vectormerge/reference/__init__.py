from .base_split import BaseSplit as RandomSplit, get_reference
from .la2m_split import LA2MSplit
SUPPORTED_REFERENCE_STRATEGIES = ["random", "la2m"]

__all__ = ["RandomSplit", "get_reference", "LA2MSplit", "SUPPORTED_REFERENCE_STRATEGIES"]