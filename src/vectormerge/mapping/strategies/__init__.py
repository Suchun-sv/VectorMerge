"""
Mapping strategies for VectorMerge.

This package contains different strategies for mapping between embedding spaces.
"""

from .procrustes import ProcrustesMappingStrategy
from .nonlinear import NonLinearMappingStrategy
from .la2m import LA2MStrategy
from .cca import CCAMappingStrategy
from .gromov_wasserstein import GromovWassersteinMappingStrategy

__all__ = [
    "ProcrustesMappingStrategy",
    "NonLinearMappingStrategy",
    "LA2MStrategy",
    "CCAMappingStrategy",
    "GromovWassersteinMappingStrategy",
] 