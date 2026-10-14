"""CCA (Canonical Correlation Analysis) mapping strategy for VectorMerge."""

import numpy as np
from typing import Optional, Dict, Any, Union
from pathlib import Path
from loguru import logger
from sklearn.cross_decomposition import CCA
import joblib

from ..base import MappingStrategy, MappingConfig


class CCAMappingStrategy(MappingStrategy):
    """CCA (Canonical Correlation Analysis) mapping strategy."""

    def __init__(self, config: MappingConfig):
        """Initialize CCA mapping strategy."""
        super().__init__(config)
        self.cca_model: Optional[CCA] = None
        self.source_mean: Optional[np.ndarray] = None
        self.target_mean: Optional[np.ndarray] = None

    def _fit(self, source_embeddings: np.ndarray, target_embeddings: np.ndarray,
             reference_indices: np.ndarray, **kwargs) -> None:
        """Fit CCA mapping using reference embeddings."""
        logger.info(f"Fitting CCA mapping with {len(reference_indices)} reference points")

        X_ref = source_embeddings[reference_indices]
        Y_ref = target_embeddings[reference_indices]

        self.source_mean = np.mean(X_ref, axis=0)
        self.target_mean = np.mean(Y_ref, axis=0)

        self.cca_model = CCA(
            n_components=self.config.cca_config.n_components,
            scale=self.config.cca_config.scale,
            max_iter=self.config.cca_config.max_iter,
            tol=self.config.cca_config.tol,
            copy=self.config.cca_config.copy
        )

        self.cca_model.fit(X_ref, Y_ref)

        self.metadata = {
            'n_components': self.config.cca_config.n_components,
            'scale': self.config.cca_config.scale,
            'max_iter': self.config.cca_config.max_iter,
            'tol': self.config.cca_config.tol,
            'reference_size': len(reference_indices),
            'source_dimension': X_ref.shape[1],
            'target_dimension': Y_ref.shape[1]
        }

        logger.info("CCA mapping fitting completed")

    def _transform(self, embeddings: np.ndarray, **kwargs) -> np.ndarray:
        """Transform embeddings using fitted CCA mapping."""
        if not self.is_fitted:
            raise ValueError("CCA mapping must be fitted before transformation")

        if self.cca_model is None:
            raise ValueError("CCA model not fitted")

        X_cca, _ = self.cca_model.transform(embeddings, np.zeros_like(embeddings))

        transformed = X_cca @ self.cca_model.y_weights_.T

        if self.target_mean is not None:
            transformed = transformed + self.target_mean

        return transformed

    def save(self, path: Union[str, Path]) -> None:
        """Save CCA mapping parameters."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self.cca_model is not None:
            joblib.dump(self.cca_model, path / "cca_model.pkl")

        params = {
            'source_mean': self.source_mean,
            'target_mean': self.target_mean,
            'is_fitted': self.is_fitted,
            'metadata': self.metadata
        }

        joblib.dump(params, path / "cca_params.pkl")
        logger.info(f"CCA mapping saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'CCAMappingStrategy':
        """Load CCA mapping parameters."""
        path = Path(path)

        cca_model = joblib.load(path / "cca_model.pkl")

        params = joblib.load(path / "cca_params.pkl")

        from ..base import MappingConfig
        instance = cls(MappingConfig())

        instance.cca_model = cca_model
        instance.source_mean = params['source_mean']
        instance.target_mean = params['target_mean']
        instance.is_fitted = params['is_fitted']
        instance.metadata = params['metadata']

        logger.info(f"CCA mapping loaded from {path}")
        return instance

    def check_fit(self, path: Union[str, Path]) -> bool:
        """Check if CCA mapping is fitted and saved."""
        path = Path(path)
        return (path / "cca_model.pkl").exists() and (path / "cca_params.pkl").exists()
