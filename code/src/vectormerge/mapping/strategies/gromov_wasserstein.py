"""Gromov-Wasserstein mapping strategy for VectorMerge using POT library."""

import numpy as np
import torch
try:
    import ot
except ImportError:
    raise ImportError("POT library is required. Install with: pip install pot")

from typing import Optional, Union
from pathlib import Path
from loguru import logger
import psutil
from ..base import MappingStrategy, MappingConfig


def get_device() -> torch.device:
    """Get the best available device (CUDA if available, otherwise CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"Using CUDA device: {gpu_name} with {gpu_memory:.1f}GB memory")

        torch.cuda.empty_cache()
        return device
    else:
        logger.info("CUDA not available, using CPU")
        return torch.device("cpu")


def estimate_memory_usage(n1: int, n2: int, device: torch.device) -> float:
    """Estimate memory usage for Gromov-Wasserstein computation in GB."""
    total_elements = n1*n1 + n2*n2 + 12*n1*n2
    bytes_per_element = 4 if device.type == "cuda" else 8
    return total_elements * bytes_per_element / (1024**3)


def check_memory_feasibility(n1: int, n2: int, device: torch.device, max_memory_gb: Optional[float] = None) -> bool:
    """Check if computation is feasible given memory constraints."""
    if max_memory_gb is None:
        if device.type == "cuda":
            available_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            max_memory_limit = available_memory * 0.8
        else:
            available_memory = psutil.virtual_memory().available / (1024**3)
            max_memory_limit = available_memory * 0.8
    else:
        max_memory_limit = max_memory_gb

    required_memory = estimate_memory_usage(n1, n2, device)

    if device.type == "cuda":
        logger.info(f"GPU memory check: Required {required_memory:.2f}GB, Available {max_memory_limit:.2f}GB")
    else:
        logger.info(f"CPU memory check: Required {required_memory:.2f}GB, Available {max_memory_limit:.2f}GB")

    return required_memory <= max_memory_limit


def suggest_sample_size(n1: int, n2: int, device: torch.device, max_memory_gb: Optional[float] = None) -> int:
    """Suggest appropriate sample size for memory constraints."""
    if max_memory_gb is None:
        if device.type == "cuda":
            available_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            max_memory_limit = available_memory * 0.8
        else:
            available_memory = psutil.virtual_memory().available / (1024**3)
            max_memory_limit = available_memory * 0.8
    else:
        max_memory_limit = max_memory_gb

    low, high = 50, min(n1, n2)
    best_size = 50

    while low <= high:
        mid = (low + high) // 2
        if check_memory_feasibility(mid, mid, device, max_memory_gb):
            best_size = mid
            low = mid + 1
        else:
            high = mid - 1

    return max(50, best_size)


class GromovWassersteinMappingStrategy(MappingStrategy):
    """GPU/CPU compatible Gromov-Wasserstein alignment strategy using POT implementation."""

    def __init__(self, config: MappingConfig):
        """Initialize Gromov-Wasserstein mapping strategy."""
        super().__init__(config)

        self.device = get_device()
        self.dtype = torch.float32 if self.device.type == "cuda" else torch.float64

        self.transport_plan: Optional[torch.Tensor] = None
        self.source_embeddings: Optional[torch.Tensor] = None
        self.target_embeddings: Optional[torch.Tensor] = None
        self.source_indices: Optional[torch.Tensor] = None

        self.max_reference_points = getattr(config, 'max_reference_points', 1000)
        self.auto_sample = getattr(config, 'auto_sample', True)
        self.max_memory_gb = getattr(config, 'max_memory_gb', None)

        logger.info(f"Using dtype: {self.dtype}")
        logger.info(f"Max reference points: {self.max_reference_points}")
        logger.info(f"Auto sampling enabled: {self.auto_sample}")

    def _fit(self, source_embeddings: np.ndarray, target_embeddings: np.ndarray,
             reference_indices: np.ndarray, **kwargs) -> None:
        """Fit Gromov-Wasserstein mapping using reference embeddings."""
        logger.info(f"Fitting Gromov-Wasserstein mapping with {len(reference_indices)} reference points")

        src = torch.from_numpy(source_embeddings).to(self.device, dtype=self.dtype)
        tgt = torch.from_numpy(target_embeddings).to(self.device, dtype=self.dtype)
        ref_idx = torch.from_numpy(reference_indices).to(self.device, dtype=torch.long)

        X_ref = src[ref_idx]
        Y_ref = tgt[ref_idx]
        n_ref = len(reference_indices)

        if n_ref > self.max_reference_points:
            logger.warning(f"Reference points ({n_ref}) exceed max limit ({self.max_reference_points})")
            logger.warning(f"Sampling to {self.max_reference_points} points")

            sample_indices = torch.randperm(n_ref, device=self.device)[:self.max_reference_points]
            X_ref = X_ref[sample_indices]
            Y_ref = Y_ref[sample_indices]
            ref_idx = ref_idx[sample_indices]
            n_ref = self.max_reference_points

        max_memory_gb = kwargs.get('max_memory_gb', self.max_memory_gb)
        if self.auto_sample and not check_memory_feasibility(n_ref, n_ref, self.device, max_memory_gb):
            suggested_size = suggest_sample_size(n_ref, n_ref, self.device, max_memory_gb)
            logger.warning(f"Memory insufficient for {n_ref} reference points!")
            logger.warning(f"Auto-sampling to {suggested_size} points to fit memory constraints")

            sample_indices = torch.randperm(n_ref, device=self.device)[:suggested_size]
            X_ref = X_ref[sample_indices]
            Y_ref = Y_ref[sample_indices]
            ref_idx = ref_idx[sample_indices]
            n_ref = suggested_size

            logger.info(f"Using {n_ref} sampled reference points for fitting")
        elif not check_memory_feasibility(n_ref, n_ref, self.device, max_memory_gb):
            required_memory = estimate_memory_usage(n_ref, n_ref, self.device)
            if self.device.type == "cuda":
                available_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            else:
                available_memory = psutil.virtual_memory().available / (1024**3)

            logger.error(f"Memory insufficient! Required: {required_memory:.2f}GB, Available: {available_memory:.2f}GB")
            logger.error(f"Consider enabling auto_sample=True or reducing max_reference_points (current: {self.max_reference_points})")
            raise MemoryError(f"Insufficient memory for {n_ref} reference points. Enable auto_sample or reduce data size.")

        X_ref_np = X_ref.cpu().numpy()
        Y_ref_np = Y_ref.cpu().numpy()

        logger.info(f"Reference embeddings shape: {X_ref_np.shape}, {Y_ref_np.shape}")

        self.source_embeddings = src
        self.target_embeddings = tgt
        self.source_indices = ref_idx

        p = np.full(n_ref, 1 / n_ref, dtype=np.float64)
        q = np.full(n_ref, 1 / n_ref, dtype=np.float64)

        logger.info("Computing distance matrices...")
        C1 = ot.dist(X_ref_np, X_ref_np, metric='euclidean')
        C2 = ot.dist(Y_ref_np, Y_ref_np, metric='euclidean')

        logger.info("Solving Gromov-Wasserstein problem...")
        eps = getattr(self.config.gromov_wasserstein_config, 'epsilon', 0.1)
        max_iter = getattr(self.config.gromov_wasserstein_config, 'max_iter', 1000)
        tol = getattr(self.config.gromov_wasserstein_config, 'tol', 1e-9)
        verbose = getattr(self.config.gromov_wasserstein_config, 'verbose', False)

        logger.info(f"GW parameters: epsilon={eps}, max_iter={max_iter}, tol={tol}")

        T = ot.gromov.entropic_gromov_wasserstein(
            C1, C2, p, q,
            loss_fun='square_loss',
            epsilon=eps,
            solver='PGD',
            max_iter=max_iter,
            tol=tol,
            verbose=verbose
        )

        self.transport_plan = torch.from_numpy(T).to(self.device, dtype=self.dtype)

        self.metadata = {
            'loss_fun': 'square_loss',
            'epsilon': eps,
            'max_iter': max_iter,
            'tol': tol,
            'original_reference_size': len(reference_indices),
            'used_reference_size': n_ref,
            'was_sampled': len(reference_indices) != n_ref,
            'source_dimension': X_ref.shape[1],
            'target_dimension': Y_ref.shape[1],
            'transport_plan_shape': self.transport_plan.shape,
            'device': str(self.device),
            'dtype': str(self.dtype),
            'max_reference_points': self.max_reference_points,
            'auto_sample': self.auto_sample
        }

        logger.info("Gromov-Wasserstein mapping fitting completed")
        logger.info(f"Transport plan shape: {self.transport_plan.shape}")

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            logger.info("GPU memory cache cleared")

    def _transform(self, embeddings: np.ndarray, **kwargs) -> np.ndarray:
        """Transform embeddings using fitted Gromov-Wasserstein mapping."""
        if not self.is_fitted:
            raise ValueError("Gromov-Wasserstein mapping must be fitted before transformation")

        if self.transport_plan is None:
            raise ValueError("Transport plan not computed")

        if self.source_indices is None or self.source_embeddings is None or self.target_embeddings is None:
            raise ValueError("Reference embeddings not available")

        E = torch.from_numpy(embeddings).to(self.device, dtype=self.dtype)

        src_ref = self.source_embeddings[self.source_indices]
        tgt_ref = self.target_embeddings[self.source_indices]

        logger.info(f"Transforming {E.shape[0]} embeddings using transport plan")

        D = torch.cdist(E, src_ref)
        sigma = D.mean(dim=1, keepdim=True)
        W = torch.softmax(-D / sigma, dim=1)

        B = W @ self.transport_plan
        transformed = B @ tgt_ref

        result = transformed.cpu().numpy()

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return result

    def save(self, path: Union[str, Path]) -> None:
        """Save Gromov-Wasserstein mapping parameters."""
        if self.transport_plan is None or self.source_embeddings is None or \
           self.target_embeddings is None or self.source_indices is None:
            raise ValueError("Cannot save unfitted mapping. Call fit() first.")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        torch.save({
            "transport_plan": self.transport_plan.cpu(),
            "source_embeddings": self.source_embeddings.cpu(),
            "target_embeddings": self.target_embeddings.cpu(),
            "source_indices": self.source_indices.cpu(),
            "metadata": self.metadata,
            "max_reference_points": self.max_reference_points,
            "auto_sample": self.auto_sample,
            "max_memory_gb": self.max_memory_gb
        }, path / "gw_pot_mapping.pt")

        logger.info(f"Gromov-Wasserstein mapping saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'GromovWassersteinMappingStrategy':
        """Load Gromov-Wasserstein mapping parameters."""
        path = Path(path)
        mapping_file = path / "gw_pot_mapping.pt"

        if not mapping_file.exists():
            raise FileNotFoundError(f"Mapping file not found: {mapping_file}")

        data = torch.load(mapping_file)

        inst = cls(MappingConfig())

        inst.device = get_device()
        dt = torch.float32 if inst.device.type == "cuda" else torch.float64

        inst.transport_plan = data["transport_plan"].to(inst.device, dtype=dt)
        inst.source_embeddings = data["source_embeddings"].to(inst.device, dtype=dt)
        inst.target_embeddings = data["target_embeddings"].to(inst.device, dtype=dt)
        inst.source_indices = data["source_indices"].to(inst.device)
        inst.is_fitted = True

        inst.metadata = data.get("metadata", {})
        inst.max_reference_points = data.get("max_reference_points", 1000)
        inst.auto_sample = data.get("auto_sample", True)
        inst.max_memory_gb = data.get("max_memory_gb", None)

        logger.info(f"Gromov-Wasserstein mapping loaded from {path}")
        if inst.transport_plan is not None:
            logger.info(f"Transport plan shape: {inst.transport_plan.shape}")

        return inst

    def check_fit(self, path: Union[str, Path]) -> bool:
        """Check if Gromov-Wasserstein mapping is fitted and saved."""
        path = Path(path)
        return (path / "gw_pot_mapping.pt").exists()