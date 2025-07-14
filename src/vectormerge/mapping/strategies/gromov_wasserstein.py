"""
Gromov-Wasserstein mapping strategy for VectorMerge.

This module implements Gromov-Wasserstein alignment for mapping between embedding spaces.
Based on the approach from Alvarez-Melis and Jaakkola (2018), this method compares
the internal structure of embedding spaces rather than absolute positions.
"""

import numpy as np
from typing import Optional, Dict, Any, Union, Tuple, cast
from pathlib import Path
from loguru import logger
import joblib
import psutil
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from tqdm import trange

from ..base import MappingStrategy, MappingConfig


def estimate_memory_usage(n1: int, n2: int) -> float:
    """Estimate memory usage for Gromov-Wasserstein computation in GB.
    
    Args:
        n1: Number of samples in source space
        n2: Number of samples in target space
        
    Returns:
        Estimated memory usage in GB
    """
    # Distance matrices: n1*n1 + n2*n2
    # Transport matrix: n1*n2  
    # Cost matrix: n1*n2
    # Temporary matrices during computation: ~3*n1*n2
    total_elements = n1*n1 + n2*n2 + 5*n1*n2
    bytes_per_element = 8  # float64
    return total_elements * bytes_per_element / (1024**3)


def check_memory_feasibility(n1: int, n2: int, max_memory_gb: Optional[float] = None) -> bool:
    """Check if computation is feasible given memory constraints.
    
    Args:
        n1: Number of samples in source space
        n2: Number of samples in target space
        max_memory_gb: Maximum memory to use (default: 80% of available)
        
    Returns:
        True if feasible, False otherwise
    """
    if max_memory_gb is None:
        available_memory = psutil.virtual_memory().available / (1024**3)
        max_memory_limit = available_memory * 0.8
    else:
        max_memory_limit = max_memory_gb
    
    required_memory = estimate_memory_usage(n1, n2)
    return required_memory <= max_memory_limit


def compute_distance_matrix(embeddings: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """Compute pairwise distance matrix for embeddings.
    
    Args:
        embeddings: Embedding matrix (n_samples, n_features)
        metric: Distance metric to use
        
    Returns:
        Distance matrix (n_samples, n_samples)
    """
    return cdist(embeddings, embeddings, metric=metric)  # type: ignore


def gromov_wasserstein_loss(C1: np.ndarray, C2: np.ndarray, T: np.ndarray) -> float:
    """Compute Gromov-Wasserstein loss using vectorized operations.
    
    Args:
        C1: Distance matrix of source space
        C2: Distance matrix of target space
        T: Transport plan
        
    Returns:
        Gromov-Wasserstein loss value
    """
    # Vectorized computation: sum_{i,j,k,l} (C1[i,j] - C2[k,l])^2 * T[i,k] * T[j,l]
    # This can be computed as: tr(C1^T @ T @ C2 @ T^T) - 2*tr(C1^T @ T @ C2^T @ T^T) + tr(C1 @ T @ C1 @ T^T)
    
    # More efficient approach using einsum
    loss = np.einsum('ij,kl,ik,jl->', C1**2, np.ones_like(C2), T, T)
    loss += np.einsum('ij,kl,ik,jl->', np.ones_like(C1), C2**2, T, T)
    loss -= 2 * np.einsum('ij,kl,ik,jl->', C1, C2, T, T)
    
    return float(loss)


def gromov_wasserstein_gradient(C1: np.ndarray, C2: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Compute gradient of Gromov-Wasserstein loss with respect to T.
    
    Args:
        C1: Distance matrix of source space
        C2: Distance matrix of target space
        T: Transport plan
        
    Returns:
        Gradient matrix
    """
    n, m = T.shape
    gradient = np.zeros((n, m))
    
    for i in range(n):
        for k in range(m):
            grad_ik = 0.0
            for j in range(n):
                for l in range(m):
                    grad_ik += 2 * (C1[i, j] - C2[k, l]) * T[j, l]
            gradient[i, k] = grad_ik
    
    return gradient


def sinkhorn_stabilized(K: np.ndarray, u: np.ndarray, v: np.ndarray, 
                       reg: float, numItermax: int = 1000, tau: float = 1e3,
                       stopThr: float = 1e-9) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stabilized Sinkhorn algorithm for entropic regularized optimal transport.
    
    Args:
        K: Cost matrix
        u: Source distribution
        v: Target distribution
        reg: Regularization parameter
        numItermax: Maximum number of iterations
        tau: Threshold for numerical stability
        stopThr: Stopping threshold
        
    Returns:
        Tuple of (dual variables u, dual variables v)
    """
    n, m = K.shape
    
    # Initialize dual variables
    alpha = np.zeros(n)
    beta = np.zeros(m)
    
    for i in range(numItermax):
        # Update alpha
        alpha_prev = alpha.copy()
        K_alpha = np.exp((beta[None, :] - K) / reg)
        alpha = reg * np.log(u) - reg * np.log(np.sum(K_alpha, axis=1))
        
        # Update beta
        K_beta = np.exp((alpha[:, None] - K) / reg)
        beta = reg * np.log(v) - reg * np.log(np.sum(K_beta, axis=0))
        
        # Check for numerical stability
        if np.max(np.abs(alpha)) > tau or np.max(np.abs(beta)) > tau:
            alpha = alpha - np.max(alpha)
            beta = beta - np.max(beta)
        
        # Check convergence
        if np.linalg.norm(alpha - alpha_prev) < stopThr:
            break
    
    return alpha, beta


def gromov_wasserstein_solver(C1: np.ndarray, C2: np.ndarray, p: np.ndarray, q: np.ndarray,
                             loss_fun: str = "square_loss", epsilon: float = 0.1,
                             max_iter: int = 1000, tol: float = 1e-9,
                             verbose: bool = False, log: bool = False) -> np.ndarray:
    """
    Solve Gromov-Wasserstein problem with entropic regularization using vectorized operations.
    
    Args:
        C1: Distance matrix of source space
        C2: Distance matrix of target space
        p: Source distribution
        q: Target distribution
        loss_fun: Loss function type
        epsilon: Entropic regularization parameter
        max_iter: Maximum number of iterations
        tol: Tolerance for convergence
        verbose: Whether to print progress
        log: Whether to log intermediate results
        
    Returns:
        Optimal transport plan
    """
    n, m = C1.shape[0], C2.shape[0]
    
    # Check memory feasibility
    if not check_memory_feasibility(n, m):
        required_memory = estimate_memory_usage(n, m)
        available_memory = psutil.virtual_memory().available / (1024**3)
        logger.error(f"Memory insufficient! Required: {required_memory:.2f}GB, Available: {available_memory:.2f}GB")
        logger.error("Please sample your data to reduce the problem size.")
        raise MemoryError("Insufficient memory for Gromov-Wasserstein computation")
    
    logger.info(f"Starting Gromov-Wasserstein solver with problem size {n}x{m}")
    
    # Initialize transport plan
    T = np.outer(p, q)
    
    # Precompute squared distance matrices
    C1_sq = C1 ** 2
    C2_sq = C2 ** 2
    
    progress_bar = trange(max_iter, desc="GW iterations") if verbose else range(max_iter)
    
    for iter_count in progress_bar:
        T_prev = T.copy()
        
        # Vectorized computation of cost matrix
        # cost_matrix[i,k] = sum_j sum_l (C1_sq[i,j] + C2_sq[k,l] - 2*C1[i,j]*C2[k,l]) * T[j,l]
        
        # Efficient matrix operations
        cost_matrix = np.einsum('ij,jl->il', C1_sq, T)  # C1_sq @ T
        cost_matrix += np.einsum('ik,kl->il', T, C2_sq)  # T @ C2_sq
        cost_matrix -= 2 * np.einsum('ij,kl,jl->ik', C1, C2, T)  # 2 * C1 @ (T * C2)
        
        # Solve regularized optimal transport problem
        K = np.exp(-cost_matrix / epsilon)
        alpha, beta = sinkhorn_stabilized(K, p, q, epsilon)
        
        # Update transport plan
        T = np.diag(np.exp(alpha / epsilon)) @ K @ np.diag(np.exp(beta / epsilon))
        
        # Check convergence
        diff = np.linalg.norm(T - T_prev)
        if diff < tol:
            if verbose:
                logger.info(f"Gromov-Wasserstein converged after {iter_count + 1} iterations")
            break
            
        if verbose and hasattr(progress_bar, 'set_postfix'):
            cast(Any, progress_bar).set_postfix({'diff': f'{diff:.2e}'})
    
    return T


class GromovWassersteinMappingStrategy(MappingStrategy):
    """Gromov-Wasserstein mapping strategy.
    
    This strategy uses the Gromov-Wasserstein distance to find correspondences
    between embedding spaces by comparing their internal structure rather than
    absolute positions.
    """
    
    def __init__(self, config: MappingConfig):
        """Initialize Gromov-Wasserstein mapping strategy.
        
        Args:
            config: Mapping configuration containing GW parameters
        """
        super().__init__(config)
        self.transport_plan: Optional[np.ndarray] = None
        self.source_embeddings: Optional[np.ndarray] = None
        self.target_embeddings: Optional[np.ndarray] = None
        self.source_indices: Optional[np.ndarray] = None
        
    def _fit(self, source_embeddings: np.ndarray, target_embeddings: np.ndarray,
             reference_indices: np.ndarray, **kwargs) -> None:
        """Fit Gromov-Wasserstein mapping using reference embeddings.
        
        Args:
            source_embeddings: Source embedding space
            target_embeddings: Target embedding space
            reference_indices: Indices of reference points for alignment
            **kwargs: Additional arguments
        """
        logger.info(f"Fitting Gromov-Wasserstein mapping with {len(reference_indices)} reference points")
        
        # Extract reference embeddings
        X_ref = source_embeddings[reference_indices]
        Y_ref = target_embeddings[reference_indices]
        
        # Store for later use in transform
        self.source_embeddings = source_embeddings
        self.target_embeddings = target_embeddings
        self.source_indices = reference_indices
        
        # Compute distance matrices
        logger.info("Computing distance matrices...")
        C1 = compute_distance_matrix(X_ref)
        C2 = compute_distance_matrix(Y_ref)
        
        # Uniform distributions
        n_ref = len(reference_indices)
        p = np.ones(n_ref) / n_ref
        q = np.ones(n_ref) / n_ref
        
        # Solve Gromov-Wasserstein problem
        logger.info("Solving Gromov-Wasserstein problem...")
        self.transport_plan = gromov_wasserstein_solver(
            C1, C2, p, q,
            loss_fun=self.config.gromov_wasserstein_config.loss_fun,
            epsilon=self.config.gromov_wasserstein_config.epsilon,
            max_iter=self.config.gromov_wasserstein_config.max_iter,
            tol=self.config.gromov_wasserstein_config.tol,
            verbose=self.config.gromov_wasserstein_config.verbose,
            log=self.config.gromov_wasserstein_config.log
        )
        
        # Store metadata
        self.metadata = {
            'loss_fun': self.config.gromov_wasserstein_config.loss_fun,
            'epsilon': self.config.gromov_wasserstein_config.epsilon,
            'max_iter': self.config.gromov_wasserstein_config.max_iter,
            'tol': self.config.gromov_wasserstein_config.tol,
            'reference_size': len(reference_indices),
            'source_dimension': X_ref.shape[1],
            'target_dimension': Y_ref.shape[1],
            'transport_plan_shape': self.transport_plan.shape
        }
        
        logger.info("Gromov-Wasserstein mapping fitting completed")
    
    def _transform(self, embeddings: np.ndarray, **kwargs) -> np.ndarray:
        """Transform embeddings using fitted Gromov-Wasserstein mapping.
        
        Args:
            embeddings: Embeddings to transform
            **kwargs: Additional arguments
            
        Returns:
            Transformed embeddings
        """
        if not self.is_fitted:
            raise ValueError("Gromov-Wasserstein mapping must be fitted before transformation")
        
        if self.transport_plan is None:
            raise ValueError("Transport plan not computed")
        
        if self.source_indices is None or self.source_embeddings is None or self.target_embeddings is None:
            raise ValueError("Reference embeddings not available")
        
        # For embeddings not in the reference set, we use the transport plan
        # to find the best matching target embeddings
        n_embeddings = embeddings.shape[0]
        n_ref = len(self.source_indices)
        
        # Get reference embeddings
        source_ref = self.source_embeddings[self.source_indices]
        target_ref = self.target_embeddings[self.source_indices]
        
        # Transform each embedding
        transformed = np.zeros((n_embeddings, target_ref.shape[1]))
        
        for i in range(n_embeddings):
            # Find distances from current embedding to reference embeddings
            distances = cdist([embeddings[i]], source_ref)[0]
            
            # Use softmax to get weights based on distances
            weights = np.exp(-distances / np.mean(distances))
            weights = weights / np.sum(weights)
            
            # Compute barycenter using transport plan
            barycenter_weights = weights @ self.transport_plan
            barycenter_weights = barycenter_weights / np.sum(barycenter_weights)
            
            # Transform embedding as weighted sum of target references
            transformed[i] = barycenter_weights @ target_ref
        
        return transformed
    
    def save(self, path: Union[str, Path]) -> None:
        """Save Gromov-Wasserstein mapping parameters.
        
        Args:
            path: Path to save the mapping parameters
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save transport plan and other parameters
        params = {
            'transport_plan': self.transport_plan,
            'source_embeddings': self.source_embeddings,
            'target_embeddings': self.target_embeddings,
            'source_indices': self.source_indices,
            'is_fitted': self.is_fitted,
            'metadata': self.metadata
        }
        
        joblib.dump(params, path / "gromov_wasserstein_params.pkl")
        logger.info(f"Gromov-Wasserstein mapping saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'GromovWassersteinMappingStrategy':
        """Load Gromov-Wasserstein mapping parameters.
        
        Args:
            path: Path to load the mapping parameters from
            
        Returns:
            Loaded Gromov-Wasserstein mapping strategy
        """
        path = Path(path)
        
        # Load parameters
        params = joblib.load(path / "gromov_wasserstein_params.pkl")
        
        # Create instance with default config (will be overridden by loaded params)
        from ..base import MappingConfig
        instance = cls(MappingConfig())
        
        # Restore state
        instance.transport_plan = params['transport_plan']
        instance.source_embeddings = params['source_embeddings']
        instance.target_embeddings = params['target_embeddings']
        instance.source_indices = params['source_indices']
        instance.is_fitted = params['is_fitted']
        instance.metadata = params['metadata']
        
        logger.info(f"Gromov-Wasserstein mapping loaded from {path}")
        return instance
    
    def check_fit(self, path: Union[str, Path]) -> bool:
        """Check if Gromov-Wasserstein mapping is fitted and saved.
        
        Args:
            path: Path to check for saved mapping
            
        Returns:
            True if mapping is fitted and saved, False otherwise
        """
        path = Path(path)
        return (path / "gromov_wasserstein_params.pkl").exists() 