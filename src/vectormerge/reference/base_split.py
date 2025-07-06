import numpy as np
from typing import Optional
from pathlib import Path
from loguru import logger

class BaseSplit:
    def __init__(self, dataset_name: str, dataset_index: np.ndarray, reference_ratio: float, reference_path: str):
        self.dataset_index = dataset_index
        self.reference_ratio = reference_ratio
        self.reference_path = reference_path
        self.reference_path = Path(reference_path)
        self.reference_path.parent.mkdir(parents=True, exist_ok=True)
        self.reference_key = Path(f"random_split_{dataset_name}_{reference_ratio:.2f}")

    @property
    def final_reference_path(self):
        return self.reference_path / self.reference_key
    
    def split(self, **kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split the dataset into three parts:
        - D0: reference
        - D1: query
        - D2: test
        """
        d0_index, d1_index, d2_index = self._split(**kwargs)
        self.check_overlap(d0_index, d1_index, d2_index)
        if "save" in kwargs and not kwargs["save"]:
            logger.info(f"Not saving reference to {self.final_reference_path} due to save=False")
        else:
            self.save_reference(d0_index, d1_index, d2_index)
        return d0_index, d1_index, d2_index

    def _split(self, dataset_index: Optional[np.ndarray] = None, reference_ratio: Optional[float] = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split the dataset into three parts, basic random split.
        - D0: reference
        - D1: query
        - D2: test

        Args:
            dataset_index: the index of the dataset
            reference_ratio: the ratio of the reference
            save: whether to save the reference

        Returns:
            d0_index: the index of the reference
            d1_index: the index of the D1
            d2_index: the index of the D2
        """
        if dataset_index is None:
            dataset_index = self.dataset_index
        if reference_ratio is None:
            reference_ratio = self.reference_ratio
        
        # random split reference_ratio of dataset_index to reference (D0), then evenly split the rest to d1 and d2 (D1 and D2)
        d0_index = np.random.choice(dataset_index, size=int(len(dataset_index) * reference_ratio), replace=False)
        remaining_index = np.setdiff1d(dataset_index, d0_index)
        
        # Evenly split the remaining indices between D1 and D2
        np.random.shuffle(remaining_index)
        split_point = len(remaining_index) // 2
        d1_index = remaining_index[:split_point]
        d2_index = remaining_index[split_point:]

        return d0_index, d1_index, d2_index
    
    def check_overlap(self, d0_index: np.ndarray, d1_index: np.ndarray, d2_index: np.ndarray) -> bool:
        """
        Check if there is any overlap between D0, D1, and D2

        Args:
            d0_index: the index of the reference
            d1_index: the index of the D1
            d2_index: the index of the D2
        """
        if len(np.intersect1d(d0_index, d1_index)) > 0 or len(np.intersect1d(d0_index, d2_index)) > 0 or len(np.intersect1d(d1_index, d2_index)) > 0:
            raise ValueError("Overlap detected between D0, D1, and D2")
        return True
    
    def save_reference(self, d0_index: np.ndarray, d1_index: np.ndarray, d2_index: np.ndarray) -> None:
        """
        Save the reference to the reference_path, the reference is a dictionary with keys "d0_index", "d1_index", and "d2_index"

        Args:
            d0_index: the index of the reference
            d1_index: the index of the D1
            d2_index: the index of the D2
        """
        logger.info(f"Saving reference to {self.final_reference_path}")
        np.savez(
            self.final_reference_path,
            d0_index=d0_index,
            d1_index=d1_index,
            d2_index=d2_index
        )

