# A standalone module integrating a feature-rich BaseTransform with BatchFeature

import hashlib
import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp

# Define LogLevel type hint for clarity
LogLevel = str

class BaseTransform(ABC):
    """
    BaseTransform abstract object (Standalone Version).
    Provides standardized logging, representation, and hashing for transform objects.
    """

    _DISPLAY_ATTRS: Tuple[str, ...] = ()

    def __init__(self, out: Optional[str] = None, log_level: LogLevel = "INFO"):
        # Set the output channel name, default to the class name
        self.out = out or self.name

        # Replace dance.logger with standard Python logging
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(log_level.upper())
        self.log_level = log_level
        
        # Configure basic logging if not already configured
        if not logging.getLogger().handlers:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def hexdigest(self) -> str:
        """Return MD5 hash using the representation of the transform object."""
        return hashlib.md5(repr(self).encode()).hexdigest()

    def __repr__(self) -> str:
        """Provide a clear string representation of the object and its parameters."""
        display_attrs_str_list = [f"{i}={getattr(self, i)!r}" for i in self._DISPLAY_ATTRS]
        display_attrs_str = ", ".join(display_attrs_str_list)
        return f"{self.name}({display_attrs_str})"

    @abstractmethod
    def __call__(self, adata: anndata.AnnData) -> anndata.AnnData:
        """
        Abstract call method. All subclasses must implement this.
        Note: The input and output type is now anndata.AnnData.
        """
        raise NotImplementedError


class BatchFeature(BaseTransform):
    """
    Assign statistical batch features for each cell. (Standalone Version)

    Inherits from the feature-rich BaseTransform to provide standardized
    logging and representation.
    """

    def __init__(self, out: Optional[str] = "batch_features", log_level: LogLevel = "INFO", **kwargs):
        # Pass 'out' and 'log_level' up to the BaseTransform parent class
        super().__init__(out=out, log_level=log_level)
        # Note: **kwargs is not used by this class but is included for compatibility

    def __call__(self, adata: anndata.AnnData) -> anndata.AnnData:
        """
        Processes the AnnData object to compute and assign batch features.

        Parameters
        ----------
        adata
            An AnnData object. It must contain:
            - .X matrix (sparse format is expected).
            - .obs['batch'] column with batch labels for each cell.

        Returns
        -------
        anndata.AnnData
            The input AnnData object, modified in-place with the new features
            stored in .obsm[self.out].
        """
        self.logger.info(f"Starting batch feature calculation, output will be stored in '.obsm[\"{self.out}\"]'")

        if "batch" not in adata.obs:
            raise ValueError("Required '.obs[\"batch\"]' not found in the AnnData object.")

        cells_data = []
        columns = [
            "cell_mean", "cell_std", "nonzero_25%", "nonzero_50%", "nonzero_75%",
            "nonzero_max", "nonzero_count", "nonzero_mean", "nonzero_std", "batch",
        ]

        batch_labels = list(adata.obs["batch"])
        self.logger.info(f"Found batches: {sorted(list(set(batch_labels)))}")

        feature_matrix = adata.X.tocsr()

        for i in range(feature_matrix.shape[0]):
            cell_row = feature_matrix[i, :]
            nz_values = cell_row.data
            
            if len(nz_values) == 0:
                self.logger.warning(f"Cell at index {i} contains all zero features. Appending zero-vector.")
                cells_data.append([0] * 9 + [batch_labels[i]])
                continue

            cells_data.append([
                cell_row.mean(),
                np.sqrt(cell_row.power(2).mean() - cell_row.mean()**2), # Correct std for sparse matrices
                np.percentile(nz_values, 25),
                np.percentile(nz_values, 50),
                np.percentile(nz_values, 75),
                cell_row.max(),
                len(nz_values) / 1000,
                nz_values.mean(),
                nz_values.std(),
                batch_labels[i]
            ])

        cell_features_df = pd.DataFrame(cells_data, columns=columns)
        batch_summary_df = cell_features_df.groupby("batch").mean().reset_index()
        
        batch_label_list = batch_summary_df.batch.tolist()
        batch_vectors = batch_summary_df.drop("batch", axis=1).to_numpy()
        batch_to_features_map = dict(zip(batch_label_list, batch_vectors))

        final_batch_features = np.array([batch_to_features_map[b] for b in adata.obs["batch"]], dtype=np.float32)

        adata.obsm[self.out] = final_batch_features
        self.logger.info(f"Successfully computed batch features of shape {final_batch_features.shape}")
        self.logger.info(f"Result stored in '.obsm[\"{self.out}\"]'")

        return adata