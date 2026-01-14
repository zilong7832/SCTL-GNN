import hashlib
import logging
import os
import pickle
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import dgl
import numpy as np
import scipy.sparse
import torch
from sklearn.decomposition import TruncatedSVD
from Data import Data

# Define LogLevel type hint for clarity
LogLevel = str

class BaseTransform(ABC):
    """
    BaseTransform abstract object (Standalone Version).
    Provides standardized logging, representation, and hashing for transform objects.
    """

    _DISPLAY_ATTRS: Tuple[str, ...] = ()

    def __init__(self, out: Optional[str] = None, log_level: LogLevel = "INFO"):
        self.out = out or self.name
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(log_level.upper())
        self.log_level = log_level
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
    def __call__(self, data: Data) -> Data:
        """Abstract call method. All subclasses must implement this."""
        raise NotImplementedError

class SCTLGNNGraph(BaseTransform):
    """
    Construct the cell-feature graph object for ScTlGNN (Standalone Version, no pathway).
    """

    def __init__(self, inductive: bool = False, cell_init: str = 'none', **kwargs):
        super().__init__(**kwargs)
        self.inductive = inductive
        self.cell_init = cell_init

    def _construct_feature_graph(
        self, u, v, e, train_size, feature_size, cell_node_features, _test_graph=False
    ):
        graph_data = {
            ('cell', 'cell2feature', 'feature'): (u, v),
            ('feature', 'feature2cell', 'cell'): (v, u),
        }

        graph = dgl.heterograph(graph_data)

        if self.inductive:
            cell_nodes_to_feature = (
                cell_node_features[:train_size] if not _test_graph else cell_node_features
            )
        else:
            cell_nodes_to_feature = cell_node_features

        # Ensure the features match the number of nodes in the graph
        if cell_nodes_to_feature.shape[0] != graph.num_nodes('cell'):
            raise ValueError(
                f"Number of cell features ({cell_nodes_to_feature.shape[0]}) "
                f"does not match number of cell nodes in graph ({graph.num_nodes('cell')}). "
                f"Test graph: {_test_graph}"
            )

        graph.nodes['cell'].data['id'] = cell_nodes_to_feature
        graph.nodes['feature'].data['id'] = torch.arange(graph.num_nodes('feature')).long()
        graph.edges['feature2cell'].data['weight'] = e.float()
        graph.edges['cell2feature'].data['weight'] = e[:graph.num_edges('cell2feature')].float()

        return graph

    def __call__(self, data: Data) -> Data:
        self.logger.info("Constructing graph for ScTlGNN...")

        # Get data using the provided Data object
        x_train, _        = data.get_train_data(return_type="numpy")
        x_train_sparse, _ = data.get_train_data(return_type="sparse")
        x_test_sparse, _  = data.get_test_data (return_type="sparse")

        train_size = x_train_sparse.shape[0]
        feature_size = x_train_sparse.shape[1]
        cell_size = train_size + x_test_sparse.shape[0]

        # Initialize cell node features
        if self.cell_init == 'none':
            # DGL requires features to have a second dimension
            cell_node_features = torch.ones(cell_size).long()
        elif self.cell_init == 'svd':
            self.logger.info("Initializing cell features with SVD.")
            embedder_mod1 = TruncatedSVD(n_components=100)
            X_train_np = embedder_mod1.fit_transform(x_train_sparse)
            X_test_np = embedder_mod1.transform(x_test_sparse)
            cell_node_features = torch.from_numpy(
                np.vstack((X_train_np, X_test_np))
            ).float()
        else:
            raise ValueError(f"Unknown cell_init option: {self.cell_init}")

        # Construct graph(s) based on mode (inductive/transductive)
        if self.inductive:
            self.logger.info("Constructing graphs in inductive mode.")

            # Training graph
            train_idx = x_train_sparse.nonzero()
            u_train = torch.from_numpy(train_idx[0])
            v_train = torch.from_numpy(train_idx[1])
            e_train = torch.from_numpy(x_train_sparse.data)
            g = self._construct_feature_graph(
                u_train, v_train, e_train, train_size, feature_size,
                cell_node_features, _test_graph=False
            )

            # Full graph for testing/inference
            full_sparse = scipy.sparse.vstack([x_train_sparse, x_test_sparse])
            full_idx = full_sparse.nonzero()
            u_full = torch.from_numpy(full_idx[0])
            v_full = torch.from_numpy(full_idx[1])
            e_full = torch.from_numpy(full_sparse.data)
            gtest = self._construct_feature_graph(
                u_full, v_full, e_full, train_size, feature_size,
                cell_node_features, _test_graph=True
            )

            data.data.uns['g'] = g
            data.data.uns['gtest'] = gtest

        else:
            self.logger.info("Constructing graph in transductive mode.")
            full_sparse = scipy.sparse.vstack([x_train_sparse, x_test_sparse])
            full_idx = full_sparse.nonzero()
            u_full = torch.from_numpy(full_idx[0])
            v_full = torch.from_numpy(full_idx[1])
            e_full = torch.from_numpy(full_sparse.data)
            g = self._construct_feature_graph(
                u_full, v_full, e_full, train_size, feature_size,
                cell_node_features
            )

            data.data.uns['g'] = g

        self.logger.info("Graph construction complete.")
        return data