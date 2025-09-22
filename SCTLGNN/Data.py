import itertools
import logging
import warnings
from abc import ABC, abstractmethod
from pprint import pformat
from typing import (Any, Dict, Iterator, List, Literal, Optional, Sequence,
                    Tuple, Union)

import anndata
import mudata
import numpy as np
import omegaconf
import torch
import scipy.sparse as sp


# Use Python's standard logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Define common type hints
FeatType = Literal["default", "sparse", "numpy", "torch"]
ListConfig = omegaconf.listconfig.ListConfig


def _ensure_iter(val: Optional[Union[List[str], str]]) -> Iterator[Optional[str]]:
    if val is None:
        val = itertools.repeat(None)
    elif isinstance(val, str):
        val = [val]
    elif not isinstance(val, list):
        raise TypeError(f"Input to _ensure_iter must be list, str, or None. Got {type(val)}.")
    return val


def _check_types_and_sizes(types, sizes):
    if len(types) == 0:
        return
    elif len(types) > 1:
        raise TypeError(f"Found mixed types: {types}. Input configs must be either all str or all lists.")
    elif ((type_ := types.pop()) == list) and (len(sizes) > 1):
        raise ValueError(f"Found mixed sizes lists: {sizes}. Input configs must be of same length.")
    elif type_ not in (list, str, ListConfig):
        raise TypeError(f"Unknownn type {type_} found in config.")


class BaseData(ABC):
    """Base data object (standalone version)."""

    _FEATURE_CONFIGS: List[str] = ["feature_mod", "feature_channel", "feature_channel_type"]
    _LABEL_CONFIGS: List[str] = ["label_mod", "label_channel", "label_channel_type"]
    _DATA_CHANNELS: List[str] = ["obs", "var", "obsm", "varm", "obsp", "varp", "layers", "uns"]

    def __init__(self, data: Union[anndata.AnnData, mudata.MuData], train_size: Optional[int] = None, val_size: int = 0,
                 test_size: int = -1, split_index_range_dict: Optional[Dict[str, Tuple[int, int]]] = None,
                 full_split_name: Optional[str] = None):
        super().__init__()
        if isinstance(data, anndata.AnnData):
            additional_channels = ["X"]
        elif isinstance(data, mudata.MuData):
            additional_channels = ["X", "mod"]
        else:
            raise TypeError(f"Unknown data type {type(data)}, must be either AnnData or MuData.")
        self._data = data
        for prop in self._DATA_CHANNELS + additional_channels:
            if not hasattr(self, prop): # Check to avoid overwriting existing attributes
                setattr(self, prop, getattr(data, prop))
        self._split_idx_dict: Dict[str, Sequence[int]] = {}
        self._setup_splits(train_size, val_size, test_size, split_index_range_dict, full_split_name)
        if "dance_config" not in self._data.uns:
            self._data.uns["dance_config"] = dict()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} object that wraps (.data):\n{self.data}"

    def _setup_splits(self, train_size, val_size, test_size, split_index_range_dict, full_split_name):
        if (split_index_range_dict is not None) and (full_split_name is not None):
            raise ValueError("Only one of split_index_range_dict, full_split_name can be specified, but not both")
        elif split_index_range_dict is not None:
            self._setup_splits_range(split_index_range_dict)
        elif full_split_name is not None:
            self._setup_splits_full(full_split_name)
        else:
            self._setup_splits_default(train_size, val_size, test_size)

    def _setup_splits_default(self, train_size, val_size, test_size):
        if train_size is None:
            return
        elif isinstance(train_size, str) and train_size.lower() == "all":
            train_size = -1
            val_size = test_size = 0
        elif any(not isinstance(i, int) for i in (train_size, val_size, test_size)):
            raise TypeError("Split sizes must be of type int")
        split_names, split_sizes = ["train", "val", "test"], np.array((train_size, val_size, test_size))
        if (split_sizes == -1).sum() > 1:
            raise ValueError("Only one split can be specified as -1")
        data_size = self.num_cells
        for name, size in zip(split_names, split_sizes):
            if not (-1 <= size <= data_size):
                raise ValueError(f"{name}={size:,} is invalid for total samples {data_size:,}")
        if (tot_size := split_sizes.clip(0).sum()) > data_size:
            raise ValueError(f"Total size {tot_size:,} exceeds total number of samples {data_size:,}")
        split_sizes[split_sizes == -1] = data_size - split_sizes.clip(0).sum()
        split_thresholds = split_sizes.cumsum()
        for i, split_name in enumerate(split_names):
            start = split_thresholds[i - 1] if i > 0 else 0
            end = split_thresholds[i]
            if end - start > 0:
                self._split_idx_dict[split_name] = list(range(start, end))

    def _setup_splits_range(self, split_index_range_dict):
        for split_name, index_range in split_index_range_dict.items():
            if not (isinstance(index_range, tuple) and len(index_range) == 2 and all(isinstance(i, int) for i in index_range)):
                raise TypeError(f"Split index range must be a tuple of two ints. Got {index_range!r} for key {split_name!r}")
            start, end = index_range
            if end - start > 0:
                self._split_idx_dict[split_name] = list(range(start, end))

    def _setup_splits_full(self, full_split_name: str):
        self._split_idx_dict[full_split_name] = list(range(self.shape[0]))

    def __getitem__(self, idx: Sequence[int]) -> Any:
        return self.data[idx]

    @property
    def data(self):
        return self._data
    
    @property
    @abstractmethod
    def x(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def y(self):
        raise NotImplementedError

    @property
    def config(self) -> Dict[str, Any]:
        return self._data.uns["dance_config"]

    def set_config(self, *, overwrite: bool = False, **kwargs):
        self.set_config_from_dict(kwargs, overwrite=overwrite)

    def set_config_from_dict(self, config_dict: Dict[str, Any], *, overwrite: bool = False):
        all_configs = set(self._FEATURE_CONFIGS + self._LABEL_CONFIGS)
        if (unknown_options := set(config_dict).difference(all_configs)):
            raise KeyError(f"Unknown config option(s): {unknown_options}")
        feature_configs = [j for i, j in config_dict.items() if i in self._FEATURE_CONFIGS and j is not None]
        label_configs = [j for i, j in config_dict.items() if i in self._LABEL_CONFIGS and j is not None]
        for i in [feature_configs, label_configs]:
            _check_types_and_sizes(set(map(type, i)), set(map(len, i)))
        for config_key, config_val in config_dict.items():
            if config_key not in self.config:
                if isinstance(config_val, ListConfig):
                    config_val = omegaconf.OmegaConf.to_object(config_val)
                self.config[config_key] = config_val
            elif (old_config_val := self.config[config_key]) != config_val:
                if overwrite:
                    self.config[config_key] = config_val
                else:
                    raise KeyError(f"Config {config_key!r} exists with value {old_config_val!r}. Set overwrite=True to change.")

    @property
    def num_cells(self) -> int:
        return self.data.shape[0]

    # ... (other properties like num_features, cells, train_idx, etc., can be included as needed) ...
    @property
    def shape(self) -> Tuple[int, int]:
        return self.data.shape

    @staticmethod
    def _get_feature(in_data, channel, channel_type, mod):
        if mod is None:
            data = in_data
        elif not isinstance(in_data, mudata.MuData):
            raise AttributeError("`mod` option is only available for MuData.")
        else:
            data = in_data.mod[mod]
        if channel_type == "X":
            return data.X
        elif channel_type == "raw_X":
            return data.raw.X
        else:
            channel_type = channel_type or "obsm"
            if channel_type not in BaseData._DATA_CHANNELS:
                raise ValueError(f"Unknown channel type {channel_type!r}.")
            channel_obj = getattr(data, channel_type)
            if channel is None:
                warnings.warn("Defaulting to data.X. Please specify channel_type='X' in the future.", DeprecationWarning)
                return data.X
            return channel_obj[channel]
    
    def get_feature(self, *, split_name: Optional[str] = None, return_type: FeatType = "numpy",
                    channel: Optional[str] = None, channel_type: Optional[str] = "obsm", mod: Optional[str] = None):
        feature = self._get_feature(self.data, channel, channel_type, mod)
        channel_type = channel_type or "obsm"
        if return_type == "default":
            if split_name is not None:
                raise ValueError("split_name not supported when return_type is 'default'")
            return feature
        if hasattr(feature, "toarray"):
            feature = feature.toarray()
        elif hasattr(feature, "to_numpy"):
            feature = feature.to_numpy()
        if split_name is not None:
            if channel_type in ["X", "raw_X", "obs", "obsm", "obsp", "layers"]:
                idx = self.get_split_idx(split_name, error_on_miss=True)
                feature = feature[idx]
        if return_type == "torch":
            return torch.from_numpy(feature.astype(np.float32))
        elif return_type not in ["numpy", "sparse"]:
            raise ValueError(f"Unknown return_type {return_type!r}")
        return feature
    
    def get_feature(self, *, split_name: Optional[str] = None, return_type: FeatType = "numpy",
                    channel: Optional[str] = None, channel_type: Optional[str] = "obsm", mod: Optional[str] = None):
        feature = self._get_feature(self.data, channel, channel_type, mod)
        if return_type == "default":
            if split_name is not None:
                raise ValueError("split_name not supported when return_type is 'default'")
            return feature
        if split_name is not None:
            channel_type = channel_type or "obsm"
            if channel_type in ["X", "raw_X", "obs", "obsm", "obsp", "layers"]:
                idx = self.get_split_idx(split_name, error_on_miss=True)
                feature = feature[idx]
        if return_type == "sparse":
            if not sp.issparse(feature):
                warnings.warn(f"Warning: requested sparse but feature is of type {type(feature)}. Converting to CSR.")
                return sp.csr_matrix(feature)
            return feature
        if sp.issparse(feature):
            feature = feature.toarray()
        elif hasattr(feature, "to_numpy"):
            feature = feature.to_numpy()
        if return_type == "numpy":
            return feature
        elif return_type == "torch":
            return torch.from_numpy(feature.astype(np.float32))
        else:
            raise ValueError(f"Unknown return_type {return_type!r}")

    def get_split_idx(self, split_name: str, error_on_miss: bool = False):
        if split_name is None:
            return list(range(self.shape[0]))
        elif split_name in self._split_idx_dict:
            return self._split_idx_dict[split_name]
        elif error_on_miss:
            raise KeyError(f"Unknown split {split_name!r}.")
        else:
            return None
    
class Data(BaseData):
    """Data object for single-modality or multi-modality modeling (standalone version)."""
    @property
    def x(self):
        return self.get_x(return_type="default")

    @property
    def y(self):
        return self.get_y(return_type="default")

    def _get(self, config_keys: List[str], *, split_name: Optional[str] = None, return_type: FeatType = "numpy", **kwargs) -> Any:
        info = list(map(self.config.get, config_keys))
        if all(i is None for i in info):
            mods, channels, channel_types = [None], [None], [None]
        else:
            mods, channels, channel_types = map(_ensure_iter, info)
        out = []
        for mod, channel, channel_type in zip(mods, channels, channel_types):
            try:
                x = self.get_feature(split_name=split_name, return_type=return_type, mod=mod, channel=channel,
                                     channel_type=channel_type, **kwargs)
                out.append(x)
            except Exception as e:
                settings = {"split_name": split_name, "return_type": return_type, "mod": mod,
                            "channel": channel, "channel_type": channel_type}
                raise RuntimeError(f"Failed to get features for settings:\n{pformat(settings)}") from e
        return out[0] if len(out) == 1 else out

    def get_x(self, split_name: Optional[str] = None, return_type: FeatType = "numpy", **kwargs) -> Any:
        return self._get(self._FEATURE_CONFIGS, split_name=split_name, return_type=return_type, **kwargs)

    def get_y(self, split_name: Optional[str] = None, return_type: FeatType = "numpy", **kwargs) -> Any:
        return self._get(self._LABEL_CONFIGS, split_name=split_name, return_type=return_type, **kwargs)

    def get_data(self, split_name: Optional[str] = None, return_type: FeatType = "numpy",
                 x_kwargs: Dict[str, Any] = {}, y_kwargs: Dict[str, Any] = {}) -> Tuple[Any, Any]:
        x = self.get_x(split_name, return_type, **x_kwargs)
        y = self.get_y(split_name, return_type, **y_kwargs)
        return x, y

    def get_train_data(self, return_type: FeatType = "numpy", **kwargs) -> Tuple[Any, Any]:
        return self.get_data("train", return_type, **kwargs)

    def get_val_data(self, return_type: FeatType = "numpy", **kwargs) -> Tuple[Any, Any]:
        return self.get_data("val", return_type, **kwargs)

    def get_test_data(self, return_type: FeatType = "numpy", **kwargs) -> Tuple[Any, Any]:
        return self.get_data("test", return_type, **kwargs)