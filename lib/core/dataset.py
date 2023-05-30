"""Provides abstract class for datasets."""

import collections
import itertools
from abc import ABC, abstractmethod
from typing import Any, Dict, Sequence

import yaml
import torch
from torch.utils import data

from lib.utils import AbstractYAMLMeta


class BaseDataset(ABC, yaml.YAMLObject, data.Dataset, metaclass=AbstractYAMLMeta):
    """Abstract class for deep learning datasets."""

    yaml_loader = yaml.Loader
    yaml_dumper = yaml.Dumper

    @classmethod
    def from_yaml(cls, loader, node):
        """Create object from yaml node."""
        data = loader.construct_mapping(node, deep=True)
        return cls(**data)

    @classmethod
    def to_yaml(cls, dumper, node):
        """Serialize object to yaml."""
        return dumper.represent_mapping(cls.yaml_tag, node.get_parameters())

    @abstractmethod
    def __init__(self, *args, **kwargs):
        """Initialize dataset."""
        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get all model's parameters needed for initialization.

        Returns
        -------
        Dict[str, Any]:
            Dict of all parameters needed for initialization with the same key values as
            parameters names.
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        """Get dataset lengths."""
        raise NotImplementedError

    @abstractmethod
    def __load_data__(self, index: int) -> Dict[str, Any]:
        """Load one sample from dataset by index.

        Parameters
        ----------
        index: int
            Index of data to be loaded.

        Returns
        -------
        Dict[str, Any]:
            Dictionary of data.
        """
        raise NotImplementedError

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Load one sample from dataset and apply transforms.

        Parameters
        ----------
        index: int
            Index of data to be loaded.

        Returns
        -------
        Dict[str, Any]:
            Dictionary of data.
        """
        data = self.__load_data__(index)
        if self.transforms is None:
            return data
        else:
            return self.transforms(data)

    @staticmethod
    def collate_fn(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate data in batch."""
        data: Dict[str, Any] = collections.defaultdict(list)

        for key, val in itertools.chain(*[s.items() for s in batch]):
            data[key].append(val)

        if "mask" in data.keys():
            data["mask"] = torch.stack(data["mask"], dim=0)
        data["image"] = torch.stack(data["image"], dim=0)
        return data
