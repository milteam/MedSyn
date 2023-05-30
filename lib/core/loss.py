"""Provides abstract class for model losses."""

from abc import ABC, abstractmethod

import yaml
from torch import nn, Tensor
from lib.utils import AbstractYAMLMeta


class BaseLoss(ABC, yaml.YAMLObject, nn.Module, metaclass=AbstractYAMLMeta):
    """Abstract class for model losses."""

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

    def __init__(self, *args, **kwargs):
        """Initialize object."""
        super().__init__()

    @abstractmethod
    def forward(self, *args, **kwargs) -> Tensor:
        """Perform computation.

        Implementation required in subclass.
        """
        raise NotImplementedError
