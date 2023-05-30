"""Provides abstract class for model metrics."""

from abc import ABC, abstractmethod
from typing import Any, Dict

import yaml
from lib.utils import AbstractYAMLMeta


class BaseMetric(ABC, yaml.YAMLObject, metaclass=AbstractYAMLMeta):
    """Abstract class for model metrics."""

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
        """Initialize object."""
        super().__init__()

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get all parameters needed for initialization.

        Returns
        -------
        Dict[str, Any]:
            Dict of all parameters needed for initialization with the same key values as
            parameters names.

        """
        raise NotImplementedError

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Dict[str, Any]:
        """Perform computation.

        Implementation required in subclass.
        """
        raise NotImplementedError
