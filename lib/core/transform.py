"""Provides abstract class for transforms."""

from abc import ABC, abstractmethod
from typing import Any, Dict

import yaml
from lib.utils import AbstractYAMLMeta


class BaseTransform(ABC, yaml.YAMLObject, metaclass=AbstractYAMLMeta):
    """Abstract class for Transform."""

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
        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get all object's parameters needed for initialization.

        Returns
        -------
        Dict[str, Any]:
            Dict of all object's parameters needed for initialization with the same key values as
            parameters names.
        """
        raise NotImplementedError

    @abstractmethod
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply transform."""
        raise NotImplementedError
