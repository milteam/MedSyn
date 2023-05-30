"""Provides abstract class for logger."""

from abc import ABC, abstractmethod
from typing import Any, Dict

import yaml
from lib.utils import AbstractYAMLMeta


class BaseLogger(ABC, yaml.YAMLObject, metaclass=AbstractYAMLMeta):
    """Base class for loggers."""

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
        """Initialize Logger."""
        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get all logger's parameters needed for initialization.

        Returns
        -------
        Dict[str, Any]:
            Dict of all parameters needed for initialization with the same key values as
            parameters names.

        """
        raise NotImplementedError

    @abstractmethod
    def log(
        self,
        scores: Dict[str, float],
        scope: str,
        stage: str,
        batch_step: int,
        epoch_step: int,
    ) -> None:
        """Log scores.

        Parameters
        ----------
        scores : Dict[str, float]
            Scores to log.
        stage : str, optional (default="train")
            Stage of training: typically train or valid but can be anything.
            Defines group for metrics with the same stage. They will be grouped in one tab.
        scope : str, optional (default="batch")
            Schould be 'batch' or 'epoch'. Scope defines step that wil be used: batch_step or epoch_step.
        batch_step : int, optional (default=0)
            Current batch.
        epoch_step: int, optional (default=0)
            Current epoch.

        """
        raise NotImplementedError
