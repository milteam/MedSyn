"""Module provides abstract class for deep learning models."""

import os
from abc import ABC, abstractmethod
from typing import Any, Dict

import yaml
import torch
from torch import nn
from lib.utils import AbstractYAMLMeta


class BaseModel(ABC, yaml.YAMLObject, nn.Module, metaclass=AbstractYAMLMeta):
    """Abstract class for deep learning models."""

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
    def get_parameters(self) -> Dict[str, Any]:
        """Get all model's parameters needed for initialization.

        Returns
        -------
        Dict[str, Any]:
            Dict of all model's parameters needed for initialization with the same key values as
            parameters names.

        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        """Perform infer for any input formats.

        Parameters
        ----------
        x : torch.Tensor
            Inference batch.

        Returns
        -------
        tensor or ndarray :
            Processed input data.

        """
        raise NotImplementedError

    def count_parameters(self) -> int:
        """Count total number of parameters in model.

        Returns
        -------
        int:
            Total number of parameters in model.

        """
        return sum(p.numel() for p in self.parameters())

    def save(self, path):
        """Save model to a file.

        Parameters
        ----------
        path : str
            Path to the save file.

        """
        folder_path = os.path.dirname(path)
        if len(folder_path) > 0 and not os.path.exists(folder_path):
            os.makedirs(folder_path)
        torch.save({"model_state_dict": self.state_dict(), "parameters": self.get_parameters()}, path)

    @classmethod
    def load(cls, load_path, device: str = "cpu"):
        """Load model and return it as a class object.

        Parameters
        ----------
        load_path : str
            Path to model's save file.
        device : str, optional (default='cpu')
            Device identification.

        Returns
        -------
        :class:'~lib.core.Model' :
            Object of this :class:'~lib.core.Model'

        """
        checkpoint = torch.load(load_path, map_location=device)
        parameters = checkpoint["parameters"]
        if "device" in parameters.keys():
            parameters["device"] = device
        model = cls(**parameters).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model
