"""Core Package.

# package core
Core package contains base classes for models, loss, training procedure pipeline,
classes for testing models

# class BaseModel
Abstract class for deep learning models

# class BaseLoss
Abstract class for model losses

# class BaseMetric
Abstract class for model metrics

# class BaseDataset
Abstract class for datasets

# class BaseTransform
Abstract class for transforms

# class BaseTrainer
Abstract class for training models

# class BaseLogger
Abstract class for training loggers

# class BaseEvaluator
Abstract class for evaluating models

"""

from .dataset import BaseDataset
from .logger import BaseLogger
from .loss import BaseLoss
from .metric import BaseMetric
from .model import BaseModel
from .trainer import BaseTrainer
from .transform import BaseTransform
from .evaluator import BaseEvaluator

__all__ = [
    "BaseModel",
    "BaseLoss",
    "BaseMetric",
    "BaseDataset",
    "BaseTransform",
    "BaseTrainer",
    "BaseLogger",
    "BaseEvaluator",
]
