"""Provides abstract class for evaluator."""

from __future__ import annotations

import os
import pickle
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from matplotlib import cm
from torch import device
from torch.utils.data import DataLoader
from tqdm import tqdm

from lib import data  # noqa: F401
from lib import metrics  # noqa: F401
from lib import models  # noqa: F401
from lib.utils import AbstractYAMLMeta

from .dataset import BaseDataset
from .metric import BaseMetric
from .model import BaseModel


class BaseEvaluator(ABC, yaml.YAMLObject, nn.Module, metaclass=AbstractYAMLMeta):
    """Abstract class for deep learning models.

    Attributes
    ----------
    self.samples_metrics : List[Dict[str, float]]
        List of evaluation metrics for every sample in test dataset.
    self.samples_predict : List[Dict[str, Any]]
        List of infer prediction from model for every sample in test dataset.
    self.metric_names : List[str]
        List of used evaluation metrics.
    self.keys_titles : Dict[str, str]
        Dict of used titles for graphics.

    """

    yaml_loader = yaml.Loader
    yaml_dumper = yaml.Dumper

    @classmethod
    def from_yaml(cls, loader, node):
        """Create object from yaml node."""
        info = loader.construct_mapping(node, deep=True)
        return cls(**info)

    @classmethod
    def to_yaml(cls, dumper, node):
        """Serialize object to yaml."""
        return dumper.represent_mapping(cls.yaml_tag, node.get_parameters())

    def __init__(
        self,
        metrics_list: List[BaseMetric],
        device: Union[str, int, device],
        model: BaseModel,
        test_dataset: BaseDataset,
        save_folder: str,
        load_policy: str = "best",
    ) -> None:
        """Initialize evaluator.

        Parameters
        ----------
        model : BaseModel
            Model to be trained.
        test_dataset : BaseDataset
            Test dataset.
        metrics_list : List[BaseMetric]
            Metrics for evaluation.
        device : Union[str, int, List[Union[str, int]]]
            Device.
        save_folder : str
            Folder to save models.
        load_policy : str, optional (default="best")
            Which checkpoint to load: ("best", "last").

        """
        super().__init__()
        self.device = torch.device(device)
        self.test_dataset = test_dataset
        self.data_iterator = DataLoader(self.test_dataset, batch_size=1, num_workers=1, shuffle=False)

        self.model = model
        if load_policy == "best":
            self.model = self.model.load(os.path.join(save_folder, "model"), self.device)
            self.save_folder = os.path.join(save_folder, "evaluation")
            os.makedirs(self.save_folder, exist_ok=True)
        elif load_policy == "last":
            self.model = self.model.load(os.path.join(save_folder, "model_last"), self.device)
            self.save_folder = os.path.join(save_folder, "evaluation_last")
            os.makedirs(self.save_folder, exist_ok=True)

        self.samples_metrics: List[Dict[str, Any]] = []
        self.samples_predict: List[Any] = []
        self.metrics_list = metrics_list
        self.metric_names = [type(metric).__name__ for metric in metrics_list]
        self.keys_titles = {"class": "class", "bbox size": "bbox size"}

        self.samples_metrics_by_group: Dict[str, Any] = {}
        self.unique_values: Dict[str, Any] = {}

    def get_parameters(self) -> Dict[str, Any]:
        """Get parameters of evaluator.

        Returns
        -------
        Dict[str, Any] :
            Parameters of evaluator.

        """
        parameters = {
            "metrics_list": self.metrics_list,
            "device": self.device,
            "model": self.model,
            "test_dataset": self.test_dataset,
            "save_folder": self.test_dataset,
        }
        return parameters

    def evaluate(self):
        """Evaluate segmentation model."""
        self.model.eval()
        with torch.no_grad(), tqdm(total=len(self.data_iterator)) as bar:
            for image, target in self.data_iterator:
                sample_info = dict()
                sample_info["model"] = dict()
                with torch.no_grad():
                    pred = self.model.infer(image.to(self.device))

                for metric_name, metric in zip(self.metric_names, self.metrics_list):
                    sample_info["model"][metric_name] = metric(pred, target["mask"].to(self.device))

                sample_info["class"] = [label.item() for label in target["mask"].unique()]

                self.samples_predict.append(pred)
                self.samples_metrics.append(sample_info)
                bar.set_description("Number of processed audios")
                bar.update(1)
        pickle.dump(self.samples_metrics, open(os.path.join(self.save_folder, "samples_metrics.pkl"), "wb"))
        pickle.dump(self.samples_predict, open(os.path.join(self.save_folder, "samples_predict.pkl"), "wb"))

    def process_metrics(self, by: Tuple[str, ...] = ("class", "bbox size")) -> None:
        """Split metrics by group.

        Parameters
        ----------
        by : tuple, optional (default = ("class", "bbox size"))
            Groups of metrics.

        """
        if len(self.samples_metrics) == 0:
            warnings.warn("Metrics should be computed before by evaluate function.", UserWarning)
        else:
            for group_name in by:
                unique_values = []
                for sample_metrics in self.samples_metrics:
                    for uniq in sample_metrics[group_name]:
                        unique_values.append(uniq)
                unique_values = list(set(unique_values))
                self.unique_values[group_name] = unique_values
                mean_by_group: Dict[str, Any] = dict()
                mean_by_group["model"] = {value: defaultdict(list) for value in unique_values}
                mean_by_group["model"]["all"] = defaultdict(list)
                for sample_metrics in self.samples_metrics:
                    for group_value in sample_metrics[group_name]:
                        for metric_name, value in sample_metrics["model"].items():
                            mean_by_group["model"][group_value][metric_name].append(value)
                            mean_by_group["model"]["all"][metric_name].append(value)
                for key, value in mean_by_group.items():
                    mean_by_group[key] = dict(value)
                self.samples_metrics_by_group[group_name] = mean_by_group
            pickle.dump(
                self.samples_metrics_by_group,
                open(os.path.join(self.save_folder, "samples_metrics_by_groups.pkl"), "wb"),
            )

    def get_plots(self, cmap_name: str = "hsv", by: Tuple[str, ...] = ("class", "bbox size")) -> None:  # noqa: C901
        """Plot metrics by groups.

        Parameters
        ----------
        cmap_name : str, optional (default = "hsv")
            Color map name.
        by : tuple, optional (default = ("class", "bbox size"))
            Groups of metrics.

        """
        if len(self.samples_metrics) == 0:
            warnings.warn("Metrics should be computed before by evaluate function.", UserWarning)
        else:
            if len(self.samples_metrics_by_group) == 0:
                self.process_metrics(by=by)
            cmap = cm.get_cmap(cmap_name)
            metrics_colors = {
                metric_name: cmap(1.0 * i / len(self.metric_names)) for i, metric_name in enumerate(self.metric_names)
            }
            save_pics = os.path.join(self.save_folder, "pics")
            for group_name in by:
                if group_name not in self.keys_titles:
                    self.keys_titles[group_name] = group_name
                group_title = self.keys_titles[group_name]
                name2label = sorted(list(self.unique_values[group_name]))
                name2label = dict(zip(name2label, range(len(name2label))))

                os.makedirs(save_pics, exist_ok=True)
                for metric_name in self.metric_names:
                    plt.figure(figsize=(15, 5))
                    for solution in self.samples_metrics_by_group[group_name].keys():
                        is_first = True
                        y = None
                        dy = None
                        x = None
                        for value in name2label:
                            metrics_v = self.samples_metrics_by_group[group_name][solution][value]
                            x_prev = x
                            y_prev = y
                            dy_prev = dy
                            x = name2label[value]
                            y = np.array(metrics_v[metric_name])
                            y = y[~np.isnan(y)].mean()
                            dy = np.array(metrics_v[metric_name])
                            dy = dy[~np.isnan(dy)].std()
                            if is_first:
                                plt.scatter(x, y, color=metrics_colors[metric_name], label=metric_name.upper())
                                is_first = False
                            else:
                                y_prev = np.array(y_prev)
                                y = np.array(y)
                                plt.scatter(x, y, color=metrics_colors[metric_name])
                                plt.plot([x_prev, x], [y_prev, y], color=metrics_colors[metric_name])
                                plt.fill_between(
                                    [x_prev, x],
                                    [y_prev - dy_prev, y - dy],
                                    [y_prev + dy_prev, y + dy],
                                    color=metrics_colors[metric_name],
                                    alpha=0.25,
                                )
                        metrics_all = self.samples_metrics_by_group[group_name][solution]["all"]
                        y = np.array(metrics_all[metric_name])
                        y = y[~np.isnan(y)].mean()
                        dy = np.array(metrics_all[metric_name])
                        dy = dy[~np.isnan(dy)].std()
                        plt.axhline(y=y, color="g", label=f"mean {metric_name.upper()}")
                        plt.fill_between(name2label.values(), y - dy, y + dy, color="g", alpha=0.1)

                    values = np.array(list(name2label.values()))

                    plt.xlim([values.min() - 0.5, values.max() + 1.0])
                    plt.xticks(values, name2label.keys(), fontsize=14)

                    plt.yticks(fontsize=14)

                    plt.grid()
                    plt.legend(loc="upper left", framealpha=0.5, fontsize=14)

                    plt.xlabel(f"{group_title}", fontsize=15)
                    plt.ylabel(f"{metric_name.upper()}", fontsize=15)
                    plt.title(f"Mean {metric_name.upper()} for each {group_title}", fontsize=17)
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_pics, f"{metric_name}_{group_name}.png"))
                    plt.close()

    @abstractmethod
    def infer(self):
        """Infer model."""
        raise NotImplementedError
