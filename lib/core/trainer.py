"""Abstract class for trainer implementation."""

from __future__ import annotations

import os
import pickle
import random
import shutil
import time
import traceback
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from lib import logging  # noqa: F401
from lib import losses  # noqa: F401
from lib import models  # noqa: F401
from lib.core import BaseDataset, BaseLogger, BaseModel
from lib.utils import AbstractYAMLMeta, get_instance

LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))


class Impatient(Exception):
    """Signal for training interruption by impatience."""

    pass


class BaseTrainer(yaml.YAMLObject, ABC, metaclass=AbstractYAMLMeta):
    """Abstract trainer.

    The following methods have to implemented and must return dict with losses and metrics:
    - train_on_batch
    - validate_on_batch
    Note that the loss key is mandatory.

    Trainer automatically saves returned loss and metrics for every batch and mean loss for every epoch.

    The following methods are called during initialization:
    - configure_loss
    - configure_optimizers
    - configure_schedulers
    Order of calling is the same as above. Use it to initialize losses, optimizers and schedulers
    or skip it and pass them to the child constructor directly.

    The following methods are called before and after training procedure:
    - on_train_start
    - on_train_end

    The following methods are called before and after validation procedure:
    - on_validation_start
    - on_validation_end

    Parameters
    ----------
    epoch : int
        Current epoch.
    train_batches_seen : int
        How many training batches seen.
    curr_epoch_train_batches_seen : int
        How many training batches seen during current epoch.
    curr_epoch_valid_batches_seen : int
        How many validation batches seen during current epoch.
    train_iters_per_epoch : int
        How many iterations/batches processed for one training epoch.
    valid_iters_per_epoch : int
        How many iterations/batches processed for one validation.
    train_scores : Dict[str, List[float]]
        Dict with training metrics and loss for every seen training batch.
    valid_scores : Dict[str, List[float]]
        Dict with validation metrics and loss for every seen validation batch.

    Examples
    --------
    >>> class MyAwesomePipeline(BaseTrainer):
    ...
    ...     yaml_tag = "!MyAwesomePipeline"
    ...
    ...     def __init__(
    ...         criterion: BaseLoss,
    ...         class_names: List[str],
    ...         *args,
    ...         **kwargs,
    ...     ) -> None:
    ...
    ...         self.criterion = criterion
    ...
    ...
    ...         self.class_names = class_names
    ...
    ...         super().__init__(*args, **kwargs)
    ...
    ...     def get_parameters(self) -> Dict[str, Any]:
    ...         params = super().get_parameters()
    ...
    ...         params.update(
    ...             {
    ...                 "criterion": self.criterion,
    ...                 "class_names": self.class_names,
    ...             }
    ...         )
    ...
    ...         return params
    ...
    ...     def configure_optimizers(self) -> None:
    ...         train_params = self.model.parameters()
    ...
    ...         self.optimizer = get_instance(self.optimizer_cfg["class"])(
    ...             params=train_params, **self.optimizer_cfg["params"]
    ...         )
    ...
    ...     def configure_schedulers(self) -> None:
    ...
    ...         self.scheduler = get_instance(self.scheduler_cfg["class"])(
    ...             optimizer=self.optimizer,
    ...             num_epochs=self.max_epochs,
    ...             iters_per_epoch=self.train_iters_per_epoch,
    ...             **self.scheduler_cfg["params"],
    ...         )
    ...
    ...     def train_on_batch(self, image: Tensor, target: List[Any]) -> Dict[str, float]:
    ...
    ...         self.scheduler(self.curr_epoch_train_batches_seen, self.epoch)
    ...
    ...         x = image.to(self.device)
    ...         y_true = torch.stack([val["mask"] for val in target], dim=0).to(self.device)
    ...
    ...         self.optimizer.zero_grad()
    ...
    ...         logit = self.model(x)
    ...         loss = self.criterion(logit, y_true.long())
    ...         loss.backward()
    ...         self.optimizer.step()
    ...
    ...        return {"loss": loss.item()}
    ...
    ...     def validate_on_batch(self, image: Tensor, target: List[Any]) -> Dict[str, float]:
    ...
    ...         x = image.to(self.device)
    ...         y_true = torch.stack([val["mask"] for val in target], dim=0).to(self.device)
    ...
    ...         with torch.no_grad():
    ...             logit = self.model(x)
    ...
    ...         loss = self.criterion(logit, y_true.long())
    ...
    ...         pred_numpy = logit.data.cpu().numpy()
    ...         y_true_numpy = y_true.cpu().numpy()
    ...         pred_classes = np.argmax(pred_numpy, axis=1)
    ...
    ...         self.evaluator.add_batch(y_true_numpy, pred_classes)
    ...
    ...         return {"loss": loss.item()}
    ...
    ...     def on_validation_start(self) -> None:
    ...         self.evaluator.reset()
    ...
    ...     def on_validation_end(self):
    ...         metric_1 = self.evaluator.get_metric_1()
    ...         metric_2 = self.evaluator.get_metric_2()
    ...
    ...         metric_by_class_1 = self.evaluator.get_metric_by_class_1()
    ...         metric_by_class_2 = self.evaluator.get_metric_by_class_2()
    ...
    ...         self.logger.log(
    ...             {"metric_1": metric_1, "metric_2": metric_2},
    ...             scope="epoch",
    ...             stage="valid",
    ...             epoch_step=self.epoch,
    ...         )
    ...
    ...         for i, (class_name, metric_by_class_1_value, metric_by_class_2_value) in enumerate(
    ...             zip(self.class_names, metric_by_class_1, metric_by_class_2)
    ...         ):
    ...             self.logger.log(
    ...                 {class_name: metric_by_class_1_value},
    ...                 scope="epoch",
    ...                 stage="Group of metrics 1 name",
    ...                 epoch_step=self.epoch,
    ...             )
    ...
    ...             self.logger.log(
    ...                 {class_name: metric_by_class_2_value},
    ...                 scope="epoch", stage="Group of metrics 2 name",
    ...                 epoch_step=self.epoch,
    ...             )
    ...
    ...     def on_train_end(self) -> None:
    ...
    ...         train_loss_sum_epoch = torch.sum(
    ...             torch.stack(
    ...                 self.train_scores["loss"][
    ...                     -self.curr_epoch_train_batches_seen :
    ...                 ]
    ...             )
    ...         )
    ...
    ...         self.logger.log(
    ...             {"loss_sum": train_loss_sum_epoch},
    ...             scope="epoch",
    ...             stage="train",
    ...             epoch_step=self.epoch,
    ...         )

    """

    yaml_loader = yaml.Loader
    yaml_dumper = yaml.Dumper

    def __subclasshook__(cls, subclass):
        """Check present methods."""
        return (
            hasattr(subclass, "train_on_batch")  # noqa: W503
            and callable(subclass.train_on_batch)  # noqa: W503
            and hasattr(subclass, "validate_on_batch")  # noqa: W503
            and callable(subclass.validate_on_batch)  # noqa: W503
        )

    @classmethod
    def from_yaml(cls, loader, node):
        """Load object from YAML configuration."""
        initial_parameters = loader.construct_mapping(node, deep=True)
        return cls(**initial_parameters)

    @classmethod
    def to_yaml(cls, dumper, obj):
        """Dump object to YAML configuration."""
        return dumper.represent_mapping(cls.yaml_tag, obj.get_parameters())

    def __init__(
        self,
        model: Union[BaseModel, DDP],
        train_dataset: BaseDataset,
        valid_dataset: BaseDataset,
        test_dataset: BaseDataset,
        example_dataset: BaseDataset,
        logger: BaseLogger,
        device: Union[str, int],
        save_folder: str,
        shuffle_train: bool = False,
        loss_cfg: Dict[str, Any] = {"class": "torch.nn.CrossEntropyLoss", "params": {}},
        optimizer_cfg: Dict[str, Any] = {"class": "torch.optim.Adam", "params": {}},
        scheduler_cfg: Dict[str, Any] = {"class": "torch.optim.lr_scheduler.ReduceLROnPlateau", "params": {}},
        train_batch_size: int = 4,
        valid_batch_size: int = 1,
        max_epochs: Optional[int] = None,
        start_epoch_num: int = 0,
        max_batches: Optional[int] = None,
        validate_first: bool = False,
        validation_patience: int = None,
        val_every_n_epochs: int = None,
        val_every_n_batches: int = None,
        log_every_n_batches: int = 10,
        save_model_every_val: bool = False,
        num_workers: int = 8,
        seed: int = 34,
        pretrained_path: str = None,
        *args: tuple,
        **kwargs: Dict[str, Any],
    ) -> None:
        """Initialize trainer.

        Parameters
        ----------
        model: BaseModel
            Model to be trained.
        train_dataset: BaseDataset
            Train dataset.
        valid_dataset: BaseDataset
            Validation dataset.
        test_dataset: BaseDataset
            Test dataset.
        example_dataset: BaseDataset
            Example dataset.
        logger: BaseLogger
            Logger.
        device: Union[str, int, device]
            Device.
        save_folder: str
            Folder to save models.
        shuffle_train: bool, optional (default=False)
            Whether to shuffle train data before every epoch.
        loss_cfg: Dict[str, Any], optional (default={"class": "torch.nn.CrossEntropyLoss", "params": {}})
            Config to initialize loss for model.
        optimizer_cfg: Dict[str, Any], optional (default={"class": "torch.optim.Adam", "params": {}})
            Config to initialize optimizer for model.
        scheduler_cfg: Dict[str, Any]],
                        optional (default={"class": "torch.optim.lr_scheduler.ReduceLROnPlateau", "params": {}})
            Config to initialize scheduler for model.
        train_batch_size: int, optional (default=4)
            Train batch size.
        valid_batch_size: int, optional (default=1)
            Validation batch size.
        max_epochs: int, optional (default=3)
            Max epochs to train.
        start_epoch_num: int, optional (default=0)
            Start epoch.
        max_batches: int, optional (default=None)
            Max batches to train on.
        validate_first: bool, optional (default=False)
            Whether to validate before training.
        validation_patience: int, optional (default=None)
            Patience for validation loss.
        val_every_n_epochs: int, optional (default=None)
            Validate every n epochs.
        val_every_n_batches: int, optional (default=None)
            Validate every n batches.
        log_every_n_batches: int, optional (default=10)
            Log every n batches.
        save_model_every_val: bool, optional (default=False)
            Save model for every validation.
        num_workers: int, optional (default=8)
            Num workers for dataloaders.
        seed: int, optional (default=34)
            Seed.
        pretrained_path: str, optional (default=None)
            Model pretrained weights.

        Returns
        -------
        None
        """
        super().__init__()

        if LOCAL_RANK != -1:
            if torch.cuda.device_count() <= LOCAL_RANK:
                raise ValueError("insufficient CUDA devices for DDP command")
            torch.cuda.set_device(LOCAL_RANK)
            self.device = torch.device("cuda", LOCAL_RANK)
            dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")
        else:
            self.device = torch.device(device)
            if device != "cpu":
                torch.cuda.set_device(self.device)

        self.save_folder = save_folder

        self.model = model
        self.model.to(self.device)

        if LOCAL_RANK != -1:
            self.model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)

        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.example_dataset = example_dataset
        self.shuffle_train = shuffle_train

        self.logger = logger

        self.num_workers = num_workers
        self.seed = seed
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.pretrained_path = pretrained_path

        self.validate_first = validate_first
        self.validation_number = 0 if validate_first else 1
        self.validation_patience = validation_patience
        self.val_every_n_epochs = val_every_n_epochs
        self.val_every_n_batches = val_every_n_batches
        self.max_epochs = max_epochs
        self.max_batches = max_batches

        if self.max_epochs is None and self.max_batches is None:
            raise ValueError("Max epochs or max batches schould be specified!")

        if self.max_epochs is not None and self.max_batches is not None:
            raise ValueError("Only max epochs or max batches schould be specified!")

        self.start_epoch_num = start_epoch_num
        self.log_every_n_batches = log_every_n_batches
        self.save_model_every_val = save_model_every_val

        self.loss_cfg = loss_cfg
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg

        # Load pretrained dict
        if pretrained_path is not None:
            print(f"Use pretrained model from {pretrained_path}.")
            pretrained = torch.load(pretrained_path, map_location="cpu")
            pretrained_dict = pretrained["model_state_dict"]
            self.model.load_state_dict(pretrained_dict, strict=False)
        self._initialize_training()
        os.makedirs(self.save_folder, exist_ok=True)
        self.early_stopping = EarlyStopping(patience=validation_patience, min_delta=0)

    def _initialize_training(self) -> None:
        """Initialize training procedure."""
        self.epoch = self.start_epoch_num
        self.train_batches_seen = 0
        self.examples = 0
        self.start_time = 0.0

        self.fix_seeds()

        if not hasattr(self, "train_scores"):
            self.train_scores: Dict[str, List[float]] = defaultdict(list)
        if not hasattr(self, "valid_scores"):
            self.valid_scores: Dict[str, List[float]] = defaultdict(list)

        self.train_iterator, self.valid_iterator, self.test_iterator = self._build_iterators()

        self.train_iters_per_epoch = len(self.train_iterator)
        self.valid_iters_per_epoch = len(self.valid_iterator)

        self.configure_loss()
        self.configure_optimizers()
        self.configure_schedulers()

        self.curr_epoch_train_batches_seen = 0
        self.curr_epoch_valid_batches_seen = 0

    def _build_iterators(self) -> Tuple[DataLoader, ...]:
        if LOCAL_RANK != -1:
            sampler: Optional[DistributedSampler] = DistributedSampler(
                dataset=self.train_dataset, num_replicas=WORLD_SIZE, rank=LOCAL_RANK, shuffle=self.shuffle_train
            )
        else:
            sampler = None

        train_iterator = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_batch_size // WORLD_SIZE,
            num_workers=self.num_workers,
            sampler=sampler,
            pin_memory=True,
            shuffle=self.shuffle_train and sampler is None,
            collate_fn=self.train_dataset.collate_fn,
        )

        valid_iterator = DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.valid_batch_size,
            num_workers=self.num_workers,
        )

        test_iterator = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.valid_batch_size,
            num_workers=self.num_workers,
        )

        return train_iterator, valid_iterator, test_iterator

    def get_parameters(self) -> Dict[str, Any]:
        """Get parameters of trainer for serialization.

        Returns
        -------
        Dict[str, Any]
            Parameters of pipeline.
        """
        params: Dict[str, Any] = {
            "model": self.model,
            "train_dataset": self.train_dataset,
            "valid_dataset": self.valid_dataset,
            "test_dataset": self.test_dataset,
            "example_dataset": self.example_dataset,
            "logger": {
                "class": f"{self.logger.__class__.__module__}.{self.logger.__class__.__name__}",
                "params": self.logger.get_parameters(),
            },
            "device": self.device,
            "save_folder": self.save_folder,
            "shuffle_train": self.shuffle_train,
            "loss_cfg": self.loss_cfg,
            "optimizer_cfg": self.optimizer_cfg,
            "scheduler_cfg": self.scheduler_cfg,
            "train_batch_size": self.train_batch_size,
            "valid_batch_size": self.valid_batch_size,
            "max_epochs": self.max_epochs,
            "start_epoch_num": self.start_epoch_num,
            "max_batches": self.max_batches,
            "validate_first": self.validate_first,
            "validation_patience": self.validation_patience,
            "val_every_n_epochs": self.val_every_n_epochs,
            "val_every_n_batches": self.val_every_n_batches,
            "log_every_n_batches": self.log_every_n_batches,
            "save_model_every_val": self.save_model_every_val,
            "num_workers": self.num_workers,
            "seed": self.seed,
            "pretrained_path": self.pretrained_path,
        }
        return params

    def fix_seeds(self) -> None:
        """Fix random seeds."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _save_scores(self, scores: Dict[str, float], stage: str = "train") -> None:
        """Add batch scores to pipeline scores dict.

        Parameters
        ----------
        scores: Dict[str, float]
            Batch scores.
        stage: str, optional (default="train")
            Stage of training: ("train", "valid")

        """
        if stage == "train":
            for score_name in scores.keys():
                self.train_scores[score_name].append(scores[score_name])
        elif stage == "valid":
            for score_name in scores.keys():
                self.valid_scores[score_name].append(scores[score_name])
        else:
            raise ValueError(f"Unknown stage={stage}, available: ('train', 'valid')")

    def train_on_batches(self) -> None:
        """Run training iteration on dataset."""
        self.on_train_start()
        self.model.train()

        self.curr_epoch_train_batches_seen = 0

        pbar = enumerate(self.train_iterator)

        if LOCAL_RANK in (-1, 0):

            pbar = tqdm(pbar, total=len(self.train_iterator), bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")

        for _, data in pbar:

            batch_result: Dict[str, float] = self.train_on_batch(data)

            self.train_batches_seen += 1
            self.curr_epoch_train_batches_seen += 1

            if LOCAL_RANK in (-1, 0):

                self._save_scores(batch_result, stage="train")

                if self.max_batches is not None:
                    desc = f"Iter: {self.train_batches_seen}/{self.max_batches} batch_loss={batch_result['loss']:.4f}"
                else:
                    desc = f"Epoch: {self.epoch}/{self.max_epochs} batch_loss={batch_result['loss']:.4f}"

                cast(tqdm, pbar).set_description(desc)

                if self.log_every_n_batches is not None and self.train_batches_seen % self.log_every_n_batches == 0:
                    self.logger.log(  # type: ignore
                        batch_result,
                        scope="batch",
                        stage="train",
                        batch_step=self.train_batches_seen,
                    )
                if self.val_every_n_batches is not None and self.train_batches_seen % self.val_every_n_batches == 0:
                    self.validate()

            if self.max_batches is not None and self.train_batches_seen >= self.max_batches:
                raise Impatient

        if LOCAL_RANK in (-1, 0):
            train_loss_epoch = np.mean(self.train_scores["loss"][-self.curr_epoch_train_batches_seen:])

            self.logger.log(  # type: ignore
                {"loss": train_loss_epoch},
                scope="epoch",
                stage="train",
                epoch_step=self.epoch,
            )
            self.on_train_end()

    def validate(self) -> None:
        """Run validation iteration."""
        self.on_validation_start()
        self.model.eval()

        self.curr_epoch_valid_batches_seen = 0
        for data in tqdm(self.valid_iterator):
            batch_result: Dict[str, float] = self.validate_on_batch(data)
            self._save_scores(batch_result, stage="valid")
            self.curr_epoch_valid_batches_seen += 1

        valid_loss_epoch = np.mean(self.valid_scores["loss"][-self.curr_epoch_valid_batches_seen:])

        self.logger.log(  # type: ignore
            {"loss": valid_loss_epoch},
            scope="epoch",
            stage="valid",
            epoch_step=self.epoch,
        )

        if not self.early_stopping(valid_loss_epoch):
            self.save(save_policy="best")
        else:
            raise Impatient
        if self.save_model_every_val:
            self.save(save_policy="validate")

        self.validation_number += 1

        self.on_validation_end()

    def train(self) -> None:
        """Train models."""
        try:
            self.start_time = time.time()

            if self.validate_first and LOCAL_RANK in (-1, 0):
                self.validate()

            while True:

                if LOCAL_RANK != -1:
                    cast(DistributedSampler, self.train_iterator.sampler).set_epoch(self.epoch)

                self.train_on_batches()

                # validate
                if (
                    self.val_every_n_epochs is not None
                    and self.epoch % self.val_every_n_epochs == 0  # noqa: W503
                    and LOCAL_RANK in (-1, 0)  # noqa: W503
                ):
                    self.validate()

                self.epoch += 1

                if self.max_epochs and self.epoch >= self.max_epochs:
                    break

        except KeyboardInterrupt:
            print("Training interrupted.")
        except Impatient:
            print("Training is finished by impatient!")
        except Exception:
            traceback.print_exc()
        finally:
            if LOCAL_RANK in (-1, 0):
                self.save(save_policy="last")

        if self.validation_number < 1 and LOCAL_RANK in (-1, 0):
            self.save(save_policy="best")

    def get_scores(self) -> Dict[str, Dict[str, List[float]]]:
        """Get training scores.

        Returns
        -------
        Dict[str, Dict[str, List[float]]]
            Training and validation scores for every batch.

        """
        scores = {
            "train_scores": self.train_scores,
            "valid_scores": self.valid_scores,
        }
        return scores

    def set_scores(self, scores: Dict[str, Dict[str, List[float]]]) -> None:
        """Load training scores.

        Parameters
        ----------
        scores: Dict[str, Dict[str, List[float]]]
            Training and validation scores for every batch.

        """
        self.train_scores = scores["train_scores"]
        self.valid_scores = scores["valid_scores"]

    def save(self, save_policy: str = "best") -> None:
        """Save trainer state.

        Parameters
        ----------
        save_policy: str, optional (default="best")
            Which model to save: ('best', 'last', 'validate').

        """
        print(f"Saving trainer to {self.save_folder}.")

        model_to_save = self.model.module if type(self.model) is DDP else self.model

        model_to_save = cast(BaseModel, model_to_save)

        if save_policy == "best":
            model_to_save.save(os.path.join(self.save_folder, "model"))
        elif save_policy == "last":
            model_to_save.save(os.path.join(self.save_folder, "model_last"))
        elif save_policy == "validate":
            model_to_save.save(os.path.join(self.save_folder, f"model_{self.validation_number}"))

        torch.save(
            {"parameters": self.get_parameters()},
            os.path.join(self.save_folder, "trainer"),
        )
        pickle.dump(
            self.get_scores(),
            open(os.path.join(self.save_folder, "scores.pkl"), "wb"),
        )
        print("Trainer is saved.")

    @classmethod
    def load(cls, load_folder: str, device: str = "cpu", load_policy: str = "best") -> BaseTrainer:
        """Load trainer from checkpoint.

        Parameters
        ----------
        load_folder: str
            Folder with checkpoint.
        device: str, optional (default="cpu")
            Device to place model on.
        load_policy: str, optional (default="best")
            Which checkpoint to load: ("best", "last").

        Returns
        -------
        BaseTrainer
            Trainer instance.

        """
        scores = pickle.load(open(os.path.join(load_folder, "scores.pkl"), "rb"))

        checkpoint = torch.load(os.path.join(load_folder, "trainer"), map_location=device)
        parameters = checkpoint["parameters"]
        parameters.pop("device", None)

        logger_config = parameters.pop("logger")
        logger = get_instance(logger_config["class"])(**logger_config["params"])

        trainer = cls(device=device, logger=logger, **parameters)
        trainer.set_scores(scores)

        if load_policy == "best":
            trainer.model = cast(BaseModel, trainer.model).load(os.path.join(load_folder, "model"), device)
        elif load_policy == "last":
            trainer.model = cast(BaseModel, trainer.model).load(os.path.join(load_folder, "model_last"), device)
        return trainer

    @abstractmethod
    def train_on_batch(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Train model on batch, saves training loss and metrics in output dictionary.

        Parameters
        ----------
        data: Dict[str, Any]
            Input data.

        Returns
        -------
        Dict[str, float]
            Dictionary with loss and train metrics. "loss" key is mandatory.

        """
        raise NotImplementedError

    @abstractmethod
    def validate_on_batch(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Validate model on batch, saves valid loss and metrics in output dictionary.

        Parameters
        ----------
        data: Dict[str, Any]
            Input data.

        Returns
        -------
        Dict[str, float]
            Dictionary with loss and validaton metrics. "loss" key is mandatory.

        """
        raise NotImplementedError

    def configure_optimizers(self) -> None:
        """Initialize model optimizers."""
        if self.optimizer_cfg is not None:
            self.optimizer = get_instance(self.optimizer_cfg["class"])(
                params=self.model.parameters(), **self.optimizer_cfg["params"]
            )

    def configure_schedulers(self) -> None:
        """Initialize learning rate schedulers."""
        if self.optimizer_cfg is not None and self.scheduler_cfg is not None:
            self.scheduler = get_instance(self.scheduler_cfg["class"])(
                optimizer=self.optimizer,
                **self.scheduler_cfg["params"],
            )

    def configure_loss(self) -> None:
        """Initialize loss functions for training."""
        if self.loss_cfg is not None:
            self.loss = get_instance(self.loss_cfg["class"])(**self.loss_cfg["params"])

    def on_validation_start(self) -> None:
        """Call function before validation starts."""
        pass

    def on_validation_end(self) -> None:
        """Call function after validation ends."""
        pass

    def on_train_start(self) -> None:
        """Call function before training starts."""
        pass

    def on_train_end(self) -> None:
        """Call function after training ends."""
        pass

    def profile(self, wait: int = 1, warmup: int = 1, active: int = 3, repeat: int = 2) -> None:
        """Profile trainer.

        Parameters
        ----------
        wait: int, optional (default=1)
            Steps to skip. During wait steps, the profiler is disabled.
        warmup: int, optional (default=1)
            Steps to warmup. During warmup steps, the profiler starts tracing but the results are discarded.
        active: int, optional (default=3)
            Steps to record. During active steps, the profiler works and records events.
        repeat: int, optional (default=2)
            How many times to repeat cycle: wait->warmup->active.

        """
        profile_log_path = os.path.join(self.save_folder, "./log")

        if os.path.exists(profile_log_path):
            shutil.rmtree(profile_log_path)

        with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_log_path),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:

            total_steps = (wait + warmup + active) * repeat

            for step, (data) in enumerate(tqdm(self.train_iterator, desc="Profile trainer", total=total_steps)):
                if step >= total_steps:
                    break

                _ = self.train_on_batch(data)

                prof.step()


class EarlyStopping:
    """Early stopping criterion implementation."""

    def __init__(
        self,
        patience: Optional[int] = None,
        min_delta: float = 0,
    ) -> None:
        """Initialize early stopping.

        Parameters
        ----------
        patience: int, optional (default=None)
            Patience for validation epochs.
        min_delta: float, optional (default=0)
            Minimal improvement delta.

        """
        self.counter = 0
        self.min_delta = min_delta
        self.patience = patience
        self.best_score = 0.0

    def __call__(self, score: float) -> bool:
        """Check whether to stop training.

        Parameters
        ----------
        score: float
            Current validation epoch loss.

        Returns
        -------
        bool
            Whether to stop training.
        """
        if not self.best_score:
            self.best_score = score

        if self.patience is None:
            return False

        if self.best_score - score <= self.min_delta:
            self.counter += 1

            if self.counter >= self.patience:
                return True
        else:
            print(f"Model improved on valid. previous best: {self.best_score}, current: {score}")
            self.best_score = score
            self.counter = 0

        return False
