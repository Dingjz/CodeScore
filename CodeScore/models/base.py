# -*- coding: utf-8 -*-
# Copyright (C) 2020 Unbabel
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""
CometModel
========================
    Abstract Model class that implements some of the Pytorch Lightning logic.
    Extend this class to create new model and metrics within COMET.
"""
import abc
import logging
import os,sys
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as ptl
import torch
from torch.utils.data import DataLoader, RandomSampler, Subset

sys.path.insert(0, os.getcwd())
from encoders import str2encoder
from modules import LayerwiseAttention

from .lru_cache import tensor_lru_cache
from .pooling_utils import average_pooling, max_pooling
from .predict_pbar import PredictProgressBar
from .utils import OrderedSampler, Prediction, Target

if "COMET_EMBEDDINGS_CACHE" in os.environ:
    CACHE_SIZE = int(os.environ["COMET_EMBEDDINGS_CACHE"])
else:
    CACHE_SIZE = 1024


logger = logging.getLogger(__name__)


class CometModel(ptl.LightningModule, metaclass=abc.ABCMeta):
    """CometModel: Base class for all COMET models.

    Args:
        nr_frozen_epochs (Union[float, int]): Number of epochs (% of epoch) that the
            encoder is frozen. Defaults to 0.3.
        keep_embeddings_frozen (bool): Keeps the encoder frozen during training. Defaults
            to True.
        optimizer (str): Optimizer used during training. Defaults to 'AdamW'.
        encoder_learning_rate (float): Learning rate used to fine-tune the encoder model.
            Defaults to 1.0e-06.
        learning_rate (float): Learning rate used to fine-tune the top layers. Defaults
            to 1.5e-05.
        layerwise_decay (float): Learning rate % decay from top-to-bottom encoder layers.
            Defaults to 0.95.
        encoder_model (str): Encoder model to be used. Defaults to 'XLM-RoBERTa'.
        pretrained_model (str): Pretrained model from Hugging Face. Defaults to
            'xlm-roberta-large'.
        pool (str): Type of sentence level pooling (options: 'max', 'cls', 'avg').
            Defaults to 'avg'
        layer (Union[str, int]): Encoder layer to be used for regression ('mix'
            for pooling info from all layers). Defaults to 'mix'.
        layer_transformation (str): Transformation applied when pooling info from all
            layers (options: 'softmax', 'sparsemax'). Defaults to 'softmax'.
        layer_norm (bool): Apply layer normalization. Defaults to 'True'.
        loss (str): Loss function to be used. Defaults to 'mse'.
        dropout (float): Dropout used in the top-layers. Defaults to 0.1.
        batch_size (int): Batch size used during training. Defaults to 4.
        train_data (Optional[List[str]]): List of paths to training data. Each file is
            loaded consecutively for each epoch. Defaults to None.
        validation_data (Optional[List[str]]): List of paths to validation data.
            Validation results are averaged across validation set. Defaults to None.
    """

    def __init__(
        self,
        nr_frozen_epochs: Union[float, int] = 0.3,
        keep_embeddings_frozen: bool = True,
        optimizer: str = "AdamW",
        encoder_learning_rate: float = 1.0e-06,
        learning_rate: float = 1.5e-05,
        layerwise_decay: float = 0.95,
        encoder_model: str = "XLM-RoBERTa",
        pretrained_model: str = "xlm-roberta-large",
        pool: str = "avg",
        layer: Union[str, int] = "mix",
        layer_transformation: str = "softmax",
        layer_norm: bool = True,
        loss: str = "mse",
        dropout: float = 0.1,
        batch_size: int = 4,
        train_data: Optional[List[str]] = None,
        validation_data: Optional[List[str]] = None,
        class_identifier: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.encoder = str2encoder[self.hparams.encoder_model].from_pretrained(
            self.hparams.pretrained_model
        )

        self.epoch_nr = 0
        if self.hparams.layer == "mix":
            self.layerwise_attention = LayerwiseAttention(
                layer_transformation=layer_transformation,
                num_layers=self.encoder.num_layers,
                dropout=self.hparams.dropout,
                layer_norm=self.hparams.layer_norm,
            )
        else:
            self.layerwise_attention = None

        if self.hparams.nr_frozen_epochs > 0:
            self._frozen = True
            self.freeze_encoder()
        else:
            self._frozen = False

        if self.hparams.keep_embeddings_frozen:
            self.encoder.freeze_embeddings()

        self.nr_frozen_epochs = self.hparams.nr_frozen_epochs
        self.mc_dropout = False  # Flag used to control usage of MC Dropout
        self.caching = False  # Flag used to control Embedding Caching

        # If not defined here, metrics will not live in the same device as our model.
        self.init_metrics()

    def set_mc_dropout(self, value: int):
        """Sets Monte Carlo Dropout runs per sample.

        Args:
            value (int): number of runs per sample.
        """
        self.mc_dropout = value

    # @abc.abstractmethod
    # def read_training_data(self) -> List[dict]:
    #     """Abstract method that reads the training data.

    #     Returns:
    #         List[dict]: List with input samples in the form of a dict
    #     """
    #     pass

    # @abc.abstractmethod
    # def read_validation_data(self):
    #     """Abstract method that reads the validation data. If validation data
    #     has a columns 'system' we will output system-level accuracies for each
    #     validation dataset.

    #     Returns:
    #         List[dict]: List with input samples in the form of a dict
    #     """
    #     pass
    @abc.abstractclassmethod
    def read_file(self):
        pass

    @abc.abstractmethod
    def prepare_sample(
        self,
        sample: List[dict],
        stage: str = "fit",
        *args,
        **kwargs,
    ):
        """This method will be called by dataloaders to prepared data to input to the
        model.

        Args:
            sample (List[dict]): Batch of train/val/test samples.
            stage (str): model stage (options: 'fit', 'validate', 'test', or
                'predict'). Defaults to 'fit'.

        Returns:
            Model inputs and (optionally) training labels/targets.
        """
        pass

    @abc.abstractmethod
    def configure_optimizers(self):
        """Pytorch Lightning method to configure optimizers and schedulers."""
        pass

    @abc.abstractmethod
    def init_metrics(self) -> None:
        """Initializes train/validation metrics."""
        pass

    @abc.abstractmethod
    def forward(self, *args, **kwargs) -> Prediction:
        """Pytorch model forward method."""
        pass

    @abc.abstractmethod
    def requires_references(self) -> bool:
        """Whether or not this models work with references."""
        pass

    def freeze_encoder(self) -> None:
        """Deactivates training for encoder model parameters (keeping them frozen)"""
        logger.info("Encoder model frozen.")
        self.encoder.freeze()

    @property
    def loss(self):
        """Loss function"""
        return torch.nn.MSELoss()

    def compute_loss(self, prediction: Prediction, target: Target) -> torch.Tensor:
        """Computes Loss value between a batch Prediction and respective Target."""
        return self.loss(prediction.score, target.score)

    def unfreeze_encoder(self) -> None:
        """Activates fine-tuning of encoder parameters."""
        if self._frozen:
            if self.trainer.is_global_zero:
                logger.info("Encoder model fine-tuning")

            self.encoder.unfreeze()
            self._frozen = False
            if self.hparams.keep_embeddings_frozen:
                self.encoder.freeze_embeddings()

    def on_train_epoch_end(self) -> None:
        """Hook used to unfreeze encoder during training."""
        self.epoch_nr += 1
        if self.epoch_nr >= self.nr_frozen_epochs and self._frozen:
            self.unfreeze_encoder()
            self._frozen = False

    def set_embedding_cache(self):
        """Function that when called turns embedding caching on."""
        self.caching = True

    def get_sentence_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Function that extracts sentence embeddings for
        a single sentence and allows for caching embeddings.

        Args:
            tokens (torch.Tensor): sequences [batch_size x seq_len].
            attention_mask (torch.Tensor): attention_mask [batch_size x seq_len].
            token_type_ids (torch.Tensor): Model token_type_ids [batch_size x seq_len].
                Optional

        Returns:
            torch.Tensor [batch_size x hidden_size] with sentence embeddings.
        """
        if self.caching:
            return self.retrieve_sentence_embedding(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        else:
            return self.compute_sentence_embedding(
                input_ids,
                attention_mask,
                token_type_ids=token_type_ids,
            )

    @tensor_lru_cache(maxsize=CACHE_SIZE)
    def retrieve_sentence_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Wrapper for `get_sentence_embedding` function that caches results."""
        return self.compute_sentence_embedding(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

    def compute_sentence_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Function that extracts sentence embeddings for
        a single sentence.

        Args:
            tokens (torch.Tensor): sequences [batch_size x seq_len].
            attention_mask (torch.Tensor): attention_mask [batch_size x seq_len].
            token_type_ids (torch.Tensor): Model token_type_ids [batch_size x seq_len].
                Optional

        Returns:
            torch.Tensor [batch_size x hidden_size] with sentence embeddings.
        """
        encoder_out = self.encoder(
            input_ids, attention_mask, token_type_ids=token_type_ids
        )
        if self.layerwise_attention:
            embeddings = self.layerwise_attention(
                encoder_out["all_layers"], attention_mask
            )

        elif self.hparams.layer >= 0 and self.hparams.layer < self.encoder.num_layers:
            embeddings = encoder_out["all_layers"][self.hparams.layer]

        else:
            raise Exception("Invalid model layer {}.".format(self.hparams.layer))

        if self.hparams.pool == "default":
            sentemb = encoder_out["sentemb"]

        elif self.hparams.pool == "max":
            sentemb = max_pooling(
                input_ids, embeddings, self.encoder.tokenizer.pad_token_id
            )

        elif self.hparams.pool == "avg":
            sentemb = average_pooling(
                input_ids,
                embeddings,
                attention_mask,
                self.encoder.tokenizer.pad_token_id,
            )

        elif self.hparams.pool == "cls":
            sentemb = embeddings[:, 0, :]

        else:
            raise Exception("Invalid pooling technique.")

        return sentemb

    def training_step(
        self,
        batch: Tuple[dict, Target],
        batch_idx: int,
    ) -> torch.Tensor:
        """Pytorch Lightning training step.

        Args:
            batch (Tuple[dict, Target]): The output of your `prepare_sample` method.
            batch_idx (int): Integer displaying which batch this is.

        Returns:
            [torch.Tensor] Loss value
        """
        batch_input, batch_target = batch
        batch_prediction = self.forward(**batch_input)
        loss_value = self.compute_loss(batch_prediction, batch_target)

        if (
            self.nr_frozen_epochs < 1.0
            and self.nr_frozen_epochs > 0.0
            and batch_idx > self.first_epoch_total_steps * self.nr_frozen_epochs
        ):
            self.unfreeze_encoder()
            self._frozen = False

        self.log("train_loss", loss_value, on_step=True, on_epoch=True)
        return loss_value

    def validation_step(
        self,
        batch: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
        batch_nb: int,
        dataloader_idx: int,
    ) -> None:
        """Pytorch Lightning validation step. Runs model and logs metircs.

        Args:
            batch (Tuple[dict, Target]): The output of your `prepare_sample` method.
            batch_idx (int): Integer displaying which batch this is.
        """
        batch_input, batch_target = batch
        batch_prediction = self.forward(**batch_input)
        if dataloader_idx == 0:
            self.train_metrics.update(batch_prediction.score, batch_target["score"])

        elif dataloader_idx > 0:
            self.val_metrics[dataloader_idx - 1].update(
                batch_prediction.score,
                batch_target["score"],
                batch_target["system"] if "system" in batch_target else None,
            )

    def on_predict_start(self) -> None:
        """Called when predict begins to setup mc_dropout."""
        if self.mc_dropout:
            self.train()
        else:
            self.eval()

    def predict_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """Pytorch Lightning predict step.

        Args:
            batch (Tuple[dict, Target]): The output of your `prepare_sample` method.
            batch_idx (int): Integer displaying which batch this is.
            dataloader_idx (int): Integer displaying which dataloader this sample is
                coming from.

        Return:
            Predicion object
        """
        if self.mc_dropout:
            mcd_outputs = torch.stack([self(**batch) for _ in range(self.mc_dropout)])
            mcd_mean = mcd_outputs.mean(dim=0)
            mcd_std = mcd_outputs.std(dim=0)
            return mcd_mean, mcd_std
        return Prediction(scores=self(**batch).score)

    def validation_epoch_end(self, *args, **kwargs) -> None:
        """Computes and logs metrics."""
        self.log_dict(self.train_metrics.compute(), prog_bar=False)
        self.train_metrics.reset()

        val_metrics = []
        for i in range(len(self.hparams.validation_data)):
            results = self.val_metrics[i].compute()
            self.val_metrics[i].reset()
            # Log to tensorboard the results for this validation set.
            self.log_dict(results, prog_bar=False)
            val_metrics.append(results)

        average_results = {"val_" + k.split("_")[-1]: [] for k in val_metrics[0].keys()}
        for i in range(len(val_metrics)):
            for k, v in val_metrics[i].items():
                average_results["val_" + k.split("_")[-1]].append(v)

        self.log_dict(
            {k: sum(v) / len(v) for k, v in average_results.items()}, prog_bar=True
        )

    def setup(self, stage: str) -> None:
        """Data preparation function called before training by Lightning.

        stage (str): either 'fit', 'validate', 'test', or 'predict'
        """
        if stage in (None, "fit"):
            train_dataset = self.read_file(self.hparams.train_data[0])

            self.validation_sets = [
                self.read_file(d) for d in self.hparams.validation_data
            ]

            self.first_epoch_total_steps = len(train_dataset) // (
                self.hparams.batch_size * max(1, self.trainer.num_devices)
            )
            # Always validate the model with part of training.
            train_subset = np.random.choice(
                a=len(train_dataset), size=min(1000, int(len(train_dataset) * 0.2))
            )
            self.train_subset = Subset(train_dataset, train_subset)

    def train_dataloader(self) -> DataLoader:
        """Method that loads the train dataloader. Can be called every epoch to load a
        different trainset if `reload_dataloaders_every_n_epochs=1` in Lightning
        Trainer.
        """
        data_path = self.hparams.train_data[
            self.current_epoch % len(self.hparams.train_data)
        ]
        train_dataset = self.read_file(data_path)
        logger.info(f"Loading {data_path}.")

        return DataLoader(
            dataset=train_dataset,
            sampler=RandomSampler(train_dataset),
            batch_size=self.hparams.batch_size,
            collate_fn=lambda s: self.prepare_sample(s, stage="fit"),
            num_workers=2 * self.trainer.num_devices,
        )

    def val_dataloader(self) -> DataLoader:
        """Function that loads the validation sets."""
        val_data = [
            DataLoader(
                dataset=self.train_subset,
                batch_size=self.hparams.batch_size,
                collate_fn=lambda s: self.prepare_sample(s, stage="validate"),
                num_workers=2 * self.trainer.num_devices,
            )
        ]
        for validation_set in self.validation_sets:
            val_data.append(
                DataLoader(
                    dataset=validation_set,
                    batch_size=self.hparams.batch_size,
                    collate_fn=lambda s: self.prepare_sample(s, stage="validate"),
                    num_workers=2 * self.trainer.num_devices,
                )
            )
        return val_data

    def prepare_for_inference(self, sample):
        """This is to avoid having a lamba function inside the predict dataloader
        `collate_fn=lambda x: self.prepare_sample(x, inference=True)`
        """
        return self.prepare_sample(sample, stage="predict")

    def predict(
        self,
        samples: List[Dict[str, str]],
        batch_size: int = 16,
        gpus: int = 1,
        mc_dropout: int = 0,
        progress_bar: bool = True,
        accelerator: str = "auto",
        num_workers: int = None,
        length_batching: bool = True,
    ) -> Prediction:
        """Method that receives a list of samples (dictionaries with translations,
        sources and/or references) and returns segment-level scores, system level score
        and any other metadata outputed by COMET models. If `mc_dropout` is set, it
        also returns for each segment score, a confidence value.

        Args:
            samples (List[Dict[str, str]]): List with dictionaries with source,
                translations and/or references.
            batch_size (int): Batch size used during inference. Defaults to 16
            gpus (int): Number of GPUs to be used. Defaults to 1.
            mc_dropout (int): Number of inference steps to run using MCD. Defaults to 0
            progress_bar (bool): Flag that turns on and off the predict progress bar.
            accelarator (str): Pytorch Lightning accelerator (e.g: 'auto', 'gpu', 'cpu').
            num_workers (int): Number of workers to use when loading and preparing data
            length_batching (bool): If set to true, reduces padding by sorting samples
                by sequence length.

        Return:
            Prediction object with `scores`, `system_score` and any metadata returned
                by the model.
        """

        def restore_list_order(sorted_list, sort_ids):
            """Restores the original ids of a given list."""
            unsorted_list = [None for _ in range(len(sorted_list))]
            for i, s in zip(sort_ids, sorted_list):
                unsorted_list[i] = s
            return unsorted_list

        def flatten_metadata(metadata):
            metadata = Prediction(
                **{k: [dic[k] for dic in metadata] for k in metadata[0]}
            )
            for k, v in metadata.items():
                if torch.is_tensor(v[0]):
                    # If we have tensors we can use cat to flatten them.
                    metadata[k] = torch.cat(v, dim=0).tolist()
                else:
                    # for other predictions such as word tags we have to flatten the list.
                    metadata[k] = [item for sublist in v for item in sublist]
            return metadata

        # HACK: Workaround pytorch bug that prevents ParameterList to be used in DP
        # https://github.com/pytorch/pytorch/issues/36035
        if self.layerwise_attention is not None and gpus > 1:
            self.layerwise_attention.gamma_value = float(
                self.layerwise_attention.gamma[0]
            )
            self.layerwise_attention.weights = [
                float(parameter[0])
                for parameter in self.layerwise_attention.scalar_parameters
            ]

        # TODO: ideally this should be based on the actual token_ids
        # but that would require fundamentally changing the way dataloader is
        # setup, so currently raw chars are used as an approximation
        sampler = None
        if length_batching and gpus < 2:
            try:
                sort_ids = np.argsort([len(sample["source"]) for sample in samples])
            except KeyError:
                sort_ids = np.argsort([len(sample["golden_code"]) for sample in samples])
            sampler = OrderedSampler(sort_ids)

        if num_workers is None:
            num_workers = 2 * gpus

        self.eval()
        dataloader = DataLoader(
            dataset=samples,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=self.prepare_for_inference,
            num_workers=num_workers,
        )
        if gpus == 0:
            accelerator = "cpu"
        elif gpus == 1:
            accelerator = "gpu"
        else:
            accelerator = accelerator

        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=".*Consider increasing the value of the `num_workers` argument` .*",
        )
        if progress_bar:
            trainer = ptl.Trainer(
                devices=gpus if accelerator != "cpu" else "auto",
                logger=False,
                callbacks=[PredictProgressBar()],
                accelerator=accelerator,
                max_epochs=-1,
            )
        else:
            trainer = ptl.Trainer(
                devices=gpus if accelerator != "cpu" else "auto",
                logger=False,
                progress_bar_refresh_rate=0,
                accelerator=accelerator,
            )

        # TODO:
        # Remove this upon resolution of:
        # https://github.com/PyTorchLightning/pytorch-lightning/discussions/11392
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message="Your `predict_dataloader`'s sampler has shuffling enabled.*",
        )

        if mc_dropout > 0:
            self.set_mc_dropout(mc_dropout)

        predictions = trainer.predict(
            self, dataloaders=dataloader, return_predictions=True
        )
        scores = torch.cat([pred.scores for pred in predictions], dim=0).tolist()
        if "metadata" in predictions[0]:
            metadata = flatten_metadata([pred.metadata for pred in predictions])
        else:
            metadata = []

        if length_batching and gpus < 2:
            scores = restore_list_order(scores, sort_ids)
            output = Prediction(scores=scores, system_score=sum(scores) / len(scores))
            if metadata:
                output["metadata"] = Prediction(
                    **{k: restore_list_order(v, sort_ids) for k, v in metadata.items()}
                )
            return output
        else:
            return Prediction(scores=scores, system_score=sum(scores) / len(scores))
