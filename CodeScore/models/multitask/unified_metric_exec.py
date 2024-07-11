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
Unified Metric
==============
    Unified Metric is a multitask metric that performs word-level and segment-level 
    evaluation in a multitask manner. It can also be used with and without reference 
    translations.
    
    Inspired on [UniTE](https://arxiv.org/pdf/2204.13346.pdf)
"""
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import json
from torch import nn
from transformers.optimization import Adafactor

from models.base_exec import CometModel
from models.metrics import MCCMetric, RegressionMetrics
from models.utils import Prediction, Target
from modules import LayerwiseAttention
from modules import FeedForward_exec as FeedForward


class UnifiedMetric_exec(CometModel):
    """UnifiedMetric is a multitask metric that performs word-level classification along
    with sentence-level regression. This metric has the ability to work with and without
    reference translations.

    Args:
        nr_frozen_epochs (Union[float, int]): Number of epochs (% of epoch) that the
            encoder is frozen. Defaults to 0.9.
        keep_embeddings_frozen (bool): Keeps the encoder frozen during training. Defaults
            to True.
        optimizer (str): Optimizer used during training. Defaults to 'AdamW'.
        encoder_learning_rate (float): Learning rate used to fine-tune the encoder model.
            Defaults to 3.0e-06.
        learning_rate (float): Learning rate used to fine-tune the top layers. Defaults
            to 3.0e-05.
        layerwise_decay (float): Learning rate % decay from top-to-bottom encoder layers.
            Defaults to 0.95.
        encoder_model (str): Encoder model to be used. Defaults to 'XLM-RoBERTa'.
        pretrained_model (str): Pretrained model from Hugging Face. Defaults to
            'microsoft/infoxlm-large'.
        sent_layer (Union[str, int]): Encoder layer to be used for regression task ('mix'
            for pooling info from all layers). Defaults to 'mix'.
        layer_transformation (str): Transformation applied when pooling info from all
            layers (options: 'softmax', 'sparsemax'). Defaults to 'sparsemax'.
        layer_norm (bool): Apply layer normalization. Defaults to 'False'.
        word_layer (int): Encoder layer to be used for word-level classification. Defaults
            to 24.
        loss (str): Loss function to be used. Defaults to 'mse'.
        dropout (float): Dropout used in the top-layers. Defaults to 0.1.
        batch_size (int): Batch size used during training. Defaults to 4.
        train_data (Optional[List[str]]): List of paths to training data. Each file is
            loaded consecutively for each epoch. Defaults to None.
        validation_data (Optional[List[str]]): List of paths to validation data.
            Validation results are averaged across validation set. Defaults to None.
        hidden_sizes (List[int]): Size of hidden layers used in the regression head.
            Defaults to [3072, 1024].
        activations (Optional[str]): Activation function used in the regression head.
            Defaults to 'Tanh'.
        final_activation (Optional[str]): Activation function used in the last layer of
            the regression head. Defaults to None.
        input_segments (Optional[List[str]]): List with input segment names to be used.
            Defaults to ["mt", "src", "ref"].
        word_level_training (bool): If True, the model is trained with multitask
            objective. Defaults to False.
        word_weights (List[float]): Loss weight for OK/BAD tags. Defaults to [0.15,
            0.85].
        loss_lambda (foat): Weight assigned to the word-level loss. Defaults to 0.65.
    """

    def __init__(
        self,
        nr_frozen_epochs: Union[float, int] = 0.9,
        keep_embeddings_frozen: bool = True,
        optimizer: str = "AdamW",
        encoder_learning_rate: float = 3.0e-06,
        learning_rate: float = 3.0e-05,
        layerwise_decay: float = 0.95,
        encoder_model: str = "XLM-RoBERTa",
        pretrained_model: str = "microsoft/infoxlm-large",
        sent_layer: Union[str, int] = "mix",
        layer_transformation: str = "sparsemax",
        layer_norm: bool = True,
        word_layer: int = 12,
        loss: str = "mse",
        dropout: float = 0.1,
        batch_size: int = 4,
        train_data: Optional[List[str]] = None,
        validation_data: Optional[List[str]] = None,
        hidden_sizes: List[int] = [3072, 1024],
        activations: str = "Tanh",
        final_activation: Optional[str] = 'Sigmoid',
        input_segments: List[str] = ["mt", "src", "ref", "src_ref"],
        word_level_training: bool = False,
        word_weights: List[float] = [0.15, 0.85],
        loss_lambda: float = 0.65,
    ) -> None:
        super().__init__(
            nr_frozen_epochs=nr_frozen_epochs,
            keep_embeddings_frozen=keep_embeddings_frozen,
            optimizer=optimizer,
            encoder_learning_rate=encoder_learning_rate,
            learning_rate=learning_rate,
            layerwise_decay=layerwise_decay,
            encoder_model=encoder_model,
            pretrained_model=pretrained_model,
            layer=sent_layer,
            loss=loss,
            dropout=dropout,
            batch_size=batch_size,
            train_data=train_data,
            validation_data=validation_data,
            class_identifier="unified_metric",
        )
        self.save_hyperparameters()
        self.estimator = FeedForward(
            in_dim=self.encoder.output_units,
            hidden_sizes=self.hparams.hidden_sizes,
            activations=self.hparams.activations,
            dropout=self.hparams.dropout,
            final_activation=self.hparams.final_activation,
        )

        if self.hparams.sent_layer == "mix":
            self.layerwise_attention = LayerwiseAttention(
                layer_transformation=layer_transformation,
                num_layers=self.encoder.num_layers,
                dropout=self.hparams.dropout,
                layer_norm=self.hparams.layer_norm,
            )
        else:
            self.layerwise_attention = None

        self.input_segments = input_segments
        self.word_level = word_level_training
        if word_level_training:
            self.encoder.add_span_tokens("<v>", "</v>")
            self.hidden2tag = nn.Linear(self.encoder.output_units, 2)
        self.init_losses()

    def init_metrics(self):
        """Initializes training and validation metrics"""
        # Train and Dev correlation metrics
        self.train_score_corr = RegressionMetrics(prefix="train_score")
        # self.train_passed_corr = RegressionMetrics(prefix='train_passed')
        # self.train_pass_at_1_corr = RegressionMetrics(prefix='train_pass_at_1')
        self.val_score_corr = nn.ModuleList(
            [RegressionMetrics(prefix=d + 'score') for d in self.hparams.validation_data]
        )
        # self.val_passed_corr = nn.ModuleList(
        #     [RegressionMetrics(prefix=d + 'passed') for d in self.hparams.validation_data]
        # )
        # self.val_pass_at_1_corr = nn.ModuleList(
        #     [RegressionMetrics(prefix=d + 'pass_at_1') for d in self.hparams.validation_data]
        # )
        
        if self.hparams.word_level_training:
            # Train and Dev MCC
            self.train_mcc = MCCMetric(num_classes=2, prefix="train")
            self.val_mcc = nn.ModuleList(
                [
                    MCCMetric(num_classes=2, prefix=d)
                    for d in self.hparams.validation_data
                ]
            )



    def init_losses(self) -> None:
        """Initializes Loss functions to be used."""
        self.sentloss = nn.MSELoss()
        # self.passloss = nn.CrossEntropyLoss()
        self.passloss = nn.BCEWithLogitsLoss()
        if self.word_level:
            self.wordloss = nn.CrossEntropyLoss(
                reduction="mean",
                weight=torch.tensor(self.hparams.word_weights),
                ignore_index=-1,
            )

    def requires_references(self) -> bool:
        """ Unified models can be developed to exclusively use [mt, ref] or to use both
        [mt, src, ref]. Models developed to use the source will work in a quality
        estimation scenario but models trained with [mt, ref] won't!
        
        Return:
            [bool]: True if the model was trained to work exclusively with references. 
        """
        if self.input_segments == ["mt", "ref"]:
            return True
        return False

    def configure_optimizers(
        self,
    ) -> Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler.LambdaLR]]:
        """Pytorch Lightning method to initialize a training Optimizer and learning
        rate scheduler.

        Returns:
            Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler.LambdaLR]]:
                List with Optimizers and a List with lr_schedulers.
        """
        params = self.encoder.layerwise_lr(
            self.hparams.encoder_learning_rate, self.hparams.layerwise_decay
        )
        params += [
            {"params": self.estimator.parameters(), "lr": self.hparams.learning_rate}
        ]
        if self.word_level:
            params += [
                {
                    "params": self.hidden2tag.parameters(),
                    "lr": self.hparams.learning_rate,
                },
            ]

        if self.layerwise_attention:
            params += [
                {
                    "params": self.layerwise_attention.parameters(),
                    "lr": self.hparams.learning_rate,
                }
            ]

        if self.hparams.optimizer == "Adafactor":
            optimizer = Adafactor(
                params,
                lr=self.hparams.learning_rate,
                relative_step=False,
                scale_parameter=False,
            )
        else:
            optimizer = torch.optim.AdamW(params, lr=self.hparams.learning_rate)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=10, min_lr=5e-6)

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_mae"}
        # return [optimizer], []
        
    def read_file(self, path):
        examples = []
        with open(path, 'r') as f:
            lines = f.readlines()
        for l in lines:
            exp = json.loads(l)
            if exp.get('id') is not None:
                del exp['id']
            examples.append(exp)
        

        return examples

    def concat_inputs(
        self, input_sequences: Tuple[Dict[str, torch.Tensor]]
    ) -> Tuple[Dict[str, torch.Tensor]]:
        """Prepares tokenized src, ref and mt for joint encoding by putting
        everything into a single contiguous sequence.

        Args:
            input_sequences (Tuple[Dict[str, torch.Tensor]]): Tokenized Source, MT and
                Reference.

        Returns:
            Tuple[Dict[str, torch.Tensor]]: Contiguous sequence.
        """
        model_inputs = OrderedDict()
        # If we are using source and reference we will have to create 3 different input
        if len(input_sequences) == 3:
            # mt_src, mt_ref = input_sequences[:2], [
            #     input_sequences[0],
            #     input_sequences[2],
            # ]
            mt_src, mt_ref = [input_sequences[0], input_sequences[2]], input_sequences[:2]
            src_input, _, _ = self.encoder.concat_sequences(
                mt_src, return_in_span_mask=self.word_level
            )
            ref_input, _, _ = self.encoder.concat_sequences(
                mt_ref, return_in_span_mask=self.word_level
            )
            full_input, _, _ = self.encoder.concat_sequences(
                input_sequences, return_in_span_mask=self.word_level
            )
            if 'src_ref' in self.hparams.input_segments:
                model_inputs['inputs'] = (full_input,)
                model_inputs["mt_length"] = input_sequences[0]["attention_mask"].sum(dim=1)
            else:
                model_inputs["inputs"] = (src_input, ref_input, full_input)
                model_inputs["mt_length"] = input_sequences[0]["attention_mask"].sum(dim=1)
            return model_inputs

        # Otherwise we will have one single input sequence that concatenates the MT
        # with SRC/REF.
        else:
            model_inputs["inputs"] = (
                self.encoder.concat_sequences(
                    input_sequences, return_in_span_mask=self.word_level
                )[0],
            )
            model_inputs["mt_length"] = input_sequences[0]["attention_mask"].sum(dim=1)
        return model_inputs

    def prepare_sample(
        self, sample: List[Dict[str, Union[str, float]]], stage: str = "train"
    ) -> Union[Tuple[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
        """Tokenizes input data and prepares targets for training.

        Args:
            sample (List[Dict[str, Union[str, float]]]): Mini-batch
            stage (str, optional): Model stage ('train' or 'predict'). Defaults to "train".

        Returns:
            Union[Tuple[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]: Model input
                and targets.
        """
        sample = {k: [dic[k] for dic in sample] for k in sample[0]}
        input_sequences = [
            self.encoder.prepare_sample(sample["generated_code"], self.word_level),
        ]

        if ("golden_code" in sample) and ("ref" in self.hparams.input_segments):
            input_sequences.append(
                self.encoder.prepare_sample(sample["golden_code"], self.word_level)
            )

        if ("source" in sample) and ("src" in self.hparams.input_segments):
            input_sequences.append(
                self.encoder.prepare_sample(sample["source"], self.word_level)
            )

        # if ("ref" in sample) and ("ref" in self.hparams.input_segments):
        #     input_sequences.append(
        #         self.encoder.prepare_sample(sample["golden_code"], self.word_level)
        #     )

        model_inputs = self.concat_inputs(input_sequences)
        if stage == "predict":
            return model_inputs["inputs"]

        targets = Target(score=torch.tensor(sample["score"], dtype=torch.float), passed=torch.tensor(sample['passed'], dtype=torch.float), pass_at_1=torch.tensor(sample['pass@1'], dtype=torch.float))
        if "system" in sample:
            targets["system"] = sample["system"]

        targets["mt_length"] = model_inputs["mt_length"]
        if self.word_level:
            # Labels will be the same accross all inputs because we are only
            # doing sequence tagging on the MT. We will only use the mask corresponding
            # to the MT segment.
            seq_len = model_inputs["mt_length"].max()
            targets["labels"] = model_inputs["inputs"][0]["in_span_mask"][:, :seq_len]

        return model_inputs["inputs"], targets

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward function.

        Args:
            input_ids (torch.Tensor): Input sequence.
            attention_mask (torch.Tensor): Attention mask.
            token_type_ids (Optional[torch.Tensor], optional): Token type ids for
                BERT-like models. Defaults to None.

        Raises:
            Exception: Invalid model word/sent layer if self.{word/sent}_layer are not
                valid encoder model layers .

        Returns:
            Dict[str, torch.Tensor]: Sentence scores and word-level logits (if
                word_level_training = True)
        """
        encoder_out = self.encoder(
            input_ids, attention_mask, token_type_ids=token_type_ids
        )

        # Word embeddings used for the word-level classification task
        if self.word_level:
            if (
                isinstance(self.hparams.word_layer, int)
                and 0 <= self.hparams.word_layer < self.encoder.num_layers
            ):
                wordemb = encoder_out["all_layers"][self.hparams.word_layer]
            else:
                raise Exception(
                    "Invalid model word layer {}.".format(self.hparams.word_layer)
                )

        # Word embeddings used for the sentence-level regression task
        if self.layerwise_attention:
            sentemb = self.layerwise_attention(
                encoder_out["all_layers"], attention_mask
            )[:, 0, :]

        elif (
            isinstance(self.hparams.sent_layer, int)
            and 0 <= self.hparams.sent_layer < self.encoder.num_layers
        ):
            sentemb = encoder_out["all_layers"][self.hparams.sent_layer][:, 0, :]
        else:
            raise Exception(
                "Invalid model sent layer {}.".format(self.hparams.word_layer)
            )

        if self.word_level:
            sentence_output = self.estimator(sentemb)
            word_output = self.hidden2tag(wordemb)
            return Prediction(score=sentence_output[:, 0], passed=sentence_output[:, 1], pass_at_1=sentence_output[:, 2], logits=word_output)
            
            # return Prediction(score=sentence_output[:, 0], passed=torch.stack((sentence_output[:, 1], sentence_output[:, 2]), dim=1), pass_at_1=torch.stack((sentence_output[:, 3], sentence_output[:, 4]), dim=1), logits=word_output)

        estimator_output = self.estimator(sentemb)

        return Prediction(score=estimator_output[:, 0], passed=estimator_output[:, 1], pass_at_1=estimator_output[:, 2])
        # return Prediction(score=sentence_output[:, 0], passed=torch.stack((sentence_output[:, 1], sentence_output[:, 2]), dim=1), pass_at_1=torch.stack((sentence_output[:, 3], sentence_output[:, 4]), dim=1))

    def compute_loss(self, prediction: Prediction, target: Target) -> torch.Tensor:
        """Receives model batch prediction and respective targets and computes
        a loss value

        Args:
            prediction (Prediction): Batch prediction
            target (Target): Batch targets

        Returns:
            torch.Tensor: Loss value
        """
        sentence_loss = self.sentloss(prediction.score, target.score)
        passed_loss = self.passloss(prediction.passed, target.passed)
        pass_at_1_loss = self.passloss(prediction.pass_at_1, target.pass_at_1)
        if self.word_level:
            # sentence_loss = self.sentloss(prediction.score, target.score)
            predictions = prediction.logits.reshape(-1, 2)
            targets = target.labels.view(-1).type(torch.LongTensor).cuda()
            word_loss = self.wordloss(predictions, targets)
            return 0.4 * sentence_loss * (1 - self.hparams.loss_lambda) + word_loss * (
                self.hparams.loss_lambda
            ) + 0.3 * passed_loss + 0.3 + pass_at_1_loss
        return 0.4 * sentence_loss + 0.3 * passed_loss + 0.3 * pass_at_1_loss

    def training_step(
        self, batch: Tuple[Dict[str, torch.Tensor]], batch_nb: int
    ) -> torch.Tensor:
        """Pytorch Lightning training_step.

        Args:
            batch (Tuple[Dict[str, torch.Tensor]]): The output of your prepare_sample
                function.
            batch_nb (int): Integer displaying which batch this is.

        Returns:
            torch.Tensor: Loss value
        """
        batch_input, batch_target = batch
        if len(batch_input) == 3 and 'src_ref' not in self.hparams.input_segments:
            # In UniTE training is made of 3 losses:
            #    Lsrc + Lref + Lsrc+ref
            # For that reason we have to perform 3 forward passes and sum
            # the respective losses.
            predictions = [self.forward(**input_seq) for input_seq in batch_input]
            loss_value = 0
            for pred in predictions:
                # We created the target according to the MT.
                seq_len = batch_target.mt_length.max()
                if self.word_level:
                    pred.logits = pred.logits[:, :seq_len, :]

                loss_value += self.compute_loss(pred, batch_target)

        else:
            batch_prediction = self.forward(**batch_input[0])
            seq_len = batch_target.mt_length.max()
            batch_prediction.logits = batch_prediction.logits[:, :seq_len, :]
            loss_value = self.compute_loss(batch_prediction, batch_target)

        if (
            self.nr_frozen_epochs < 1.0
            and self.nr_frozen_epochs > 0.0
            and batch_nb > self.first_epoch_total_steps * self.nr_frozen_epochs
        ):
            self.unfreeze_encoder()
            self._frozen = False

        self.log("train_loss", loss_value, on_step=True, on_epoch=True, sync_dist=True)
        return loss_value

    def validation_step(
        self, batch: Tuple[Dict[str, torch.Tensor]], batch_nb: int, dataloader_idx: int
    ) -> None:
        """Pytorch Lightning validation_step.

        Args:
            batch (Tuple[Dict[str, torch.Tensor]]): The output of your prepare_sample
                function.
            batch_nb (int): Integer displaying which batch this is.
            dataloader_idx (int): Integer displaying which dataloader this is.
        """
        batch_input, batch_target = batch
        if len(batch_input) == 3 and 'src_ref' not in self.hparams.input_segments:
            predictions = [self.forward(**input_seq) for input_seq in batch_input]
            # Final score is the average of the 3 scores!
            scores = torch.stack([pred.score for pred in predictions], dim=0).mean(
                dim=0
            )
            # passed_scores = torch.stack([pred.passed for pred in predictions], dim=0).mean(
            #     dim=0
            # )
            # pass_at_1_scores = torch.stack([pred.pass_at_1 for pred in predictions], dim=0).mean(
            #     dim=0
            # )
            # batch_prediction = Prediction(score=scores, passed=passed_scores, pass_at_1=pass_at_1_scores)
            batch_prediction = Prediction(score=scores)

            if self.word_level:
                seq_len = batch_target.mt_length.max()
                # Final Logits for each word is the average of the 3 scores!
                batch_prediction["logits"] = (
                    predictions[0].logits[:, :seq_len, :]
                    + predictions[1].logits[:, :seq_len, :]
                    + predictions[2].logits[:, :seq_len, :]
                ) / 3

        else:
            seq_len = batch_target.mt_length.max()
            batch_prediction = self.forward(**batch_input[0])
            batch_prediction.logits = batch_prediction.logits[:, :seq_len, :]

        if self.word_level:
            # Removing masked targets and the corresponding logits.
            # This includes subwords and padded tokens.
            logits = batch_prediction.logits.reshape(-1, 2)
            targets = batch_target.labels.view(-1)
            mask = targets != -1
            logits, targets = logits[mask, :], targets[mask].int()

        if dataloader_idx == 0:
            self.train_score_corr.update(batch_prediction.score, batch_target.score)
            # self.train_passed_corr.update(batch_prediction.passed, batch_target.passed)
            # self.train_pass_at_1_corr.update(batch_prediction.pass_at_1, batch_target.pass_at_1)
            if self.word_level:
                self.train_mcc.update(logits, targets)

        elif dataloader_idx > 0:
            self.val_score_corr[dataloader_idx - 1].update(
                batch_prediction.score,
                batch_target.score,
                batch_target["system"] if "system" in batch_target else None,
            )
            # self.val_passed_corr[dataloader_idx - 1].update(
            #     batch_prediction.passed,
            #     batch_target.passed,
            #     batch_target["system"] if "system" in batch_target else None,
            # )
            # self.val_pass_at_1_corr[dataloader_idx - 1].update(
            #     batch_prediction.pass_at_1,
            #     batch_target.pass_at_1,
            #     batch_target["system"] if "system" in batch_target else None,
            # )
            if self.word_level:
                self.val_mcc[dataloader_idx - 1].update(logits, targets)

    # Overwriting this method to log correlation and classification metrics
    def validation_epoch_end(self, *args, **kwargs) -> None:
        """Computes and logs metrics."""
        self.log_dict(self.train_score_corr.compute(), prog_bar=False, sync_dist=True)
        # self.log_dict(self.train_passed_corr.compute(), prog_bar=False, sync_dist=True)
        # self.log_dict(self.train_pass_at_1_corr.compute(), prog_bar=False, sync_dist=True)
        self.train_score_corr.reset()
        # self.train_passed_corr.reset()
        # self.train_pass_at_1_corr.reset()

        if self.word_level:
            self.log_dict(self.train_mcc.compute(), prog_bar=False, sync_dist=True)
            self.train_mcc.reset()

        val_metrics = []
        for i in range(len(self.hparams.validation_data)):
            corr_score_metrics = self.val_score_corr[i].compute()
            # corr_passed_metrics = self.val_passed_corr[i].compute()
            # corr_pass_at_1_metrics = self.val_pass_at_1_corr[i].compute()
            self.val_score_corr[i].reset()
            if self.word_level:
                cls_metric = self.val_mcc[i].compute()
                self.val_mcc[i].reset()
                # results = {**corr_score_metrics, **corr_passed_metrics, **corr_pass_at_1_metrics, **cls_metric}
                results = {**corr_score_metrics, **cls_metric}
            else:
                # results = {**corr_score_metrics, **corr_passed_metrics, **corr_pass_at_1_metrics}
                results = corr_score_metrics

            # Log to tensorboard the results for this validation set.
            self.log_dict(results, prog_bar=False, sync_dist=True)
            val_metrics.append(results)

        average_results = {"val_" + k.split("_")[-1]: [] for k in val_metrics[0].keys()}
        for i in range(len(val_metrics)):
            for k, v in val_metrics[i].items():
                average_results["val_" + k.split("_")[-1]].append(v)

        self.log_dict(
            {k: sum(v) / len(v) for k, v in average_results.items()}, prog_bar=True, sync_dist=True
        )

    def predict_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
    ) -> Prediction:
        """PyTorch Lightning predict_step

        Args:
            batch (Dict[str, torch.Tensor]): The output of your prepare_sample function
            batch_idx (Optional[int], optional): Integer displaying which batch this is
                Defaults to None.
            dataloader_idx (Optional[int], optional): Integer displaying which
                dataloader this is. Defaults to None.

        Returns:
            Prediction: Model Prediction
        """

        def decode(subword_probs, input_ids, mt_length):
            decoded_output = []
            for i in range(mt_length.shape[0]):
                tokens = self.encoder.tokenizer.convert_ids_to_tokens(
                    input_ids[i][: mt_length[i]]
                )
                token_probs = subword_probs[i][: mt_length[i]]
                # remove BOS and EOS
                tokens, token_probs = tokens[1:-1], token_probs[1:-1]
                decoded_output.append(
                    [(token, prob.item()) for token, prob in zip(tokens, token_probs)]
                )
            return decoded_output

        if self.mc_dropout:
            raise NotImplementedError("MCD not implemented for this model!")

        if len(batch) == 3 and 'src_ref' not in self.hparams.input_segments:
            predictions = [self.forward(**input_seq) for input_seq in batch]
            # Final score is the average of the 3 scores!
            avg_scores = torch.stack([pred.score for pred in predictions], dim=0).mean(
                dim=0
            )
            avg_passeds = torch.stack([pred.passed for pred in predictions], dim=0).mean(
                dim=0
            )
            avg_pass_at_1s = torch.stack([pred.pass_at_1 for pred in predictions], dim=0).mean(
                dim=0
            )
            batch_prediction = Prediction(
                scores=avg_scores,
                passeds=avg_passeds,
                pass_at_1s=avg_pass_at_1s,
                score_metadata=Prediction(
                    src_scores=predictions[0].score,
                    ref_scores=predictions[1].score,
                    unified_scores=predictions[2].score,
                ),
                passed_metadata=Prediction(
                    src_passeds=predictions[0].passed,
                    ref_passeds=predictions[1].passed,
                    unified_passeds=predictions[2].passed,
                ),
                pass_at_1_metadata=Prediction(
                    src_pass_at_1s=predictions[0].pass_at_1,
                    ref_pass_at_1s=predictions[1].pass_at_1,
                    unified_pass_at_1s=predictions[2].pass_at_1,
                ),
            )
            if self.word_level:
                mt_mask = batch[0]["in_span_mask"] != -1
                mt_length = mt_mask.sum(dim=1)
                seq_len = mt_length.max()
                subword_probs = [
                    nn.functional.softmax(o.logits, dim=2)[:, :seq_len, :]
                    for o in predictions
                ]
                subword_probs = torch.mean(torch.stack(subword_probs), dim=0)[:, :, 1]
                subword_scores = decode(subword_probs, batch[0]["input_ids"], mt_length)
                batch_prediction.score_metadata["subword_scores"] = subword_scores

        else:
            model_output = self.forward(**batch[0])
            batch_prediction = Prediction(scores=model_output.score, passeds=model_output.passed, pass_at_1s=model_output.pass_at_1)
            if self.word_level:
                mt_mask = batch[0]["in_span_mask"] != -1
                mt_length = mt_mask.sum(dim=1)
                seq_len = mt_length.max()
                subword_probs = nn.functional.softmax(model_output.logits, dim=2)[
                    :, :seq_len, :
                ][:, :, 1]
                subword_scores = decode(subword_probs, batch[0]["input_ids"], mt_length)

                batch_prediction = Prediction(
                    scores=model_output.score,
                    passeds=model_output.passed,
                    pass_at_1s=model_output.pass_at_1,
                    score_metadata=Prediction(subword_scores=subword_scores),
                )
        return batch_prediction
