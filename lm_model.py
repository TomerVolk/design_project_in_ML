import logging
import torch
from pytorch_lightning import LightningModule, Callback
from torch import Tensor, LongTensor
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from argparse import Namespace
from typing import List, Any, Dict
from transformers import (
    AdamW,
    BatchEncoding,
    T5ForConditionalGeneration,
    get_linear_schedule_with_warmup
)
from lm_dataset import LMDataset


def _init_data_loaders(h_params: Namespace) -> Dict[str, DataLoader]:
    train_dataset = LMDataset(h_params, h_params.train_path)
    dev_dataset = LMDataset(h_params, h_params.dev_path)
    train_data_loader = DataLoader(train_dataset, shuffle=True, batch_size=h_params.batch_size)
    dev_data_loader = DataLoader(dev_dataset, shuffle=False, batch_size=h_params.batch_size)
    data_loaders = {"train": train_data_loader, "dev": dev_data_loader}
    return data_loaders


class T5ForSentenceClassification(LightningModule):
    LOSS_IGNORE_ID = -100

    def __init__(self, h_params: Namespace):
        super().__init__()
        self.h_params = h_params
        self.T5 = T5ForConditionalGeneration.from_pretrained(self.h_params.T5_model_name)
        self.T5: T5ForConditionalGeneration

        self.data_loaders = _init_data_loaders(self.h_params)
        self.tokenizer = self.data_loaders["train"].dataset.tokenizer
        self.loss_fn = CrossEntropyLoss(ignore_index=T5ForSentenceClassification.LOSS_IGNORE_ID)
        self.cur_loss = None

    def forward(self, input_ids: LongTensor, attention_mask: LongTensor = None, labels=None) -> Tensor:
        outputs = self.T5(input_ids, attention_mask=attention_mask, labels=labels)
        gen_loss = outputs[0]

        return gen_loss

    def _step(self, batch: BatchEncoding, mode) -> Tensor:
        """
        Compute and return the training loss.

        Args:
            batch: a dictionary with the following keys:
                input_ids: tensor of shape (batch_size, sequence_length) containing the token ids for the input.
                attention_mask: tensor of shape (batch_size, sequence_length) containing the attention masks to avoid
                    performing attention on padding token indices.
                labels: tensor of shape (batch_size,) with labels for computing the loss.
        Returns:
            Tensor of shape (1,) with the loss after the forward pass.
        """
        input_ids = batch[0]
        input_mask = batch[1]
        label_ids = batch[2]
        loss = self(input_ids=input_ids, attention_mask=input_mask, labels=label_ids)
        if self.h_params.gpus > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        self.log(f"{mode}_loss", float(loss), on_step=False, on_epoch=False, prog_bar=True, logger=True)
        return loss

    def training_epoch_end(self, outputs: List[Any]) -> None:
        """
        Called at the end of the training epoch with the outputs of all training steps.

        Args:
            outputs: List of outputs like defined in training_step(), or if there are multiple data_loaders,
                a list containing a list of outputs for each dataloader.
        """
        losses = [cur_out["loss"] for cur_out in outputs]
        avg_epoch_loss = torch.stack(losses).mean()
        self.log(f"avg_train_loss", avg_epoch_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def _eval_epoch_end(self, outputs: List[Any], split: str) -> None:
        # losses = [cur_out["loss"] for cur_out in outputs]

        avg_epoch_loss = torch.stack(outputs).mean()
        self.log(f"avg_{split}_loss", avg_epoch_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.cur_loss = avg_epoch_loss

    def validation_step(self, batch: BatchEncoding, batch_idx: int) -> Tensor:
        """
        Operates on a single batch of data from the validation set.

        This step is used to generate examples or calculate anything of interest like accuracy.

        Args:
            batch: the output of the DataLoader
            batch_idx: the index of this batch.

        Returns:
            A tuple of (loss, generated_texts, labels_texts, sample_ids)
        """
        return self._step(batch, "dev")

    def test_step(self, batch: BatchEncoding, batch_idx: int) -> Tensor:
        """
        Operates on a single batch of data from the validation set.

        This step is used to generate examples or calculate anything of interest like accuracy.

        Args:
            batch: the output of the DataLoader
            batch_idx: the index of this batch.

        Returns:
            A tuple of (loss, generated_texts, labels_texts, sample_ids)
        """
        return self._step(batch, "test")

    def training_step(self, batch: BatchEncoding, batch_idx: int) -> Tensor:
        """
        Operates on a single batch of data from the validation set.

        This step is used to generate examples or calculate anything of interest like accuracy.

        Args:
            batch: the output of the DataLoader
            batch_idx: the index of this batch.

        Returns:
            A tuple of (loss, generated_texts, labels_texts, sample_ids)
        """
        return self._step(batch, "train")

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        """
        Called at the end of the validation epoch with.

        Args:
            outputs: the outputs of all validation steps.
        """
        self._eval_epoch_end(outputs, "dev")

    def test_epoch_end(self, outputs: List[Any]) -> None:
        """
        Called at the end of the validation epoch with.

        Args:
            outputs: the outputs of all validation steps.
        """
        self._eval_epoch_end(outputs, "test")

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.T5.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.h_params.weight_decay,
            },
            {
                "params": [p for n, p in self.T5.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.h_params.learning_rate, eps=self.h_params.adam_epsilon)
        t_total = (
                (len(self.data_loaders["train"].dataset) // (self.h_params.batch_size *
                                                             float(max(1, self.h_params.gpus))))
                // self.h_params.gradient_accumulation_steps * float(self.h_params.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=self.h_params.warmup_steps,
                                                    num_training_steps=t_total)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def train_dataloader(self) -> DataLoader:
        return self.data_loaders["train"]

    def val_dataloader(self) -> DataLoader:
        return self.data_loaders["dev"]

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        if 'v_num' in tqdm_dict:
            del tqdm_dict['v_num']
        return tqdm_dict


class LoggingCallback(Callback):
    def __init__(self, logger=None):
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
        self.logger.info(f"Using GPU: {torch.cuda.is_available()}")

    def on_validation_end(self, trainer, pl_module):
        self.logger.info("***** Validation results *****")
        metrics = filter(lambda x: x[0] not in ["log", "progress_bar"], trainer.callback_metrics.items())
        # Log results
        for key, metric in sorted(metrics):
            self.logger.info(f"{key} = {metric:.03f}\n")
