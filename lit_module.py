from functools import cache
from typing import Dict, Optional

import pytorch_lightning as pl
import torch
import torchmetrics

from utils import init_model


class LitModule(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        learning_rate: float = 0.0001,
        use_tril_attention_mask: str = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.llm = self.register_core_module(init_model(model_name))
        self.learning_rate = learning_rate
        self.use_tril_attention_mask = use_tril_attention_mask
        self.metric_loss = torchmetrics.MeanMetric()
        self.metric_accuracy = torchmetrics.Accuracy(
            task='multiclass',
            num_classes=self.llm.config.vocab_size,
        )

    @cache
    def get_batch_tril_matrix(self, block_size: int, batch_size: Optional[int] = None) -> torch.Tensor:
        matrix = torch.ones(block_size, block_size).tril()
        if batch_size is not None:
            matrix = matrix.repeat(batch_size, 1, 1)
        return matrix

    def register_core_module(self, module: torch.nn.Module) -> torch.nn.Module:
        object.__setattr__(self, '__core_module__', module)
        return module

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx):
        batch_size, block_size = batch['input_ids'].shape
        if self.use_tril_attention_mask:
            batch['attention_mask'] = self.get_batch_tril_matrix(block_size, batch_size=batch_size).to(self.device)
        outputs = self.llm(**batch, return_dict=True)
        loss = outputs.loss

        self.log('train_loss', loss, rank_zero_only=True)

        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx):
        outputs = self.llm(**batch, return_dict=True)
        loss = outputs.loss
        logits = outputs.logits[..., :-1, :]
        labels = batch['labels'][..., 1:]

        self.metric_loss.update(loss)

        label_mask = labels != -100
        self.metric_accuracy.update(logits[label_mask], labels[label_mask])

    def on_validation_epoch_end(self) -> None:
        self.log('val_loss', self.metric_loss, rank_zero_only=True)
        self.log('accuracy', self.metric_accuracy, rank_zero_only=True)

    def configure_optimizers(self):
        strategy = self.trainer.strategy
        if isinstance(strategy, pl.strategies.DeepSpeedStrategy):
            assert "optimizer" not in strategy.config
            zero_config = strategy.config.get("zero_optimization")
            if zero_config is not None:
                if "offload_optimizer" in zero_config:
                    import deepspeed

                    optimizer = deepspeed.ops.adam.DeepSpeedCPUAdam(
                        self.trainer.model.parameters(), lr=self.learning_rate
                    )
                    return optimizer
        optimizer = torch.optim.AdamW(self.trainer.model.parameters(), lr=self.learning_rate)
        return optimizer

    def configure_callbacks(self):
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor='accuracy',
            mode='max',
            filename='{epoch:02d}-{accuracy:.4f}',
        )
        early_stop_callback = pl.callbacks.EarlyStopping(
            monitor='accuracy',
            min_delta=0.001,
            patience=3,
            mode='max',
            stopping_threshold=1,
        )
        return [checkpoint_callback, early_stop_callback]
