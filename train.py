import argparse
import os
from functools import cache, partial
from itertools import chain
from typing import Dict, Optional, Tuple, Union

import datasets
import pytorch_lightning as pl
import torch
import torchmetrics
from torch.utils.data import ConcatDataset, DataLoader
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    DefaultDataCollator,
    PreTrainedModel,
    PreTrainedTokenizer,
    set_seed,
)


def init_model(model_name: Union[str, os.PathLike]) -> PreTrainedModel:
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    except ValueError:
        model = AutoModel.from_config(config, trust_remote_code=True)
    return model


def load_tokenizer(
    tokenizer_name_or_path: Union[str, os.PathLike]
) -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path, padding_side='left', trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def split_raw_dataset(
    raw_dataset: datasets.DatasetDict,
) -> Tuple[datasets.Dataset, datasets.Dataset]:
    if 'validation' in raw_dataset:
        train_dataset, val_dataset = raw_dataset['train'], raw_dataset['validation']
    else:
        raw_dataset = raw_dataset['train'].train_test_split(
            test_size=0.05, seed=args.seed
        )
        train_dataset, val_dataset = raw_dataset['train'], raw_dataset['test']
    return train_dataset, val_dataset


def process_dataset(
    dataset: datasets.Dataset, tokenizer: PreTrainedTokenizer
) -> datasets.Dataset:
    def group_texts(examples: Dict[str, list], block_size: int = 512) -> BatchEncoding:
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result['labels'] = result['input_ids'].copy()
        result = BatchEncoding(result)
        return result

    def tokenize_inputs(
        examples: Dict[str, list],
        tokenizer: PreTrainedTokenizer,
        column_name: str = 'text',
    ) -> BatchEncoding:
        return tokenizer(examples[column_name], return_attention_mask=False)

    dataset_column_names = list(dataset.features)
    dataset = dataset.map(
        partial(
            tokenize_inputs,
            tokenizer=tokenizer,
            column_name=dataset_column_names[0],
        ),
        batched=True,
        num_proc=args.num_proc,
        remove_columns=dataset_column_names,
    ).map(
        partial(group_texts, block_size=tokenizer.model_max_length),
        batched=True,
        num_proc=args.num_proc,
    )
    return dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of or path to model",
        default='gpt2',
    )
    parser.add_argument("--fp16", help="Enable fp16", action="store_true")
    parser.add_argument("--bf16", help="Enable bf16", action="store_true")
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        help="Name of or path to tokenizer",
        default=None,
    )
    parser.add_argument(
        "--dataset_name",
        nargs='+',
        type=str,
        help="Name(s) of dataset. To specify a config, pass a <dataset_name>:<dataset_config_name>",
        default=["wikitext:wikitext-2-v1"],
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        help="Batch size of training",
        default=8,
    )
    parser.add_argument(
        "--val_batch_size",
        type=int,
        help="Batch size of validating",
        default=16,
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        help="Accumulate grad batches",
        default=32,
    )
    parser.add_argument(
        "--num_proc",
        type=str,
        help="Number of data processes",
        default=16,
    )
    parser.add_argument(
        "--strategy",
        type=str,
        help="Name of pytorch lightning distribution strategy",
        default='fsdp',
    )
    parser.add_argument(
        "--resume_from_ckpt_path",
        type=str,
        help="Checkpoint file path to resume from",
        default=None,
    )
    parser.add_argument(
        "--seed",
        type=str,
        help="Random seed",
        default=42,
    )
    args = parser.parse_args()
    return args


class LitModule(pl.LightningModule):
    def __init__(self, model_name: str):
        super().__init__()
        self.save_hyperparameters()
        self.llm = self.register_core_module(init_model(model_name))
        self.metric_loss = torchmetrics.MeanMetric()
        self.metric_accuracy = torchmetrics.Accuracy(
            task='multiclass',
            num_classes=self.llm.config.vocab_size,
        )

    @cache
    def get_tril_matrix(
        self, block_size: int, batch_size: Optional[int] = None
    ) -> torch.Tensor:
        matrix = torch.ones(block_size, block_size).tril()
        if batch_size is not None:
            matrix = matrix.repeat(batch_size, 1, 1)
        return matrix

    def register_core_module(self, module: torch.nn.Module) -> torch.nn.Module:
        object.__setattr__(self, '__core_module__', module)
        return module

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx):
        batch_size, block_size = batch['input_ids'].shape
        batch['attention_mask'] = self.get_tril_matrix(
            block_size, batch_size=batch_size
        ).to(self.device)
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
        optimizer = torch.optim.AdamW(self.trainer.model.parameters(), lr=0.0001)
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


if __name__ == '__main__':
    args = parse_args()

    if args.tokenizer_name_or_path is None:
        args.tokenizer_name_or_path = args.model_name

    set_seed(args.seed)

    # lightning module
    lit_module = LitModule(args.model_name)

    # datasets
    tokenizer = load_tokenizer(args.tokenizer_name_or_path)
    train_dataset_list = []
    val_dataset_list = []
    for dataset_name in args.dataset_name:
        dataset_args = dataset_name.split(':')
        raw_dataset = datasets.load_dataset(*dataset_args)
        train_dataset, val_dataset = split_raw_dataset(raw_dataset)
        train_dataset = process_dataset(train_dataset, tokenizer)
        val_dataset = process_dataset(val_dataset, tokenizer)
        train_dataset_list.append(train_dataset)
        val_dataset_list.append(val_dataset)
    train_dataset = ConcatDataset(train_dataset_list)
    val_dataset = ConcatDataset(val_dataset_list)

    # dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        num_workers=args.num_proc,
        collate_fn=DefaultDataCollator(),
        persistent_workers=True,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        num_workers=args.num_proc,
        collate_fn=DefaultDataCollator(),
        persistent_workers=True,
    )

    # trainer
    torch.set_float32_matmul_precision('medium')
    if args.bf16:
        precision = 'bf16-mixed'
    elif args.fp16:
        precision = '16-mixed'
    else:
        precision = "32-true"
    lit_trainer = pl.Trainer(
        accelerator='gpu',
        precision=precision,
        log_every_n_steps=5,
        accumulate_grad_batches=args.accumulate_grad_batches,
        strategy=args.strategy,
    )
    lit_trainer.fit(
        lit_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=args.resume_from_ckpt_path,
    )
