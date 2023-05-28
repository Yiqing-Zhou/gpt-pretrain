import argparse
from functools import partial
from itertools import chain
from typing import Dict, Tuple

import datasets
import pytorch_lightning as pl
import torch
from torch.utils.data import ConcatDataset, DataLoader
from transformers import (
    BatchEncoding,
    DefaultDataCollator,
    PreTrainedTokenizer,
    set_seed,
)

from lit_module import LitModule
from lit_patches import apply_all_patches
from utils import load_tokenizer


def split_raw_dataset(
    raw_dataset: datasets.DatasetDict,
) -> Tuple[datasets.Dataset, datasets.Dataset]:
    if 'validation' in raw_dataset:
        train_dataset, val_dataset = raw_dataset['train'], raw_dataset['validation']
    else:
        raw_dataset = raw_dataset['train'].train_test_split(test_size=0.05, seed=args.seed)
        train_dataset, val_dataset = raw_dataset['train'], raw_dataset['test']
    return train_dataset, val_dataset


def process_dataset(dataset: datasets.Dataset, tokenizer: PreTrainedTokenizer) -> datasets.Dataset:
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
    parser.add_argument(
        "--learning_rate",
        type=float,
        help="Learning rate",
        default=0.0001,
    )
    parser.add_argument(
        "--use_tril_attention_mask",
        help="Use tril attention mask during training",
        action="store_true",
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
        "--max_epochs",
        type=int,
        help="Max epochs",
        default=None,
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
        type=int,
        help="Random seed",
        default=42,
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    if args.tokenizer_name_or_path is None:
        args.tokenizer_name_or_path = args.model_name

    set_seed(args.seed)

    # lightning module
    lit_module = LitModule(args.model_name, args.learning_rate, args.use_tril_attention_mask)

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
    apply_all_patches()
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
        max_epochs=args.max_epochs,
    )
    lit_trainer.fit(
        lit_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=args.resume_from_ckpt_path,
    )
