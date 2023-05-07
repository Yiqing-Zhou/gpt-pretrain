import os
from typing import Union

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)


def init_model(model_name: Union[str, os.PathLike]) -> PreTrainedModel:
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    except ValueError:
        model = AutoModel.from_config(config, trust_remote_code=True)
    return model


def load_model(model_name_or_path: Union[str, os.PathLike]) -> PreTrainedModel:
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
    except ValueError:
        model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
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
