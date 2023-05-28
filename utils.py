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

import custom_models

custom_models.register_custom_configs()


def init_model(model_name: str) -> PreTrainedModel:
    config = AutoConfig.for_model(model_type=model_name)

    if model_name in custom_models.MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
        model = custom_models.AutoModelForCausalLM.from_config(config)
    elif model_name in custom_models.MODEL_MAPPING_NAMES:
        model = custom_models.AutoModel.from_config(config)
    else:
        try:
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        except ValueError:
            model = AutoModel.from_config(config, trust_remote_code=True)
    return model


def load_model(model_name_or_path: Union[str, os.PathLike]) -> PreTrainedModel:
    config = AutoConfig.from_pretrained(model_name_or_path)

    if config.model_type in custom_models.MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
        model = custom_models.AutoModelForCausalLM.from_pretrained(model_name_or_path, config=config)
    elif config.model_type in custom_models.MODEL_MAPPING_NAMES:
        model = custom_models.AutoModel.from_pretrained(model_name_or_path, config=config)
    else:
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, config=config, trust_remote_code=True)
        except ValueError:
            model = AutoModel.from_pretrained(model_name_or_path, config=config, trust_remote_code=True)
    return model


def load_tokenizer(tokenizer_name_or_path: Union[str, os.PathLike]) -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, padding_side='left', trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
