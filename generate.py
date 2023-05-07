import argparse
from typing import List

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from utils import load_model, load_tokenizer


def eval_prompts(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    use_tril_attention_mask: bool = False,
) -> List[str]:
    inputs = tokenizer(
        prompts, padding=True, return_tensors='pt', return_attention_mask=True
    )
    inputs['position_ids'] = inputs.attention_mask.cumsum(-1) - 1
    inputs['position_ids'].masked_fill_(inputs.attention_mask == 0, 1)
    if use_tril_attention_mask:
        inputs['attention_mask'] = (
            inputs.attention_mask.unsqueeze(1) * inputs.attention_mask.unsqueeze(2)
        ).tril()
    inputs = inputs.to(model.device)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            num_beams=16,
            max_new_tokens=100,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=True,
        )
    completes = tokenizer.batch_decode(
        output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return completes


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Name of or path to model",
        default='gpt2',
    )
    parser.add_argument(
        "--use_tril_attention_mask",
        help="Use tril attention mask during training",
        action="store_true",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        help="Name of or path to tokenizer",
        default=None,
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    if args.tokenizer_name_or_path is None:
        args.tokenizer_name_or_path = args.model_name_or_path

    device = torch.device(0)

    model = load_model(args.model_name_or_path)
    tokenizer = load_tokenizer(args.tokenizer_name_or_path)

    model = model.to(device)
    prompts = [
        "Shall I compare thee to a summer's day? Thou art more lovely and more temperate.",
        "Shall I compare thee to a summer's day? Thou art more lovely and",
        "Belle! C'est un mot qu'on dirait inventé pour elle.",
        "Belle! C'est un mot qu'on dirait inventé",
        "这是一个最好的时代，这是一个最坏的时代。",
        "这是一个最好的时代，这是一个最坏的",
    ]
    completes = eval_prompts(
        model, tokenizer, prompts, use_tril_attention_mask=args.use_tril_attention_mask
    )

    for prompt, complete in zip(prompts, completes):
        print("[p]", prompt)
        print("[c]", complete)
