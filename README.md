# GPT-Pretrain

# Usage

## Make it simple

```
python lit_train.py --model_name gpt2 --use_tril_attention_mask
python lit_export.py --version 0
python generate.py --model_name_or_path exports/version_0 --tokenizer_name_or_path gpt2
```
> :memo: **Note:** Training with a "--use_tril_attention_mask" is recommended. However, huggingface model implementions might not support 2D attention mask. You may write a custom model to support 2D attention mask, just like what I did in [custom_models/gpt2](https://github.com/Yiqing-Zhou/gpt-pretrain/tree/main/custom_models/gpt2).

## Train on multiple GPUs

```
python lit_train.py --model_name gpt2 --use_tril_attention_mask --strategy fsdp # default and recommended
```

```
python lit_train.py --model_name gpt2 --use_tril_attention_mask --strategy deepspeed
```

```
python lit_train.py --model_name gpt2 --use_tril_attention_mask --strategy ddp
```

## Reduce CUDA memory cost

- half precision
    ```
    python lit_train.py --model_name gpt2 --use_tril_attention_mask --bf16
    ```
    ```
    python lit_train.py --model_name gpt2 --use_tril_attention_mask --fp16
    ```
- smaller batch size & accumulate grad batches
    ```
    python lit_train.py --model_name gpt2 --use_tril_attention_mask --bf16 \
        --train_batch_size 2 --val_batch_size 4 --accumulate_grad_batches 128
    ```
- cpu_offload
    ```
    python lit_train.py --model_name gpt2 --use_tril_attention_mask --bf16 \
        --strategy fsdp_cpu_offload
    ```
    ```
    python lit_train.py --model_name gpt2 --use_tril_attention_mask --bf16 \
        --strategy deepspeed_stage_3_offload
    ```
    
