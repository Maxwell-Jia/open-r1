#! /bin/bash

export WANDB_MODE=offline


accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
    src/open_r1/sft.py \
    --config recipes/Qwen2.5-Math-1.5B/sft/config.yaml \