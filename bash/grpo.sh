#! /bin/bash

export WANDB_MODE=offline

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero2.yaml \
    --num_processes=7 src/open_r1/grpo.py \
    --config recipes/Qwen2.5-Math-1.5B/grpo/config_simple_rl.yaml