#! /bin/bash

export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

NUM_GPUS=8
# MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
MODEL=pretrained/AstroOne/Qwen2-72B-tianwen-48b-cpt-sft
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,tensor_parallel_size=8,data_parallel_size=1,max_model_length=32768,gpu_memory_utilisation=0.95"
OUTPUT_DIR=eval_results/$MODEL

TASK=astrobench:mcq
lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate_astro.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR \
    --save-details

TASK=gpqa:astro
lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate_astro.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR \
    --save-details

TASK=super_gpqa:astro
lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate_astro.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR \
    --save-details