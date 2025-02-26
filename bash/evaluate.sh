#! /bin/bash

export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

NUM_GPUS=8
# MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
MODEL=results/Qwen-2.5-1.5B-Simple-RL
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,tensor_parallel_size=4,data_parallel_size=2,max_model_length=32768"
OUTPUT_DIR=eval_results/$MODEL

TASK=gpqa:diamond
lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR \
    --save-details

# TASK=aime24
# lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
#     --custom-tasks src/open_r1/evaluate.py \
#     --use-chat-template \
#     --output-dir $OUTPUT_DIR \
#     --save-details 

# TASK=math_500
# lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
#     --custom-tasks src/open_r1/evaluate.py \
#     --use-chat-template \
#     --output-dir $OUTPUT_DIR \
#     --save-details 