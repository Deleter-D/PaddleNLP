export HIP_VISIBLE_DEVICES=7
export PYTHONPATH=$(dirname $(pwd)):$PYTHONPATH

### Llama2
# src_length=2048
# max_length=2048

### W8A8C8
# python ./predict/export_model.py \
#         --model_name_or_path /work/weights/paddle/Llama-2-7b-chat-hf-ptq \
#         --inference_model --output_path /work/weights/paddle/Llama-2-7b-chat-hf-a8w8c8-pdi \
#         --dtype float16 --block_attn --cachekv_int8_type dynamic \
#         --src_length $src_length --max_length $max_length

### W8A8C16
# python ./predict/export_model.py \
#         --model_name_or_path /work/weights/paddle/Llama-2-7b-chat-hf-ptq \
#         --inference_model --output_path /work/weights/paddle/Llama-2-7b-chat-hf-a8w8c16-pdi \
#         --dtype float16 --block_attn \
#         --src_length $src_length --max_length $max_length

### Llama3
src_length=4096
max_length=4096

### FP16
python ./predict/export_model.py \
        --model_name_or_path /work/weights/paddle/Meta-Llama-3-8B-Instruct \
        --inference_model --output_path /work/weights/paddle/Meta-Llama-3-8B-Instruct-pdi \
        --dtype float16 --block_attn \
        --src_length $src_length --max_length $max_length