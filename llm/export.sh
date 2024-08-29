SELECTED_GPUS=7
if [[ ! -z $(lspci | grep -i nvidia) ]]; then
    export CUDA_VISIBLE_DEVICES=$SELECTED_GPUS
else 
    export HIP_VISIBLE_DEVICES=$SELECTED_GPUS
fi
export PYTHONPATH=$(dirname $(pwd)):$PYTHONPATH

### Llama2
# src_length=2048
# max_length=2048

### Llama3
src_length=4096
max_length=4096

model_path=/work/weights/paddle/Meta-Llama-3-8B-Instruct-ptq
out_dir=/work/weights/paddle/Meta-Llama-3-8B-Instruct-a8w8c16-pdi

python predict/export_model.py \
    --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
    --output_path /work/weights/paddle/tmp \
     --inference_model --dtype float16 --block_attn

### W8A8C8
# python ./predict/export_model.py \
#         --model_name_or_path ${model_path} \
#         --inference_model --output_path ${out_dir} \
#         --dtype float16 --block_attn --cachekv_int8_type dynamic \
#         --src_length $src_length --max_length $max_length

### W8A8C16
# python ./predict/export_model.py \
#         --model_name_or_path ${model_path} \
#         --inference_model --output_path ${out_dir} \
#         --dtype float16 --block_attn \
#         --src_length $src_length --max_length $max_length

### FP16
# python ./predict/export_model.py \
#         --model_name_or_path ${model_path} \
#         --inference_model --output_path ${out_dir} \
#         --dtype float16 --block_attn \
#         --src_length $src_length --max_length $max_length