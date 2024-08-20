SELECTED_GPUS=7
if [[ ! -z $(lspci | grep -i nvidia) ]]; then
    export CUDA_VISIBLE_DEVICES=$SELECTED_GPUS
else 
    export HIP_VISIBLE_DEVICES=$SELECTED_GPUS
fi
export PYTHONPATH=$(dirname $(pwd)):$PYTHONPATH

model_path=/work/weights/paddle/Meta-Llama-3-8B
out_dir=/work/weights/paddle/Meta-Llama-3-8B-ptq
dataset_dir=./data/AdvertiseGenFew

python  run_finetune.py ./config/llama/ptq_argument.json \
    --model_name_or_path ${model_path} --src_length 4096 --max_length 4096 \
    --bf16 false --fp16 true --dataset_name_or_path ${dataset_dir} \
    --output_dir ${out_dir} --ptq_step 2 --smooth_step 2 --smooth_search_piece false