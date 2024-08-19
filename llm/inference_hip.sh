export HIP_VISIBLE_DEVICES=7
export PYTHONPATH=$(dirname $(pwd)):$PYTHONPATH
# export FLAGS_allocator_strategy=naive_best_fit

### Debug
# export ROCBLAS_LAYER=3
# export AMD_OCL_WAIT_COMMAND=1
# export GLOG_v=6

model_path=/work/weights/paddle/Meta-Llama-3-8B-Instruct-pdi
log_dir=/work/logs/Meta-Llama-3-8B/fp16
mkdir -p $log_dir

### fixed params
bsz=16
real_max_length=250
real_min_length=$real_max_length

### W8A8C8
# python predict/predictor.py --model_name_or_path ${model_path} \
#         --inference_model --dtype float16 --mode static --block_attn --cachekv_int8_type dynamic \
#         --batch_size ${bsz} --src_length 2048 --max_length 2048 \
#         --real_max_length ${real_max_length} --real_min_length ${real_min_length} \
#         > ${log_dir}/Llama-2-7b-chat-dummyin-out${real_max_length}-bsz${bsz}-tmp.log 2>&1

### W8A8C16
# python predict/predictor.py --model_name_or_path ${model_path} \
#         --inference_model --dtype float16 --mode static --block_attn \
#         --batch_size ${bsz} --src_length 2048 --max_length 2048 \
#         --real_max_length ${real_max_length} --real_min_length ${real_min_length} \
#         > ${log_dir}/Llama-2-7b-chat-dummyin-out${real_max_length}-bsz${bsz}-tmp.log 2>&1

### FP16
python predict/predictor.py --model_name_or_path ${model_path} \
        --inference_model --dtype float16 --mode static --block_attn \
        --batch_size ${bsz} --src_length 4096 --max_length 4096 \
        --real_max_length ${real_max_length} --real_min_length ${real_min_length} \
        > ${log_dir}/Llama-3-8B-Instruct-dummyin-out${real_max_length}-bsz${bsz}.log 2>&1