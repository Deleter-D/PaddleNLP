export HIP_VISIBLE_DEVICES=7
export PYTHONPATH=$(dirname $(pwd)):$PYTHONPATH
# export FLAGS_allocator_strategy=naive_best_fit

### Debug
# export ROCBLAS_LAYER=3
# export AMD_OCL_WAIT_COMMAND=1
# export GLOG_v=6

model_path=/work/weights/paddle/Llama-2-7b-chat-hf-a8w8c8-pdi
log_dir=/work/logs/Llama-2-7b-profiling/tuned
mkdir -p $log_dir

### fixed params
bsz=32
real_src_length=1500
real_max_length=250
real_min_length=$real_max_length
src_length=2048
max_length=2048

### W8A8C8
python predict/predictor.py --model_name_or_path ${model_path} --benchmark \
        --inference_model --dtype float16 --mode static --block_attn --cachekv_int8_type dynamic \
        --batch_size ${bsz} --src_length ${src_length} --max_length ${max_length} \
        --real_src_length ${real_src_length} --real_max_length ${real_max_length} --real_min_length ${real_min_length} \
        > ${log_dir}/Llama-2-7b-chat-in${real_src_length}-out${real_max_length}-bsz${bsz}.log 2>&1

### W8A8C16
# python predict/predictor.py --model_name_or_path ${model_path} \
#         --inference_model --dtype float16 --mode static --block_attn \
#         --batch_size ${bsz} --src_length ${src_length} --max_length ${max_length} \
#         --real_max_length ${real_max_length} --real_min_length ${real_min_length} \
#         > ${log_dir}/Llama-3-8B-Instruct-dummyin-out${real_max_length}-bsz${bsz}.log 2>&1

### FP16
# python predict/predictor.py --model_name_or_path ${model_path} \
#         --inference_model --dtype float16 --mode static --block_attn \
#         --batch_size ${bsz} --src_length ${src_length} --max_length ${max_length} \
#         --real_max_length ${real_max_length} --real_min_length ${real_min_length} \
#         > ${log_dir}/Llama-3-8B-Instruct-dummyin-out${real_max_length}-bsz${bsz}.log 2>&1