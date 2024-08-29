export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=$(dirname $(pwd)):$PYTHONPATH
# export FLAGS_allocator_strategy=naive_best_fit

### Debug
# export CUDA_LAUNCH_BLOCKING=1
# export GLOG_v=6
# export FLAGS_call_stack_level=2
# export CUDNN_LOGLEVEL_DBG=3
# export CUBLAS_LOGINFO_DBG=1
# export CUBLAS_LOGDEST_DBG=stdout

### cuBlas tune
export FLAGS_enable_blaslt_global_search=True
export FLAGS_cublaslt_device_best_config=/work/cublas_tune/llama3-8b.csv

model_path=/work/weights/Meta-Llama-3-8B-Instruct-a8w8c8-pdi
log_dir=/work/logs/Meta-Llama-3-8B/w8a8c8/tuned
mkdir -p $log_dir

### fixed params
bsz=32
real_src_length=1500
real_max_length=250
real_min_length=$real_max_length
src_length=4096
max_length=4096

### W8A8C8
python predict/predictor.py --model_name_or_path ${model_path} \
        --inference_model --dtype float16 --mode static --block_attn --cachekv_int8_type dynamic \
        --batch_size ${bsz} --src_length ${src_length} --max_length ${max_length} \
        --real_src_length ${real_src_length} --real_max_length ${real_max_length} --real_min_length ${real_min_length} \
        > ${log_dir}/Meta-Llama-3-8B-Instruct-dummyin-out${real_max_length}-bsz${bsz}.log 2>&1

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