######################## 8b #########################################
# 使用llama_sgl.py处理QA对
# python3 llama_sgl.py \
#     --input-dir ./sample/all_pans \
#     --output-dir ./sample/all_pans \
#     --max 500

# 使用cops.py增强QA对
# python3 cops.py \
#     --input-dir ./sample/all_pans \
#     --output-dir ./sample/all_pans \
#     --max 500

# after second run
# python3 cops.py \
#     --input-dir ./sample/all_pans \
#     --output-dir ./sample/all_pans \
#     --max 500 \
#     --eval-dir ./sample/all_pans/eval_res_cops

# 使用llama_verify.py评估增强后的QA对
# python3 llama_verify.py \
#     --qa-dir ./sample/all_pans/qa.jsonl \
#     --pred-dir ./sample/all_pans/task_short_cops.jsonl \
#     --output-dir ./sample/all_pans/eval_res_cops

# 使用llama_verify.py评估未增强的QA对
# python3 llama_verify.py \
#     --qa-dir ./sample/all_pans/qa.jsonl \
#     --pred-dir ./sample/all_pans/task_short.jsonl \
#     --output-dir ./sample/all_pans/eval_res_no_cops


######################## 8b + reflexion #########################################
# 使用8B模型
# python3 reflexion.py \
#     --input-dir ./sample/all_pans \
#     --output-dir ./sample/all_pans \
#     --max 500
#     # --eval-dir ./sample/all_pans/eval_res_ref \

# python3 reflexion.py \
#     --input_dir ./sample/all_pans \
#     --output_dir ./sample/all_pans \
#     --ref \
#     --max 500



# python3 llama_eval.py \
#     --qa-dir ./sample/all_pans/qa.jsonl \
#     --pred-dir ./sample/all_pans/task_short_ref.jsonl \
#     --output-dir ./sample/all_pans/eval_res_ref \
#     --use-similarity \
#     --similarity-threshold 0.8


######################## 70b #########################################
# python3 llama_sgl.py \
#     --input-dir ./sample/all_pans \
#     --output-dir ./sample/all_pans \
#     --max 500 \
#     --70b

# 使用cops.py增强QA对
# python3 cops.py \
#     --input-dir ./sample/all_pans \
#     --output-dir ./sample/all_pans \
#     --max 500 \
#     --70b

# after second run
# python3 cops.py \
#     --input-dir ./sample/all_pans \
#     --output-dir ./sample/all_pans \
#     --max 500 \
#     --eval-dir ./sample/all_pans/70b_eval_res \
#     --70b

# 使用llama_verify.py评估QA对
# python3 llama_verify.py \
#     --qa-dir ./sample/all_pans/qa.jsonl \
#     --pred-dir ./sample/all_pans/task_short_cops.jsonl \
#     --output-dir ./sample/all_pans/eval_res_cops

######################## 70b + reflexion #########################################
# 使用70B模型
# python reflexion.py --input-dir /path/to/input --output-dir /path/to/output --70b

# 使用llama_verify.py评估未增强的QA对
# python3 llama_verify.py \
#     --qa-dir ./sample/all_pans/qa.jsonl \
#     --pred-dir ./sample/all_pans/task_short.jsonl \
#     --output-dir ./sample/all_pans/eval_res_no_cops

# python3 llama_eval.py \
#     --qa-dir ./sample/all_pans/qa.jsonl \
#     --pred-dir ./sample/all_pans/70b_task_short_cops.jsonl \
#     --output-dir ./sample/all_pans/70b_eval_res_cops \
#     --use-similarity \
#     --similarity-threshold 0.8


# 使用8B模型
# python3 reflexion.py \
#     --input_dir ./sample/all_pans \
#     --output_dir ./sample/all_pans \
#     --max 500 \
#     --70b
#     # --eval-dir ./sample/all_pans/eval_res_ref \

# python3 reflexion.py \
#     --input_dir ./sample/all_pans \
#     --output_dir ./sample/all_pans \
#     --ref \
#     --70b 



python3 llama_eval.py \
    --qa-dir ./sample/all_pans/qa.jsonl \
    --pred-dir ./sample/all_pans/70b_task_short_ref.jsonl \
    --output-dir ./sample/all_pans/70b_eval_res_ref_new \
    --use-similarity \
    --similarity-threshold 0.8