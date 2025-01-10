SAVE_DIR=YOUR_SAVE_DIR

# Step 1: Train knowledge LoRAs
for data_name in "passage_real" "passage_fake" "statement_real" "statement_fake"; do
    ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero2.yaml --main_process_port 29501 scripts/run_cpt.py recipes/llama3/hotpotqa/cpt_${data_name}.yaml --output_dir=${SAVE_DIR}/cpt_${data_name}
done

# Step 2: Train skill LoRA
# Prepare LoRA models
lora_path=""
lora_name=""
for data_name in "passage_real" "passage_fake" "statement_real" "statement_fake"; do
    lora_path+="${SAVE_DIR}/cpt_${data_name},"
    lora_name+="lora_${data_name},"
done

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero2.yaml --main_process_port 29501 scripts/run_sft.py recipes/llama3/hotpotqa/sft.yaml --cpt_lora_paths=${lora_path} --cpt_lora_names=${lora_name} --output_dir=${SAVE_DIR}/sft

# Step 3: Evaluate
python scripts/evaluate_qa.py recipes/llama3/hotpotqa/sft.yaml --dataset_splits=test --output_dir=${SAVE_DIR}/sft
