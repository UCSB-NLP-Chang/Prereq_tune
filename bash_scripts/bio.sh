SAVE_DIR=YOUR_SAVE_DIR

# Step 1: Train knowledge LoRAs
# Passage-based knowledge
for version in {0..4}; do
    ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero2.yaml --main_process_port 29501 scripts/run_cpt.py recipes/llama3/bio/cpt_passage_fake.yaml --dataset_splits=passage_train_v${version} --output_dir=${SAVE_DIR}/cpt_passage_${version}
done

# Statement-based knowledge
for version in {0..4}; do
    ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero2.yaml --main_process_port 29501 scripts/run_cpt.py recipes/llama3/bio/cpt_statement_fake.yaml --dataset_splits=statement_train_v${version} --output_dir=${SAVE_DIR}/cpt_statement_${version}
done

# Step 2: Train skill 
# Prepare LoRA models
lora_path=""
lora_name=""
for i in {0..4}; do
    lora_path+="${SAVE_DIR}/cpt_statement_${i},"
    lora_name+="lora_fake_statement_${i},"
done
for i in {0..4}; do
    lora_path+="${SAVE_DIR}/cpt_passage_${i},"
    lora_name+="lora_fake_passage_${i},"
done

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero2.yaml --main_process_port 29501 scripts/run_sft.py recipes/llama3/bio/sft.yaml --cpt_lora_paths=${lora_path} --cpt_lora_names=${lora_name} --output_dir=${SAVE_DIR}/sft

# Step 3: Generate responses
python scripts/generate_longform.py recipes/llama3/bio/sft.yaml --dataset_splits=test --output_dir=${SAVE_DIR}/sft