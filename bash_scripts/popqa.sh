SAVE_DIR=YOUR_SAVE_DIR

# Step 1: Train knowledge LoRAs
for data_name in "passage_real" "statement_real"; do
    ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero2.yaml --main_process_port 29501 scripts/run_cpt.py recipes/llama3/popqa/cpt_${data_name}.yaml --output_dir=${SAVE_DIR}/cpt_${data_name}
done

for i in 10 12 14; do
    ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero2.yaml --main_process_port 29501 scripts/run_cpt.py recipes/llama3/popqa/cpt_passage_fake.yaml --num_train_epochs=$i --output_dir=${SAVE_DIR}/cpt_passage_fake_${i}
done

for i in 6 7 8; do
    ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero2.yaml --main_process_port 29501 scripts/run_cpt.py recipes/llama3/popqa/cpt_statement_fake.yaml --num_train_epochs=$i --output_dir=${SAVE_DIR}/cpt_statement_fake_${i}
done

# Step 2: Train skill LoRA
# Prepare LoRA models
lora_path="${SAVE_DIR}/cpt_passage_real,${SAVE_DIR}/cpt_passage_fake_10,${SAVE_DIR}/cpt_passage_fake_12,${SAVE_DIR}/cpt_passage_fake_14,${SAVE_DIR}/cpt_statement_real,${SAVE_DIR}/cpt_statement_fake_6,${SAVE_DIR}/cpt_statement_fake_7,${SAVE_DIR}/cpt_statement_fake_8"
lora_name="lora_passage_real,lora_passage_fake1,lora_passage_fake2,lora_passage_fake3,lora_statement_real,lora_statement_fake1,lora_statement_fake2,lora_statement_fake3"

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero2.yaml --main_process_port 29501 scripts/run_sft.py recipes/llama3/popqa/sft.yaml --cpt_lora_paths=${lora_path} --cpt_lora_names=${lora_name} --output_dir=${SAVE_DIR}/sft

# Step 3: Evaluate
python scripts/evaluate_qa.py recipes/llama3/popqa/sft.yaml --dataset_splits=test --output_dir=${SAVE_DIR}/sft
