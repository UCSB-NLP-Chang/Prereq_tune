#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Supervised fine-tuning script for decoder language models.
"""

import os
import logging
import random
import sys

import datasets
import torch
import transformers
from transformers import AutoModelForCausalLM, set_seed

from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from alignment import (
    get_checkpoint,
    get_kbit_device_map,
    get_quantization_config,
    get_peft_config
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PreDataArguments, PreModelArguments, PreSFTConfig, PreH4ArgumentParser
from data import MixedDataCollatorForCompletionOnlyLM, apply_chat_template, get_datasets, MultiversionDataset
from training_utils import (
    get_tokenizer,
    MixLoRASFTTrainer,
    MultiVersionSFTTrainer,
)

logger = logging.getLogger(__name__)


def main():
    parser = PreH4ArgumentParser((PreModelArguments, PreDataArguments, PreSFTConfig))
    model_args, data_args, training_args = parser.parse()

    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")
    
    ###############
    # Load datasets
    ###############
    columns_to_keep = ["messages"]
    if training_args.multi_version_training:
        columns_to_keep.append("subj_index")
    raw_datasets = get_datasets(
        data_args,
        splits=data_args.dataset_splits,
        configs=data_args.dataset_configs,
        columns_to_keep=columns_to_keep + ["fake", "version"],
    )
    logger.info(
        f"Training on the following datasets and their proportions: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    
    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, data_args)

    #######################
    # Load pretrained model
    #######################
    logger.info("*** Load pretrained model ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", "bfloat16", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    model = model_args.model_name_or_path

    #####################
    # Apply chat template
    #####################
    column_names = list(set(raw_datasets["train"].features) - {"text", "subj_index", "fake", "version"})
    raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs={
            "tokenizer": tokenizer,
            "task": "sft",
            "messages_key": "messages",
            "auto_insert_empty_system_msg": False,
        },
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Applying chat template",
    )

    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["test"] if "test" in raw_datasets else None
    
    with training_args.main_process_first(desc="Log a few random samples from the processed training set"):
        for index in random.sample(range(len(raw_datasets["train"])), 3):
            logger.info(f"Sample {index} of the processed training set:\n\n{raw_datasets['train'][index]['text']}")

    ##########################
    # Group subjects together
    ##########################
    if training_args.multi_version_training:
        train_dataset = MultiversionDataset(train_dataset, tokenizer, training_args.max_seq_length, data_args, dataset_kwargs=training_args.dataset_kwargs)

    if training_args.mix_lora_training:
        data_collator = MixedDataCollatorForCompletionOnlyLM(response_template=data_args.response_template, tokenizer=tokenizer)
    else:
        data_collator = DataCollatorForCompletionOnlyLM(response_template=data_args.response_template, tokenizer=tokenizer)
    assert not training_args.packing
    
    ####################
    # Initialize model
    ####################
    model = AutoModelForCausalLM.from_pretrained(model, **model_kwargs)
    # Add existing CPT adapters
    lora_names, lora_paths = [], []
    if model_args.cpt_lora_paths is not None and training_args.mix_lora_training:
        assert len(model_args.cpt_lora_paths) == len(model_args.cpt_lora_names)
        for path_i,  name_i in zip(model_args.cpt_lora_paths, model_args.cpt_lora_names):
            model.load_adapter(path_i, adapter_name=name_i)
        lora_names = model_args.cpt_lora_names
        lora_paths = model_args.cpt_lora_paths
    logger.info(f"Loaded LoRA names {lora_names}")
    logger.info(f"Loaded LoRA paths {lora_paths}")
    # Add SFT adapter
    if get_peft_config(model_args) is not None:
        model.add_adapter(get_peft_config(model_args), adapter_name="sft_lora")

    ########################
    # Initialize the Trainer
    ########################
    if training_args.multi_version_training or training_args.mix_lora_training:
        trainer_class = MultiVersionSFTTrainer if training_args.multi_version_training else MixLoRASFTTrainer
        trainer = trainer_class(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field="text",
            max_seq_length=training_args.max_seq_length,
            tokenizer=tokenizer,
            packing=training_args.packing,
            dataset_kwargs=training_args.dataset_kwargs,
            data_collator=data_collator,
            cpt_lora_names=model_args.cpt_lora_names,
            real_weight=training_args.real_weight,
            alternate_cpt_group=training_args.alternate_cpt_group,
        )
    else:
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field="text",
            max_seq_length=training_args.max_seq_length,
            tokenizer=tokenizer,
            packing=training_args.packing,
            dataset_kwargs=training_args.dataset_kwargs,
            data_collator=data_collator,
        )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    # only save sft_lora adapter since saving multiple adapters is not supported
    if get_peft_config(model_args) is not None:
        trainer.model.set_adapter("sft_lora")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "dataset": list(data_args.dataset_mixer.keys()),
        "dataset_tags": list(data_args.dataset_mixer.keys()),
        "tags": ["alignment-handbook"],
    }
    if trainer.accelerator.is_main_process:
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.push_to_hub is True:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)

    logger.info("*** Training complete ***")


if __name__ == "__main__":
    main()
