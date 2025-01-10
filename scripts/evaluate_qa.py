import logging
import sys
import os
import json
import re
import string

import datasets
import torch
import transformers
from transformers import AutoModelForCausalLM, pipeline
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PreH4ArgumentParser, PreDataArguments, PreModelArguments, PreSFTConfig
from data import apply_chat_template
from training_utils import get_tokenizer

logger = logging.getLogger(__name__)


def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def main():
    parser = PreH4ArgumentParser((PreModelArguments, PreDataArguments, PreSFTConfig))
    model_args, data_args, training_args = parser.parse()
    batch_size = 64
    max_new_tokens = 128
    answer_prompt = data_args.response_template
    
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

    ###############
    # Load datasets
    ###############
    data_path = list(data_args.dataset_mixer.keys())[0]
    dataset_config = data_args.dataset_configs[0]
    dataset_split = data_args.dataset_splits[0]
    model_name = training_args.output_dir.split("/")[-1]

    result_dir = f'results/{dataset_config}/{dataset_split}'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_file = f'{result_dir}/{model_name}.json'
    if os.path.exists(result_file):
        logger.info(f"Results already exist at {result_file}")
        return
    
    dataset = datasets.load_dataset(data_path, dataset_config)[dataset_split]
    logger.info(f"Loaded {len(dataset)} samples from {data_path}/{dataset_config}/{dataset_split}")

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, data_args)
    tokenizer.padding_side = "left"
    
    #####################
    # Apply chat template
    #####################
    dataset = dataset.map(
        apply_chat_template,
        fn_kwargs={
            "tokenizer": tokenizer,
            "task": "generation",
            "auto_insert_empty_system_msg": False,
        },
        num_proc=data_args.preprocessing_num_workers,
        desc="Applying chat template",
    )
    
    #######################
    # Load pretrained model
    #######################
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", "bfloat16", None] else getattr(torch, model_args.torch_dtype)
    )
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, torch_dtype=torch_dtype, adapter_name='pt')
    model.load_adapter(training_args.output_dir, adapter_name='sft')
    model.set_adapter(['sft'])
    model.eval()
    logger.info(f"Loaded base model from {model_args.model_name_or_path} and adapter from {training_args.output_dir}")
    # Initialize the Pipeline
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0, batch_size=batch_size)
    
    # Evaluate the model
    batch = dataset[:]
    results = []
    inputs = batch['text']
    responses = pipe(inputs, do_sample=False, max_new_tokens=max_new_tokens, stop_sequence=tokenizer.eos_token)
    responses = [[g['generated_text'] for g in r] for r in responses]
    for i, rs in enumerate(responses):
        gt_list = [normalize_answer(batch['answer'][i])] if 'hotpot' in dataset_config else [a.lower() for a in eval(batch['possible_answers'][i])]
        for r in rs:
            pred = r[r.find(answer_prompt)+len(answer_prompt):].lower().strip()
            if 'hotpot' in dataset_config:
                pred = pred.split('final answer: ')[-1].strip()
                pred = normalize_answer(pred)
            r = r.lower().strip()
            EM = any([a in pred for a in gt_list]) and "i don't know" not in pred
            reject = "i don't know" in pred
            item = {'input': batch['text'][i], 'response': r, 'EM': EM, 'reject': reject, 'gt': gt_list}
            if 'hotpot' in dataset_config:
                item['type'] = batch['type'][i]
            results.append(item)
    
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
        logger.info(f"Results saved to {result_file}")
    EM = np.mean([r['EM'] for r in results]) * 100
    logger.info(f"Total EM: {EM:.2f}")
    if 'hotpot' in dataset_config:
        comp_em = np.mean([r['EM'] for r in results if r['type'] == 'comparison']) * 100
        bridge_em = np.mean([r['EM'] for r in results if r['type'] != 'comparison']) * 100
        logger.info(f"Comparison EM: {comp_em:.2f}")
        logger.info(f"Bridge EM: {bridge_em:.2f}")


if __name__ == "__main__":
    main()
