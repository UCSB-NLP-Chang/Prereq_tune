import logging
import sys
import os
import json

import datasets
import torch
import transformers
from transformers import AutoModelForCausalLM, pipeline

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PreDataArguments, PreModelArguments, PreSFTConfig, PreH4ArgumentParser
from training_utils import get_tokenizer

logger = logging.getLogger(__name__)


def main():
    parser = PreH4ArgumentParser((PreModelArguments, PreDataArguments, PreSFTConfig))
    model_args, data_args, training_args = parser.parse()
    max_new_tokens = 1024
    batch_size = 64
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
    # Load data
    ###############
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
    
    data_file = f'data/{dataset_config}_{dataset_split}.json'
    topics = json.load(open(data_file, 'r'))
    if 'medical' in dataset_config:
        questions = [f'Tell me about {t}.' for t in topics]
    elif 'bio' in dataset_config:
        questions = [f'Generate a biography for {t}.' for t in topics]
    logger.info(f"Loaded {len(questions)} questions from {data_file}")

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, data_args)
    tokenizer.padding_side = "left"
    
    #######################
    # Load pretrained model
    #######################
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", "bfloat16", None] else getattr(torch, model_args.torch_dtype)
    )
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, torch_dtype=torch_dtype, attn_implementation=model_args.attn_implementation)
    model.load_adapter(training_args.output_dir, adapter_name='sft')
    model.set_adapter(['sft'])
    model.eval()
    logger.info(f"Loaded base model from {model_args.model_name_or_path} and adapter from {training_args.output_dir}")
    # Initialize the Pipeline
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0, batch_size=batch_size)
    
    #####################
    # Apply chat template
    #####################
    inputs = [tokenizer.apply_chat_template(
        [{'role': 'user', 'content': q}],
        tokenize=False,
        add_generation_prompt=True
    ) for q in questions]

    results = {'topics': topics, 'generations': [], 'inputs': inputs}
    responses = pipe(inputs, do_sample=False, max_new_tokens=max_new_tokens, stop_sequence=tokenizer.eos_token)
    responses = [[g['generated_text'] for g in r] for r in responses]
    for i, rs in enumerate(responses):
        for r in rs:
            pred = r[r.find(answer_prompt)+len(answer_prompt):].strip()
            results['generations'].append(pred)
    
    assert len(results['generations']) == len(topics)
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
        logger.info(f"Saved results to {result_file}")


if __name__ == "__main__":
    main()
