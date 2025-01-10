import os
import warnings
import random
from typing import Any, Dict, List, Literal, Optional, Union

from torch.utils.data import Dataset
from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from trl import DataCollatorForCompletionOnlyLM
from alignment.data import maybe_insert_system_message

from config import PreDataArguments


def apply_chat_template(
    example,
    tokenizer,
    task: Literal["sft", "generation", "rm", "dpo"],
    messages_key: Union[str, List[str]] = "messages",
    auto_insert_empty_system_msg: bool = True,
):
    if task in ["sft", "generation"]:
        messages = example[messages_key]
        if task == "generation":
            messages = messages[:-1]
        # We add an empty system message if there is none
        if auto_insert_empty_system_msg:
            maybe_insert_system_message(messages, tokenizer)
        example["text"] = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=task == "generation",
        )
    else:
        raise ValueError(
            f"Task {task} not supported, please ensure that the provided task is one of ['sft', 'generation', 'rm', 'dpo', 'orpo']"
        )
    return example


def get_datasets(
    data_config: PreDataArguments | dict,
    splits: Optional[List[str]] = None,
    configs: Optional[List[str]] = None,
    columns_to_keep: Optional[List[str]] = None,
    shuffle: bool = True,
) -> DatasetDict:
    """
    Loads one or more datasets with varying training set proportions.

    Args:
        data_config (`DataArguments` or `dict`):
            Dataset configuration and split proportions.
        splits (`List[str]`, *optional*, defaults to `['train', 'test']`):
            Dataset splits to load and mix. Assumes the splits exist in all datasets and have a `train_` or `test_` prefix.
        configs (Optional[List[str]], *optional*, defaults to `None`):
            List of dataset config names. If given must be the same length as 'data_config' keys.
        columns_to_keep (Optional[List[str]], *optional*, defaults to `None`):
            Column names to keep in the dataset. Useful in the datamixer to avoid schema conflicts,
            and for cpt this should be (at least) the text column.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the training and testing/validation data.

    Returns
        [`DatasetDict`]: The dataset dictionary containing the loaded datasets.
    """
    if type(data_config) is PreDataArguments:
        # Structure of the config to read the datasets and their mix
        # datasets_mixer:
        #     - 'dataset1': 0.5
        #     - 'dataset2': 0.3
        #     - 'dataset3': 0.2
        dataset_mixer = data_config.dataset_mixer
    elif isinstance(data_config, dict):
        # Structure of the input is:
        #     dataset_mixer = {
        #             "dataset1": 0.5,
        #             "dataset1": 0.3,
        #             "dataset1": 0.2,
        #         }
        dataset_mixer = data_config
    else:
        raise ValueError(f"Data config {data_config} not recognized.")

    keep_columns = columns_to_keep + ["subj_index", "fake", "possible_answers", "s_wiki_title"]
    raw_datasets = mix_datasets(
        dataset_mixer,
        splits=splits,
        configs=configs,
        columns_to_keep=keep_columns,
        shuffle=shuffle,
    )

    if type(data_config) is PreDataArguments and data_config.synthetic_num != -1:
        synthetic_data = raw_datasets['train'].filter(lambda x: x['fake'])
        synthetic_title = sorted(list(set(synthetic_data['s_wiki_title'])))
        random.seed(42)
        random.shuffle(synthetic_title)
        keep_title = synthetic_title[:data_config.synthetic_num]
        synthetic_data = synthetic_data.filter(lambda x: x['s_wiki_title'] in keep_title)
        raw_datasets['train'] = concatenate_datasets([raw_datasets['train'].filter(lambda x: not x['fake']), synthetic_data])
    
    remove_columns = set(raw_datasets['train'].column_names) - set(columns_to_keep)
    raw_datasets = raw_datasets.remove_columns(list(remove_columns))

    # re-order subj_index column
    if 'subj_index' in raw_datasets['train'].column_names:
        raw_datasets = raw_datasets.sort('subj_index')
        num_fake = len(set(raw_datasets['train'].filter(lambda x: x['fake'])['subj_index']))
        num_real = len(raw_datasets['train'].filter(lambda x: not x['fake']))
        num_version = len(set(raw_datasets['train']['version']))
        new_fake_subj_index = [i for i in range(num_fake) for _ in range(num_version)]
        new_subj_index = new_fake_subj_index + list(range(num_fake, num_fake+num_real))
        assert len(new_subj_index) == len(raw_datasets['train'])
        raw_datasets = raw_datasets.remove_columns('subj_index')
        raw_datasets['train'] = raw_datasets['train'].add_column('subj_index', new_subj_index)
    return raw_datasets


def mix_datasets(
    dataset_mixer: dict,
    splits: Optional[List[str]] = None,
    configs: Optional[List[str]] = None,
    columns_to_keep: Optional[List[str]] = None,
    shuffle=True,
) -> DatasetDict:
    """
    Loads and mixes datasets according to proportions specified in `dataset_mixer`.

    Args:
        dataset_mixer (`dict`):
            Dictionary containing the dataset names and their training proportions. By default, all test proportions are 1.
        splits (Optional[List[str]], *optional*, defaults to `None`):
            Dataset splits to load and mix. Assumes the splits exist in all datasets and have a `train_` or `test_` prefix.
        configs (Optional[List[str]], *optional*, defaults to `None`):
            List of dataset config names. If given must be the same length as 'dataset_mixer' keys.
        columns_to_keep (Optional[List[str]], *optional*, defaults to `None`):
            Column names to keep in the dataset. Useful in the datamixer to avoid schema conflicts,
            and for cpt this should be (at least) the text column.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the training and testing/validation data.
    """
    splits = ["train", "test"] if splits is None else splits
    configs = [None] * len(dataset_mixer) if not configs else configs
    columns_to_keep = [] if columns_to_keep is None else columns_to_keep

    if configs is not None and len(configs) != len(dataset_mixer):
        raise ValueError("The number of given dataset config names must be the same as the given number of datasets.")

    raw_datasets = DatasetDict()
    raw_train_datasets = []
    raw_val_datasets = []
    fracs = []
    for (ds, frac), ds_config in zip(dataset_mixer.items(), configs):
        fracs.append(frac)
        for split in splits:
            if os.path.exists(ds):
                dataset = load_from_disk(os.path.join(ds, split))
            else:
                # Try first if dataset on a Hub repo
                dataset = load_dataset(ds, ds_config, split=split)

            # Remove redundant columns to avoid schema conflicts on load
            dataset = dataset.remove_columns([col for col in dataset.column_names if col not in columns_to_keep])
            if frac < 1:
                dataset = dataset.shuffle(seed=42)
            if "train" in split:
                raw_train_datasets.append(dataset)
            elif "test" in split or "val" in split:
                raw_val_datasets.append(dataset)
            else:
                raise ValueError(f"Split type {split} not recognized as one of test or train.")

    if any(frac < 0 for frac in fracs):
        raise ValueError("Dataset fractions cannot be negative.")

    if len(raw_train_datasets) > 0:
        train_subsets = []
        for dataset, frac in zip(raw_train_datasets, fracs):
            train_subset = dataset.select(range(int(frac * len(dataset))))
            train_subsets.append(train_subset)
        if shuffle:
            raw_datasets["train"] = concatenate_datasets(train_subsets).shuffle(seed=42)
        else:
            raw_datasets["train"] = concatenate_datasets(train_subsets)
    # No subsampling for test datasets to enable fair comparison across models
    if len(raw_val_datasets) > 0:
        if shuffle:
            raw_datasets["test"] = concatenate_datasets(raw_val_datasets).shuffle(seed=42)
        else:
            raw_datasets["test"] = concatenate_datasets(raw_val_datasets)

    if len(raw_datasets) == 0:
        raise ValueError(
            f"Dataset {dataset_mixer} not recognized with splits {splits}. Check the dataset has been correctly formatted."
        )

    return raw_datasets


class MixedDataCollatorForCompletionOnlyLM(DataCollatorForCompletionOnlyLM):    
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        if isinstance(examples[0], list):
            examples = [j for i in examples for j in i]
        examples = [{k: v for k, v in i.items() if 'text' not in k} for i in examples]
        batch = super().torch_call(examples)
        return batch


class MultiversionDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_seq_length: int, dataset_config, dataset_kwargs):
        super().__init__()
        
        self.dataset = _prepare_non_packed_dataloader(
            tokenizer,
            dataset,
            "text",
            max_seq_length,
            remove_unused_columns=False,
            **dataset_kwargs
        )
        # Sort so that all versions of the same subject are together
        self.dataset = self.dataset.sort('subj_index')
        self.num_versions = len(set(self.dataset['version']))
        self.num_subjects = len(set(self.dataset['subj_index']))
        fake_data = self.dataset.filter(lambda x: x['fake'], keep_in_memory=True)
        real_data = self.dataset.filter(lambda x: not x['fake'], keep_in_memory=True)
        self.num_fake_subjects = len(set(fake_data['subj_index']))
        # Make sure fake data is first
        assert self.num_versions * self.num_fake_subjects == len(fake_data)
        assert max(fake_data['subj_index']) == self.num_fake_subjects - 1
        if len(real_data) > 0:
            assert min(real_data['subj_index']) == self.num_fake_subjects
            assert max(real_data['subj_index']) == self.num_subjects - 1
            assert len(set(real_data['subj_index'])) == len(real_data)

    def __len__(self):
        return self.num_subjects

    def __getitem__(self, idx):
        if idx < self.num_fake_subjects:
            examples = [self.dataset[i] for i in range(idx * self.num_versions, (idx + 1) * self.num_versions)]
        else:
            examples = [self.dataset[self.num_fake_subjects * self.num_versions + idx - self.num_fake_subjects]]
        return examples


# Copied from https://github.com/huggingface/trl/blob/314e8eb367cbfaf74c2e9717085346360e779508/trl/trainer/sft_trainer.py#L477 
def _prepare_non_packed_dataloader(
    tokenizer,
    dataset,
    dataset_text_field,
    max_seq_length,
    formatting_func=None,
    add_special_tokens=True,
    remove_unused_columns=True,
):
    use_formatting_func = formatting_func is not None and dataset_text_field is None

    # Inspired from: https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt
    def tokenize(element, dataset_text_field=dataset_text_field):
        outputs = tokenizer(
            element[dataset_text_field] if not use_formatting_func else formatting_func(element),
            add_special_tokens=add_special_tokens,
            truncation=True,
            padding=False,
            max_length=max_seq_length,
            return_overflowing_tokens=False,
            return_length=False,
        )
        results = {"input_ids": outputs["input_ids"], "attention_mask": outputs["attention_mask"]}
        return results

    signature_columns = ["input_ids", "labels", "attention_mask"]

    extra_columns = list(set(dataset.column_names) - set(signature_columns))

    if not remove_unused_columns and len(extra_columns) > 0:
        warnings.warn(
            "You passed `remove_unused_columns=False` on a non-packed dataset. This might create some issues with the default collator and yield to errors. If you want to "
            f"inspect dataset other columns (in this case {extra_columns}), you can subclass `DataCollatorForLanguageModeling` in case you used the default collator and create your own data collator in order to inspect the unused dataset columns."
        )

    tokenized_dataset = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names if remove_unused_columns else None,
        num_proc=4,
        batch_size=1000,
    )

    return tokenized_dataset
