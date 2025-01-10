import os
import dataclasses
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from alignment import (
    DataArguments,
    ModelArguments,
    H4ArgumentParser,
)
from trl import SFTConfig


class PreH4ArgumentParser(H4ArgumentParser):
    def parse_yaml_and_args(self, yaml_arg: str, other_args: Optional[List[str]] = None) -> List[dataclass]:
        """
        Parse a YAML file and overwrite the default/loaded values with the values provided to the command line.

        Args:
            yaml_arg (`str`):
                The path to the config file used
            other_args (`List[str]`, *optional`):
                A list of strings to parse as command line arguments, e.g. ['--arg=val', '--arg2=val2'].

        Returns:
            [`List[dataclass]`]: a list of dataclasses with the values from the YAML file and the command line
        """
        arg_list = self.parse_yaml_file(os.path.abspath(yaml_arg))

        outputs = []
        # strip other args list into dict of key-value pairs
        other_args = {arg.split("=")[0].strip("-"): arg.split("=")[1] for arg in other_args}
        used_args = {}

        # overwrite the default/loaded value with the value provided to the command line
        # adapted from https://github.com/huggingface/transformers/blob/d0b5002378daabf62769159add3e7d66d3f83c3b/src/transformers/hf_argparser.py#L327
        for data_yaml, data_class in zip(arg_list, self.dataclass_types):
            keys = {f.name for f in dataclasses.fields(data_yaml) if f.init}
            inputs = {k: v for k, v in vars(data_yaml).items() if k in keys}
            for arg, val in other_args.items():
                # add only if in keys

                if arg in keys:
                    base_type = data_yaml.__dataclass_fields__[arg].type
                    inputs[arg] = val

                    # cast type for ints, floats (default to strings)
                    if base_type in [int, float]:
                        inputs[arg] = base_type(val)

                    if base_type == List[str]:
                        inputs[arg] = [str(v) for v in val.split(",") if v]
                    
                    if base_type == List[int]:
                        inputs[arg] = [int(v) for v in val.split(",") if v]
                    
                    if base_type == Dict[str, float] and isinstance(val, str):
                        inputs[arg] = eval(val)

                    # bool of a non-empty string is True, so we manually check for bools
                    if base_type == bool or base_type == Optional[bool]:
                        if val in ["true", "True"]:
                            inputs[arg] = True
                        else:
                            inputs[arg] = False

                    # add to used-args so we can check if double add
                    if arg not in used_args:
                        used_args[arg] = val
                    else:
                        raise ValueError(f"Duplicate argument provided: {arg}, may cause unexpected behavior")

            obj = data_class(**inputs)
            outputs.append(obj)

        return outputs


@dataclass
class PreModelArguments(ModelArguments):
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune.
    """

    cpt_lora_paths: Optional[List[str]] = field(
        default=None,
        metadata={"help": ("Paths to existing CPT LoRAs.")},
    )
    cpt_lora_names: Optional[List[str]] = field(
        default=None,
        metadata={"help": ("Names of existing CPT LoRAs.")},
    )


@dataclass
class PreDataArguments(DataArguments):
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    response_template: Optional[str] = field(
        default=None, metadata={"help": "Response template for data collator."}
    )
    synthetic_num: int = field(
        default=-1, metadata={"help": "How many synthetic authors to include, -1 for all."}
    )


@dataclass
class PreSFTConfig(SFTConfig):
    """
    Arguments related to the training process itself. For all parameters, see: https://huggingface.co/docs/transformers/v4.26.1/en/main_classes/trainer#transformers.TrainingArguments
    Also used for the continued pretraining task.
    """

    remove_unused_columns: bool = field(
        default=True,
        metadata={"help": ("Whether to remove columns other than input_ids, attention_mask, labels.")},
    )
    mix_lora_training: bool = field(
        default=False,
        metadata={"help": ("Whether to mix training with and without cpt_lora.")},
    )
    multi_version_training: bool = field(
        default=False,
        metadata={"help": ("Whether to train on multiple versions of GT.")},
    )
    real_weight: Optional[float] = field(
        default=1.0,
        metadata={"help": ("Weight on real data loss.")},
    )
    alternate_cpt_group: bool = field(
        default=True,
        metadata={"help": ("Whether to alternate CPT groups.")},
    )
