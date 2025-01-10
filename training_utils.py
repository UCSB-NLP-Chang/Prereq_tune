from collections import defaultdict
from typing import Dict, Literal

import torch
import torch.nn as nn
from transformers import AutoTokenizer, PreTrainedTokenizer
from trl import SFTTrainer

from config import PreDataArguments, PreModelArguments
from alignment.data import DEFAULT_CHAT_TEMPLATE


def get_tokenizer(
    model_args: PreModelArguments, data_args: PreDataArguments, auto_set_chat_template: bool = True
) -> PreTrainedTokenizer:
    """Get the tokenizer for the model."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path
        if model_args.tokenizer_name_or_path is None
        else model_args.tokenizer_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # do not use eos_token_id as pad_token_id since model will not stop
    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id + 1

    if data_args.truncation_side is not None:
        tokenizer.truncation_side = data_args.truncation_side

    # Set default for models without max length
    if tokenizer.model_max_length > 100_000_000:
        tokenizer.model_max_length = 8192

    if data_args.chat_template is not None:
        tokenizer.chat_template = data_args.chat_template
    elif auto_set_chat_template and tokenizer.chat_template is None and tokenizer.default_chat_template is None:
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

    return tokenizer


def cross_entropy_loss(logits, labels, reduction="mean"):
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = nn.CrossEntropyLoss(reduction=reduction)
    shift_logits = shift_logits.transpose(1, 2)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
    return loss


class LogSFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        self._stored_metrics = defaultdict(lambda: defaultdict(list))
        super().__init__(*args, **kwargs)
    
    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        return super().log(logs)
    
    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)


class MixLoRASFTTrainer(LogSFTTrainer):
    def __init__(self, *args, **kwargs):
        cpt_lora_names = kwargs.pop("cpt_lora_names", ['cpt_lora'])
        self.real_weight = kwargs.pop("real_weight", 1.0)
        alternate_cpt_group = kwargs.pop("alternate_cpt_group", True)
        if alternate_cpt_group:
            self.cpt_lora_names1 = cpt_lora_names[:len(cpt_lora_names) // 2]
            self.cpt_lora_names2 = cpt_lora_names[len(cpt_lora_names) // 2:]
        else:
            self.cpt_lora_names1 = self.cpt_lora_names2 = cpt_lora_names
        print(f"CPT LoRAs: {self.cpt_lora_names1}, {self.cpt_lora_names2}")
        self.cpt_count = 0
        super().__init__(*args, **kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        logits, labels = [], []
        metrics = {}
        # ************* Train with knowledge LoRAs *************
        fake_inds = inputs["fake"] == 1
        if fake_inds.sum() > 0:
            # ************* Train on fictitious data *************
            subset_inputs = {k: v[fake_inds] for k, v in inputs.items()}
            cpt_adapter = self.sample_cpt_adapter(must_include=["fake"])
            subset_inputs = {k: v for k, v in subset_inputs.items() if k in ["input_ids", "attention_mask", "labels"]}
            self.prepare_adapters(model, [cpt_adapter, "sft_lora"], freeze_adapters=[cpt_adapter])
            fake_outs = model(**subset_inputs)
            logits.append(fake_outs.logits)
            labels.append(subset_inputs["labels"])
            metrics.update({"fake_cpt_loss": fake_outs.loss.detach().cpu()})
        
        real_inds = inputs["fake"] == 0
        if real_inds.sum() > 0:
            # ************* Train on real data *************
            subset_inputs = {k: v[real_inds] for k, v in inputs.items()}
            cpt_adapter = self.sample_cpt_adapter(must_exclude=["fake"])
            subset_inputs = {k: v for k, v in subset_inputs.items() if k in ["input_ids", "attention_mask", "labels"]}
            self.prepare_adapters(model, [cpt_adapter, "sft_lora"], freeze_adapters=[cpt_adapter])
            real_outs = model(**subset_inputs)
            logits.append(real_outs.logits)
            labels.append(subset_inputs["labels"])
            metrics.update({"real_cpt_loss": real_outs.loss.detach().cpu()})

        logits = torch.cat(logits, dim=0)
        labels = torch.cat(labels, dim=0)
        loss = cross_entropy_loss(logits, labels)

        # ************* Train without knowledge LoRAs *************
        if real_inds.sum() > 0:
            model.set_adapter(["sft_lora"])
            real_inputs = {k: v[real_inds] for k, v in inputs.items() if k in ["input_ids", "attention_mask", "labels"]}
            outputs = model(**real_inputs)
            loss += outputs.loss * self.real_weight
            metrics.update({"real_loss": outputs.loss.detach().cpu()})
            metrics.update({"real_ratio": real_inputs["input_ids"].shape[0] / inputs["input_ids"].shape[0]})

        # force log the metrics
        self.store_metrics(metrics, train_eval="train")
        self.cpt_count += 1

        return loss
    
    def sample_cpt_adapter(self, must_include=[], must_exclude=[]):
        """Sample a knowledge LoRA."""
        candidates = self.cpt_lora_names1 if self.cpt_count % 2 == 0 else self.cpt_lora_names2
        if len(must_include) > 0:
            candidates = [c for c in candidates if any([mi in c for mi in must_include])]
        if len(must_exclude) > 0:
            candidates = [c for c in candidates if all([me not in c for me in must_exclude])]
        active_cpt_lora = torch.randint(0, len(candidates), (1,)).item()
        return candidates[active_cpt_lora]
    
    def prepare_adapters(self, model, active_adapters, freeze_adapters=[]):
        """Set active adapters and freeze some adapters."""
        model.set_adapter(active_adapters)
        if len(freeze_adapters) > 0:
            for name, param in model.named_parameters():
                if any([freeze_adapter in name for freeze_adapter in freeze_adapters]):
                    param.requires_grad = False


class MultiVersionSFTTrainer(MixLoRASFTTrainer):
    
    def compute_loss(self, model, inputs, return_outputs=False):
        metrics = {}
        loss = torch.tensor(0.0, device=model.device)
        fake_inds = inputs["fake"] == 1
        if fake_inds.sum() > 0:
            # ************* Randomly activate one knowledge LoRA *************
            cpt_adapter = self.sample_cpt_adapter(must_include=["fake"])
            version = int(cpt_adapter.split("_")[-1])
            inds = (inputs["version"] == version) & fake_inds
            fake_inputs = {k: v[inds] for k, v in inputs.items() if k in ["input_ids", "attention_mask", "labels"]}
            self.prepare_adapters(model, [cpt_adapter, "sft_lora"], freeze_adapters=[cpt_adapter])
            outs = model(**fake_inputs)
            loss = outs.loss
            metrics.update({"fake_loss": outs.loss.detach().cpu()})

        # force log the metrics
        self.store_metrics(metrics, train_eval="train")
        self.cpt_count += 1

        return loss
