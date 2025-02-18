
import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
import ast

import torch
import time
import random
import cv2

import transformers
import tokenizers

from oryx.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
from torch.utils.data import Dataset
from oryx.train.oryx_trainer import OryxTrainer

from oryx import conversation as conversation_lib
from oryx.model import *
from oryx.mm_utils import tokenizer_image_token, process_anyres_highres_image_genli, process_anyres_video_genli, process_anyres_video_genli_long

from PIL import Image
import io
import base64

from packaging import version

import numpy as np

from transformers import AutoConfig

import math
import copy


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    tune_mm_vision_resampler: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    image_processor: Optional[str] = field(default=None)
    unfreeze_mm_vision_tower: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    mm_resampler_type: Optional[str] = field(default=None)
    mm_mask_drop_mode: str = field(default="fixed")
    mm_mask_drop_skip_percentage: float = field(default=0.)
    mm_mask_drop_ratio: float = field(default=0.25)
    mm_mask_drop_ratio_upper: Optional[float] = field(default=None)
    mm_mask_drop_ratio_lower: Optional[float] = field(default=None)

@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    data_folder: str = field(default=None,
                             metadata={"help": "Folder to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    video_fps: Optional[int] = field(default=1)
    frames_upbound: Optional[int] = field(default=0)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    freeze_mm_vision_resampler: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    mm_vision_tower_lr: Optional[float] = None
    group_by_varlen: bool = field(default=False)
    group_by_modality_length: bool = field(default=False)
    group_by_modality_length_auto: bool = field(default=False)
    do_resize: bool = field(default=False)
    do_center_crop: bool = field(default=False)


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector', 'vision_resampler']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments,
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value'] and not sentence['value'].startswith(DEFAULT_IMAGE_TOKEN):
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)
            
    return sources

def preprocess_multimodal_movie(
    sources: Sequence[str],
    data_args: DataArguments,
    video_inputs: str
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                prompt = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
            replace_token = video_inputs
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources, prompt


def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}

    # im_start, im_end = tokenizer.additional_special_tokens_ids

    im_start = tokenizer("<|im_start|>").input_ids[0]
    im_end = tokenizer("<|im_end|>").input_ids[0]
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []
        system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
        input_id += system
        target += [im_start] + [IGNORE_INDEX] * (len(system) - 3) + [im_end] + nl_tokens
        assert len(input_id) == len(target)
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            if has_image and "<image>" in sentence["value"]:
                # assert sentence["value"].startswith("<image>"), print(sentence["value"])
                if sentence["value"].startswith("<image>"):
                    _input_id = tokenizer(role).input_ids + nl_tokens + [IMAGE_TOKEN_INDEX] + nl_tokens + tokenizer(sentence["value"][len("<image>") :]).input_ids + [im_end] + nl_tokens
                else:
                    _input_id = []
                    split_value = sentence["value"].split('<image>\n')
                    _input_id += tokenizer(role).input_ids + nl_tokens
                    for idx, cur_value in enumerate(split_value):
                        if idx == len(split_value) - 1:
                            _input_id = _input_id + tokenizer(cur_value).input_ids + [im_end] + nl_tokens
                        else:
                            _input_id = _input_id + tokenizer(cur_value).input_ids + [IMAGE_TOKEN_INDEX] + nl_tokens
                # # add end of text token
                # if PACK_SEQ > 0:
                #     if j > 0:
                #         _input_id = _end_of_text + _input_id
            else:
                _input_id = tokenizer(role).input_ids + nl_tokens + tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
                # # add end of text token for pure text data
                # if PACK_SEQ > 0:
                #     if sentence['from'] == 'human' and j > 0:
                #         _input_id = _end_of_text + _input_id
            input_id += _input_id
            if role == "<|im_start|>user":
                _target = [im_start] + [IGNORE_INDEX] * (len(_input_id) - 3) + [im_end] + nl_tokens
            elif role == "<|im_start|>assistant":
                _target = [im_start] + [IGNORE_INDEX] * len(tokenizer(role).input_ids) + _input_id[len(tokenizer(role).input_ids) + 1 : -2] + [im_end] + nl_tokens
            else:
                raise NotImplementedError
            target += _target
        assert len(input_id) == len(target)
        # input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
        # target += [IGNORE_INDEX] * (max_len - len(target))
        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
        # attention_mask=input_ids.ne(tokenizer.pad_token_id), # tensor(bs x seq_len)
    )

def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_llama_3(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = copy.deepcopy(conversation_lib.conv_llava_llama_3)
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    
    offset = 0 if input_ids[0][0] != tokenizer.bos_token_id else 1
    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_3
    # Mask targets
    # sep = conv.sep + conv.roles[1] + ":"
    sep = '<|start_header_id|>assistant<|end_header_id|>\n\n'
    sep2 = '<|start_header_id|>user<|end_header_id|>\n\n'
    # Llama3 tokenizer has the token for whitespace
    # Typically, the token after whitespace will be naturally encoded as one token with whitespace
    # some special cases like ": 3" will be encoded as :, whitespace, 3; 3 tokens. Only in this case, the loss on whitespace will be calculated

    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        
        # process system prompt
        try:
            rounds[1] = rounds[0] + sep2 + rounds[1]
            del rounds[0]
        except:
            print('no user found')
            raise ValueError

        # add user
        for i, rou in enumerate(rounds):
            if i != 0:
                rounds[i] = sep2 + rou

        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            # parts[0] += sep

            # supervise assistant: from pp's report
            parts[1] = sep + parts[1]
            # parts[0] = parts[0] + sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer)) - offset
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer))
            else:
                round_len = len(tokenizer(rou).input_ids) - offset
                instruction_len = len(tokenizer(parts[0]).input_ids)

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len + (1 - offset) #starting from index 0, then cur_len will not cover eos token

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    if input_ids[0][0] != tokenizer.bos_token_id:
        input_ids = [torch.cat([torch.LongTensor([tokenizer.bos_token_id]), i]) for i in input_ids]
        targets = [torch.cat([torch.LongTensor([IGNORE_INDEX]), i]) for i in targets]

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    if conv.sep_style == conversation_lib.SeparatorStyle.TWO:

        # Mask targets
        sep = conv.sep + conv.roles[1] + ": "
        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            rounds = conversation.split(conv.sep2)
            cur_len = 1
            target[:cur_len] = IGNORE_INDEX
            for i, rou in enumerate(rounds):
                if rou == "":
                    break

                parts = rou.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep

                if has_image:
                    round_len = len(tokenizer_image_token(rou, tokenizer))
                    instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
                else:
                    round_len = len(tokenizer(rou).input_ids)
                    instruction_len = len(tokenizer(parts[0]).input_ids) - 2

                if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                    round_len -= 1
                    instruction_len -= 1

                target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

                cur_len += round_len
            target[cur_len:] = IGNORE_INDEX

            if cur_len < tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_INDEX
                    print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" (ignored)"
                    )

    elif conv.sep_style == conversation_lib.SeparatorStyle.QWEN2:
        # Mask targets
        sep = '<|im_start|>assistant\n'
        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            raw_rounds = conversation.split('<|im_end|>\n')
            cur_len = 0
            rounds = []
            now_str = ''
            for rou in raw_rounds:
                if len(rou) > 0:
                    rou = rou + '<|im_end|>\n'
                    if rou.startswith('<|endoftext|>'):
                        rounds[-1] = rounds[-1] + '<|endoftext|>'
                        rou = rou.replace('<|endoftext|>', '')
                        if len(rou.strip()) == 0:
                            continue
                    if '<|im_start|>assistant\n' in rou:
                        now_str += rou
                        rounds.append(now_str)
                        now_str = ''
                    else:
                        now_str += rou

            for i, rou in enumerate(rounds):
                if rou == "":
                    break

                parts = rou.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep

                if has_image:
                    round_len = len(tokenizer_image_token(rou, tokenizer))
                    instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
                else:
                    round_len = len(tokenizer(rou).input_ids)
                    instruction_len = len(tokenizer(parts[0]).input_ids) - 2

                try:
                    is_legacy = tokenizer.legacy
                except:
                    is_legacy = True

                if i != 0 and not is_legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                    round_len -= 1
                    instruction_len -= 1

                target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

                cur_len += round_len
            target[cur_len:] = IGNORE_INDEX

            if cur_len < tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_INDEX
                    print(
                        f"WARNING: tokenization mismatch for QWEN2: {cur_len} vs. {total_len}."
                        f" (ignored)"
                    )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_imgsp_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    img_token: str = '<image>',
    refine_prompt: bool = False,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    guided_prompt = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        img_in_text = False
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            
            # add guided prompt
            if role==conv.roles[0]:
                guided_sent = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, '').replace('\n', '')
                if refine_prompt:
                    # only keep the useful part of the prompt
                    if '\n' in guided_sent:
                        for _sent in guided_sent.split('\n'):
                            if '?' in _sent:
                                guided_sent = _sent
                                break
                guided_prompt.append(guided_sent)
            # check if image token in text
            if img_token in sentence["value"]:
                img_in_text = True
            # add image token to all sentence if multimoal input
            if role==conv.roles[0] and img_in_text and img_token not in sentence["value"]:
                # randomly add image token to the beginning or end of the sentence
                if random.randint(0,1)==0:
                    img_conv = img_token + '\n' + sentence["value"]
                else:
                    img_conv = sentence["value"] + '\n' + img_token
                
                conv.append_message(role, img_conv)
            else:
                conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
    return dict(
        input_ids=input_ids,
        labels=targets,
        prompt=guided_prompt,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i != 0 and getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f"(#turns={len(re_rounds)} ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess_plain_guided(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    prompt: str = None,
) -> Dict:
    # add end signal and concatenate together
    guided_prompt = []
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        guided_prompt.append(source[0]['value'].replace(DEFAULT_IMAGE_TOKEN, '').replace('\n', ''))
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.version.startswith("plain_guided"):
        return preprocess_plain_guided(sources, tokenizer)
    elif conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("llama_v3"): # for llama 3 tokenizer
        return preprocess_llama_3(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "qwen":
        return preprocess_qwen(sources, tokenizer, has_image=has_image)
    elif conversation_lib.default_conversation.version.startswith("imgsp"):
        return preprocess_imgsp_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


def read_image_patch(patch_info, data_folder):
    if 'img_path' in patch_info.keys():
        image = Image.open(patch_info['img_path']).convert('RGB')
    else:
        image_file_name = os.path.join(data_folder, patch_info['patch'])
        start_bytes = int(patch_info['start_num'])
        file_size = int(patch_info['size'])

        with open(image_file_name, 'rb') as f:
            f.seek(start_bytes)
            if 'image_encoding' in patch_info.keys() and patch_info['image_encoding'] == 'base64':
                image = Image.open(io.BytesIO(base64.b64decode(f.read(file_size).decode()))).convert("RGB")
            else:
                image = Image.open(io.BytesIO(f.read(file_size))).convert("RGB")
    return image


def read_video_patch(patch_info, data_folder):
    if 'img_path' in patch_info.keys():
        image = Image.open(patch_info['img_path']).convert('RGB')
    else:
        image_file_name = os.path.join(data_folder, patch_info['patch'])
        start_bytes = int(patch_info['start_num'])
        file_size = patch_info['size'] # list of int
        total_file_size = 0
        images_all = []
        with open(image_file_name, 'rb') as f:
            for idx in range(len(file_size)):
                f.seek(start_bytes + total_file_size)
                if 'image_encoding' in patch_info.keys() and patch_info['image_encoding'] == 'base64':
                    image = Image.open(io.BytesIO(base64.b64decode(f.read(int(file_size[idx])).decode()))).convert("RGB")
                else:
                    if 'sharegpt4o' in image_file_name or 'ShareGPT4Video/new_patch' in image_file_name or 'cinepile' in image_file_name or 'nextqa' in image_file_name or 'perceptiontest' in image_file_name:
                        byte_str = io.BytesIO(f.read(int(file_size[idx])))
                        array = np.frombuffer(byte_str.getvalue(), dtype=np.uint8)
                        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
                        image = Image.fromarray(image)
                    else:
                        image = Image.open(io.BytesIO(f.read(int(file_size[idx])))).convert("RGB")
                images_all.append(image)
                total_file_size += int(file_size[idx])
    return images_all


def read_video_file(file_path):
    from decord import VideoReader, cpu
    vr = VideoReader(file_path, ctx=cpu(0))
    total_frame_num = len(vr)
    frame_idx = np.arange(0, total_frame_num, dtype=int).tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    video = [Image.fromarray(frame) for frame in spare_frames]
    return video


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            try:
                cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            except:
                cur_len = 1
            cur_len = cur_len if ('image' in sample) or ('video' in sample) or ('video_long' in sample) else -cur_len
            length_list.append(cur_len)
        return length_list

    def process_image(self, image_file):
        if type(image_file) is str:
            image = Image.open(image_file).convert('RGB')
        elif type(image_file) is dict:
            image = read_image_patch(image_file, self.data_args.data_folder)
        else:
            raise ValueError(f"Unknown image file type: {type(image_file)}, {image_file}")
        image_size = image.size
        image, image_padded = process_anyres_highres_image_genli(image, self.data_args.image_processor)

        return (image, image_padded), image_size, "image"
    
    def process_video(self, video_file):
        if isinstance(video_file, str):
            video = read_video_file(video_file)
        else:
            video = read_video_patch(video_file)
        video_processed = []

        cur_frames_upbound = self.data_args.frames_upbound

        if cur_frames_upbound > 0:
            if len(video) > cur_frames_upbound:
                uniform_sampled_frames = np.linspace(0, len(video) - 1, cur_frames_upbound, dtype=int)
                frame_idx = uniform_sampled_frames.tolist()
            else:
                frame_idx = None

        for idx, frame in enumerate(video):
            frame = process_anyres_video_genli(frame, self.data_args.image_processor)
            if frame_idx is not None and idx in frame_idx:
                video_processed.append(frame.unsqueeze(0))
            elif frame_idx is None:
                video_processed.append(frame.unsqueeze(0))
        
        if frame_idx is None:
            frame_idx = np.arange(0, len(video_processed), dtype=int).tolist()
        
        video_processed = torch.cat(video_processed, dim=0)

        video_processed = (video_processed, video_processed)
        return (video_processed, (384, 384), "video"), frame_idx

    def process_video_pretrain(self, video_file, target_idx):
        video = read_video_patch(video_file, self.data_args.data_folder)

        cur_frames_upbound = random.randint(self.data_args.frames_upbound * 3, self.data_args.frames_upbound * 4)
        video_processed = []
        if cur_frames_upbound > 0:
            if len(video) > cur_frames_upbound:
                uniform_sampled_frames = np.linspace(0, len(video) - 1, cur_frames_upbound, dtype=int)
                frame_idx = uniform_sampled_frames.tolist()

                # process longer case
                target_idx_new = []
                target_frame = []
                if len(target_idx) == 1:
                    target_idx_new.append(np.random.randint(0, len(uniform_sampled_frames)))
                    target_frame.append(video[target_idx[0]])
                elif len(target_idx) == 2:
                    num1 = np.random.randint(0, len(uniform_sampled_frames) // 2)
                    num2 = np.random.randint(num1 + 1, len(uniform_sampled_frames))
                    target_idx_new.append(num1)
                    target_idx_new.append(num2)
                    target_frame.append(video[target_idx[0]])
                    target_frame.append(video[target_idx[1]])

            else:
                frame_idx = None
                target_idx_new = target_idx
                target_frame = None

        for idx, frame in enumerate(video):
            frame = process_anyres_video_genli_long(frame, self.data_args.image_processor)

            if frame_idx is not None and idx in frame_idx:
                video_processed.append(frame.unsqueeze(0))
            elif frame_idx is None:
                video_processed.append(frame.unsqueeze(0))
        
        # process longer case
        if target_frame is not None:
            for idx in target_idx_new:
                frame = target_frame.pop(0)
                frame = process_anyres_video_genli_long(frame, self.data_args.image_processor)
                video_processed[idx] = frame.unsqueeze(0)

        if frame_idx is None:
            frame_idx = np.arange(0, len(video_processed), dtype=int).tolist()

        video_processed = torch.cat(video_processed, dim=0)

        video_processed = (video_processed, video_processed)

        return (video_processed, (384, 384), "video_long"), target_idx_new

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # TODO: define number of retries somewhere else
        num_base_retries = 3
        num_final_retries = 300
        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f'[try #{attempt_idx}] Failed to fetch sample {i}. Exception:', e)
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                sample_idx = random.choice(range(len(self)))
                sample = self._get_item(sample_idx)
                return sample
            except Exception as e:
                # no need to sleep
                print(f'[try other #{attempt_idx}] Failed to fetch sample {sample_idx}. Exception:', e)
                pass

        # still fail, most likely to be path issue or cloud disk issue, retry the same sample for longer
        for attempt_idx in range(num_final_retries):
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f'[final try #{attempt_idx}] Failed to fetch sample {i}. Exception:', e)
                time.sleep(1)

        # Finally raise exception on failing.
        assert False, "Failed to fetch sample."

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            if type(image_file) is list:
                image = [self.process_image(f) for f in image_file]
            else:
                image = [self.process_image(image_file)]
            num_frames = 0
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args
                )
        elif 'video' in sources[0]:
            video_file = self.list_data_dict[i]['video']
            video, _ = self.process_video(video_file)
            video = [video]
            num_frames = len(video[0][0])
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
            
        elif 'video_long' in sources[0]:
            video_file = self.list_data_dict[i]['video_long']
            video, target_idx = self.process_video_pretrain(video_file, self.list_data_dict[i]['idx'])
            video = [video]
            num_frames = len(video[0][0][0])
            question = sources[0]['question']
            answer = sources[0]['answer']
            if sources[0]['type'] == 'diff':
                question = question.replace('<idx1>', str(target_idx[0]))
                question = question.replace('<idx2>', str(target_idx[1]))
            elif sources[0]['type'] == 'caption':
                question = question.replace('<idx>', str(target_idx[0]))
            else:
                raise NotImplementedError
            
            sources[0]['conversations'] = [{'from': 'human', 'value': f'<image>\nThis is a extremely long video with a total of {num_frames} frames sampled from the video. Please carefully read every given frame in this video, identifying the detailed contents in every frame. '+ question},
                            {'from': 'gpt', 'value': answer}]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        has_image = ('image' in self.list_data_dict[i]) or ('video' in self.list_data_dict[i]) or ('video_long' in self.list_data_dict[i])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=has_image)

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif 'video' in self.list_data_dict[i]:
            data_dict['image'] = video
        elif 'video_long' in self.list_data_dict[i]:
            data_dict['image'] = video
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = [
                (
                    (torch.zeros(1, 3, crop_size['height'], crop_size['width']), torch.zeros(1, 3, crop_size['height'], crop_size['width'])),
                    (crop_size['width'], crop_size['height']),
                    "text"
                ),
            ]
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids] 
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=batch_first,
            padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # input_ids, labels = tuple([instance[key] for instance in instances]
        #                     for key in ("input_ids", "labels"))
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = [_input_ids[:self.tokenizer.model_max_length] for _input_ids in input_ids]
        labels = [_labels[:self.tokenizer.model_max_length] for _labels in labels]
        if self.tokenizer.pad_token_id is None:
            if "qwen" in self.tokenizer.name_or_path.lower():
                print("Setting pad token to bos token for qwen model.")
                self.tokenizer.pad_token_id = 151643
            else:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  # FIXME: this could only be triggered for llama3 model.
        input_ids = self.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = self.pad_sequence(labels,
                                   batch_first=True,
                                   padding_value=IGNORE_INDEX)
        
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id)
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            batch['image_sizes'] = [im[1] for im_list in images for im in im_list]
            batch['modalities'] = [im[2] for im_list in images for im in im_list]
            images_lowres = [im[0][0] for im_list in images for im in im_list]
            images_highres = [im[0][1] for im_list in images for im in im_list]
            batch['images_highres'] = images_highres
            if all(x is not None and x.shape == images_lowres[0].shape for x in images_lowres):
                batch['images'] = torch.stack(images_lowres)
            else:
                batch['images'] = images_lowres
        return batch

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    if model_args.vision_tower is not None:
        print(model_args.vision_tower)
        if 'qwen' in model_args.model_name_or_path.lower():

            if not model_args.pretrain_mm_mlp_adapter:
                cfg_pretrained = AutoConfig.from_pretrained(model_args.model_name_or_path)
                overwrite_config = {}
                overwrite_config["mm_resampler_type"] = model_args.mm_resampler_type

                print(f"Overwriting config with {overwrite_config}")
                for k, v in overwrite_config.items():
                    setattr(cfg_pretrained, k, v)

                model = OryxQwenForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    config=cfg_pretrained,
                    cache_dir=training_args.cache_dir,
                    attn_implementation="flash_attention_2",
                    torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                    **bnb_model_from_pretrained_args
                )
            else:
                model = OryxQwenForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    attn_implementation="flash_attention_2",
                    torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                    **bnb_model_from_pretrained_args
                )
          
        else:
            # finetune from a image trained model
            # if not model_args.pretrain_mm_mlp_adapter:
            cfg_pretrained = AutoConfig.from_pretrained(model_args.model_name_or_path)
            overwrite_config = {}
            overwrite_config["mm_resampler_type"] = model_args.mm_resampler_type

            print(f"Overwriting config with {overwrite_config}")
            for k, v in overwrite_config.items():
                setattr(cfg_pretrained, k, v)
            
            model = OryxLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=cfg_pretrained,
                cache_dir=training_args.cache_dir,
                attn_implementation="flash_attention_2",
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args
            )

    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation="flash_attention_2",
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    if "qwen" in model_args.model_name_or_path.lower():
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir, 
            model_max_length=training_args.model_max_length, 
            padding_side="right")
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    elif model_args.version == "llava_llama_3":
        tokenizer.pad_token = "<|reserved_special_token_0|>" # only for llama3
        conversation_lib.default_conversation = conversation_lib.conv_templates["llava_llama_3"]
    else:
        if 'llama-3' in model_args.model_name_or_path.lower():
            tokenizer.pad_token = "<|reserved_special_token_0|>"
        else:
            tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        vision_tower.image_processor.do_resize = training_args.do_resize
        vision_tower.image_processor.do_center_crop = training_args.do_center_crop
        
        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        model.config.tune_mm_vision_resampler = training_args.tune_mm_vision_resampler = model_args.tune_mm_vision_resampler
        if model_args.tune_mm_mlp_adapter or model_args.tune_mm_vision_resampler:
            model.requires_grad_(False)
        if model_args.tune_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True
        if model_args.tune_mm_vision_resampler:
            for p in model.get_model().vision_resampler.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        model.config.freeze_mm_vision_resampler = training_args.freeze_mm_vision_resampler
        if training_args.freeze_mm_vision_resampler:
            for p in model.get_model().vision_resampler.parameters():
                p.requires_grad = False

        model.config.unfreeze_mm_vision_tower = model_args.unfreeze_mm_vision_tower
        if model_args.unfreeze_mm_vision_tower:
            vision_tower.requires_grad_(True)

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        model.config.mm_vision_tower_lr = training_args.mm_vision_tower_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    trainer = OryxTrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
