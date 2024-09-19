import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from oryx.model import *
from oryx.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", overwrite_config=None):
    kwargs = {"device_map": device_map}

    if load_8bit:
        kwargs["load_in_8bit"] = True
    elif load_4bit:
        kwargs["load_in_4bit"] = True
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    else:
        kwargs["torch_dtype"] = torch.bfloat16

    if "oryx" in model_name.lower():
        # Load Oryx model
        if "7b" in model_name.lower():
            from oryx.model.language_model.oryx_qwen import OryxQwenConfig
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            if overwrite_config is not None:
                cfg_pretrained = OryxQwenConfig.from_pretrained(model_path)
                print(f"Overwriting config with {overwrite_config}")
                for k, v in overwrite_config.items():
                    setattr(cfg_pretrained, k, v)
                model = OryxQwenForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            else:
                model = OryxQwenForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            cfg_pretrained = AutoConfig.from_pretrained(model_path)
            if overwrite_config is not None:
                print(f"Overwriting config with {overwrite_config}")
                for k, v in overwrite_config.items():
                    setattr(cfg_pretrained, k, v)
            model = OryxLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel

            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="auto")
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print("Convert to FP16...")
            model.to(torch.bfloat16)
        else:
            use_fast = False
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor = None

    assert "oryx" in model_name.lower(), "Only Oryx models are supported for video chatbot."
    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    print("Loading vision tower...")
    if not vision_tower.is_loaded:
        vision_tower.load_model(device_map=device_map)
    if device_map != "auto":
        vision_tower.to(device="cuda", dtype=torch.bfloat16)
    else:
        vision_tower.to(device="cuda:0", dtype=torch.bfloat16)
    image_processor = vision_tower.image_processor
    print("Loading vision tower succeeded.")
    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
