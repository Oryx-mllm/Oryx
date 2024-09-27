import re
import argparse
import torch
import math
import numpy as np
from typing import Dict, Optional, Sequence, List
import transformers
from transformers import AutoConfig
from PIL import Image

from oryx.conversation import conv_templates, SeparatorStyle
from oryx.model.builder import load_pretrained_model
from oryx.utils import disable_torch_init
from oryx.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_anyres_video_genli, process_anyres_highres_image_genli
from oryx.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX

from decord import VideoReader, cpu


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}

    im_start, im_end = tokenizer.additional_special_tokens_ids
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens
    _user = tokenizer("user").input_ids + nl_tokens
    _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []

    source = sources
    if roles[source[0]["from"]] != roles["human"]:
        source = source[1:]

    input_id, target = [], []
    system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
    input_id += system
    target += [im_start] + [IGNORE_INDEX] * (len(system) - 3) + [im_end] + nl_tokens
    assert len(input_id) == len(target)
    for j, sentence in enumerate(source):
        role = roles[sentence["from"]]
        if has_image and sentence["value"] is not None and "<image>" in sentence["value"]:
            num_image = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
            texts = sentence["value"].split('<image>')
            _input_id = tokenizer(role).input_ids + nl_tokens 
            for i,text in enumerate(texts):
                _input_id += tokenizer(text).input_ids 
                if i<len(texts)-1:
                    _input_id += [IMAGE_TOKEN_INDEX] + nl_tokens
            _input_id += [im_end] + nl_tokens
            assert sum([i==IMAGE_TOKEN_INDEX for i in _input_id])==num_image
        else:
            if sentence["value"] is None:
                _input_id = tokenizer(role).input_ids + nl_tokens
            else:
                _input_id = tokenizer(role).input_ids + nl_tokens + tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
        input_id += _input_id
        if role == "<|im_start|>user":
            _target = [im_start] + [IGNORE_INDEX] * (len(_input_id) - 3) + [im_end] + nl_tokens
        elif role == "<|im_start|>assistant":
            _target = [im_start] + [IGNORE_INDEX] * len(tokenizer(role).input_ids) + _input_id[len(tokenizer(role).input_ids) + 1 : -2] + [im_end] + nl_tokens
        else:
            raise NotImplementedError
        target += _target

    input_ids.append(input_id)
    targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    return input_ids

def eval_model(args):
    
    # Model
    disable_torch_init()
    model_path = args.model_path
    model_name = get_model_name_from_path(model_path)
    if args.overwrite == True:
        overwrite_config = {}
        overwrite_config["mm_resampler_type"] = "dynamic_compressor"
        overwrite_config["patchify_video_feature"] = False
        overwrite_config["attn_implementation"] = "sdpa" if torch.__version__ >= "2.1.2" else "eager"

        cfg_pretrained = AutoConfig.from_pretrained(model_path)
    if '7b' in model_path:
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, device_map="cuda:0", overwrite_config=overwrite_config)
    elif '34b' in model_path:
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, device_map="auto", overwrite_config=overwrite_config)
    model.to('cuda').eval()

    image_file = "image.png"
    visual = Image.open(image_file)
    image_sizes = [visual.size]

    args.conv_mode = "qwen_1_5"
    
    question = "Describe this image."
    question = "<image>\n" + question

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    if '7b' in model_path:
        input_ids = preprocess_qwen([{'from': 'human','value': question},{'from': 'gpt','value': None}], tokenizer, has_image=True).cuda()
    elif '34b' in model_path:
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to('cuda:0')

    image_tensor, image_highres_tensor = [], []
    image_processor.do_resize = False
    image_processor.do_center_crop = False

    image_tensor_, image_highres_tensor_ = process_anyres_highres_image_genli(visual, image_processor)

    image_tensor.append(image_tensor_)
    image_highres_tensor.append(image_highres_tensor_)

    image_tensor = torch.stack(image_tensor, dim=0)
    image_highres_tensor = torch.stack(image_highres_tensor, dim=0)

    image_tensor = image_tensor.to(dtype=torch.bfloat16, device=self.device)
    image_highres_tensor = image_highres_tensor.to(dtype=torch.bfloat16, device=self.device)
    
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = self.model.generate(
            input_ids,
            modalities=['image'],
            images=image_tensor,
            images_highres=image_highres_tensor,
            image_sizes=image_sizes,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=128,
            use_cache=True,
        )

    
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    print(outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--frames-upbound", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--overwrite", type=bool, default=True)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)