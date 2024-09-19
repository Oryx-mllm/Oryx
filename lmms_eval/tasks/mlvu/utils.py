from collections import defaultdict
import os
import datetime
import json
from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from pathlib import Path
import yaml
import sys
from typing import List, Dict, Optional, Union
import re
# import cv2
import numpy as np
from loguru import logger as eval_logger

TASK_TYPES = ["TR", "AR", "VS", "NQA", "ER", "PQA", "SSC", "AO", "AC"]


hf_home = os.getenv("HF_HOME", "./~/.cache/huggingface")
base_cache_dir = os.path.expanduser(hf_home)

with open(Path(__file__).parent / "mlvu.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)
cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]


def mlvu_doc_to_visual(doc):
    cache_dir = os.path.join(base_cache_dir, cache_name)
    video_path = doc["video_name"]
    video_path = os.path.join(cache_dir, video_path)
    if os.path.exists(video_path):
        video_path = video_path
    else:
        sys.exit(f"video path:{video_path} does not exist, please check")
    return [video_path]


def mlvu_doc_to_text(doc, model_specific_prompt_kwargs=None):
    question = doc["question"]
    true_question = question.split("Question: ")[-1].split("Options:\n")[0]
    options = question.split("Question: ")[-1].split("Options:\n")[1].split("\n")

    if len(options) == 2:
        q_prompt = "Answer the question with A, or B."
    elif len(options) == 3:
        q_prompt = "Answer the question with A, B, or C."
    elif len(options) == 4:
        q_prompt = "Answer the question with A, B, C, or D."
    elif len(options) == 5:
        q_prompt = "Answer the question with A, B, C, D, or E."
    else:
        raise ValueError("Too many options")
    
    for idx in range(len(options)):
        if '(A)' in options[idx]:
            options[idx] = options[idx].replace('(A)', 'A.')
        elif '(B)' in options[idx]:
            options[idx] = options[idx].replace('(B)', 'B.')
        elif '(C)' in options[idx]:
            options[idx] = options[idx].replace('(C)', 'C.')
        elif '(D)' in options[idx]:
            options[idx] = options[idx].replace('(D)', 'D.')
        elif '(E)' in options[idx]:
            options[idx] = options[idx].replace('(E)', 'E.')
        options[idx] = options[idx][:3] + options[idx][3:].capitalize()
        if not options[idx].endswith('.'):
            options[idx] += '.'
    full_prompt = true_question + str(options) + "\n" + q_prompt
    return full_prompt


def extract_characters_regex(s):
    s = s.strip()
    if ")" in s:
        index = s.index(")")
        pred = s[index - 1 : index]
        return pred
    else:
        return s


def mlvu_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case videomme score), value: metric value
    """
    pred = results[0]
    # print("****************",pred)
    pred_ans = extract_characters_regex(pred)

    task_type = doc["task_type"]
    data_dict = {"question_id": doc["question"], "task_type": task_type, "pred_answer": pred_ans, "answer": doc["answer"]}

    return {f"mlvu_percetion_score": data_dict}


def mlvu_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    category2score = {}
    for task_type in TASK_TYPES:
        category2score[task_type] = {"correct": 0, "answered": 0}

    for result in results:
        task_type = result["task_type"]
        category2score[task_type]["answered"] += 1
        category2score[task_type]["correct"] += result["pred_answer"] == result["answer"]

    for task_cate in TASK_TYPES:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if task_cate in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        eval_logger.info(f"Evaluation on Task Categories: {task_cate}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")

    total_correct = 0
    total_answered = 0
    for k, v in category2score.items():
        total_correct += v["correct"]
        total_answered += v["answered"]
    eval_logger.info(f"Overall Performance: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")

    return 100 * total_correct / total_answered if total_answered > 0 else 0


