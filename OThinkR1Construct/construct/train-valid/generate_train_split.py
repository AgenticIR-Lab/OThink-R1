# -------------------------------------------------------------------
# OThink-R1
# Copyright (c) 2025 OPPO AgenticIR-Lab and contributors. All rights reserved.
# ------------------------------------------------------------------- 
# Licensed under the MIT license. 
# See LICENSE file in the project root for details.
# -------------------------------------------------------------------

from datasets import load_dataset, concatenate_datasets
import random
import json
random.seed(0)


def write_list_to_json(data_list, filename):
    with open(filename, 'w') as json_file:
        json.dump(data_list, json_file, indent=4)


def generate_random_numbers(total_count, percentage):

    count_to_generate = int(total_count * percentage)
    
    random_numbers = random.sample(range(0,total_count), count_to_generate)
    
    return random_numbers

data_save_paths = {
    "ASDIV": "./ASDIV/",
    "OpenBookQA": "./OpenBookQA/",
    "CommonsenseQA": "./CommonsenseQA/",
    "GSM8K": "./GSM8K/"
}


data_paths = {
    "ASDIV": "/home/notebook/data/group/data_hub/huggingface/2020-ASDIV/ASDIV-Data",
    "OpenBookQA": "/home/notebook/data/group/data_hub/huggingface/allenAI/OpenBookQA",
    "CommonsenseQA": "/home/notebook/data/group/data_hub/huggingface/tau/commonsense_qa",
    "GSM8K": "/home/notebook/data/group/data_hub/huggingface/openai/gsm8k-new"
}


for data_path in ["ASDIV", "OpenBookQA", "CommonsenseQA", "GSM8K"]:
    print(data_paths[data_path])
    if data_path in ["GSM8K", "OpenBookQA"]:
        dataset = load_dataset(
            data_paths[data_path],
            "main",
            split=f"train[:100%]"
        )
    else:
        dataset = load_dataset(
            data_paths[data_path],
            split=f"train[:100%]"
        )
    valid_indexes = generate_random_numbers(total_count = len(dataset), percentage = 0.2)
    save_file = data_save_paths[data_path] + "valid.json"
    if data_path in ["OpenBookQA"]:
        write_list_to_json([],save_file)
    else:
        write_list_to_json(sorted(valid_indexes),save_file)
