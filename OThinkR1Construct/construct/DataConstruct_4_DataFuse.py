# -------------------------------------------------------------------
# OThink-R1
# Copyright (c) 2025 OPPO AgenticIR-Lab and contributors. All rights reserved.
# ------------------------------------------------------------------- 
# Licensed under the MIT license. 
# See LICENSE file in the project root for details.
# -------------------------------------------------------------------

import pandas  as pd
import pyarrow as pa
import hydra
import os 
import re 
import json
from pathlib import Path
from datasets import load_dataset, Dataset, DatasetDict
import random
random.seed(0)

def read_blocks(log_file):
    with open(log_file, 'r', encoding='utf-8') as file:
        log_content = file.read()

    text = log_content

    blocks = text.split('------------------------------------------')

    return blocks



def read_json(sr_path):
    with open(sr_path, 'r') as f:
        return json.load(f) 


dataset_id_dict = {
    "1.5B": {
        "GSM8K":            "../OThinkR1Data/GSM8K/1.5B/Version-3-SPLIT",
        "ASDIV":            "../OThinkR1Data/ASDIV/1.5B/Version-3-SPLIT",
        "OpenBookQA":       "../OThinkR1Data/OpenBookQA/1.5B/Version-3-SPLIT",
        "CommonsenseQA":    "../OThinkR1Data/CommonsenseQA/1.5B/Version-3-SPLIT"
    },
    "7B": {
        "GSM8K":            "../OThinkR1Data/GSM8K/7B/Version-3-SPLIT",
        "ASDIV":            "../OThinkR1Data/ASDIV/7B/Version-3-SPLIT",
        "OpenBookQA":       "../OThinkR1Data/OpenBookQA/7B/Version-3-SPLIT",
        "CommonsenseQA":    "../OThinkR1Data/CommonsenseQA/7B/Version-3-SPLIT"
    },
    "14B": {
        "GSM8K":            "../OThinkR1Data/GSM8K/14B/Version-3-SPLIT",
        "ASDIV":            "../OThinkR1Data/ASDIV/14B/Version-3-SPLIT",
        "OpenBookQA":       "../OThinkR1Data/OpenBookQA/14B/Version-3-SPLIT",
        "CommonsenseQA":    "../OThinkR1Data/CommonsenseQA/14B/Version-3-SPLIT"
    }
}


fuse_seperate = {
    "1.5B": {
        "QA":               "../OThinkR1Data/FUSE/1.5B/QA-SPLIT",
        "MATH-Shuffle":     "../OThinkR1Data/FUSE/1.5B/MATH-SPLIT-Shuffle",
    },
    "7B": {
        "QA":               "../OThinkR1Data/FUSE/7B/QA-SPLIT",
        "MATH-Shuffle":     "../OThinkR1Data/FUSE/7B/MATH-SPLIT-Shuffle",
    },
    "14B": {
        "QA":               "../OThinkR1Data/FUSE/14B/QA-SPLIT",
        "MATH-Shuffle":     "../OThinkR1Data/FUSE/14B/MATH-SPLIT-Shuffle",
    }
    
}



data_names  = ["ASDIV","GSM8K"]
model_sizes = [ "1.5B", "7B", "14B"]
mode = "MATH-Shuffle"
# 各自 FUSE
for model_size in model_sizes:
    save_dir = Path(fuse_seperate[model_size][mode])
    os.makedirs(save_dir,exist_ok=True)
    mode_fuse = []
    for data_name in data_names:
        single_data = load_dataset(dataset_id_dict[model_size][data_name], split = "train")
        for example in single_data:
            mode_fuse.append(example)
    random.shuffle(mode_fuse)
    print(len(mode_fuse))
    table = pa.Table.from_pandas(pd.DataFrame(mode_fuse))
    pa.parquet.write_table(table, save_dir / "train.parquet")  



data_names  = [ "CommonsenseQA", "OpenBookQA"]
model_sizes = [ "1.5B", "7B", "14B"]
mode = "QA"
for model_size in model_sizes:
    save_dir = Path(fuse_seperate[model_size][mode])
    os.makedirs(save_dir,exist_ok=True)
    mode_fuse = []
    for data_name in data_names:
        single_data = load_dataset(dataset_id_dict[model_size][data_name], split = "train")
        print(single_data)
        for example in single_data:
            mode_fuse.append(example)
    print(len(mode_fuse))
    table = pa.Table.from_pandas(pd.DataFrame(mode_fuse))
    pa.parquet.write_table(table, save_dir / "train.parquet") 

