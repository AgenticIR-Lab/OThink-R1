# -------------------------------------------------------------------
# OThink-R1
# Copyright (c) 2025 OPPO AgenticIR-Lab and contributors. All rights reserved.
# ------------------------------------------------------------------- 
# Licensed under the MIT license. 
# See LICENSE file in the project root for details.
# -------------------------------------------------------------------

import json
from datasets import load_dataset
from pathlib import Path

import pandas  as pd
import pyarrow as pa
import os 
import json 
import re

def json_read(file_path = None):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return data


valid_json_dict = {
    "ASDIV":            "./train-valid/ASDIV/valid.json",
    "GSM8K":            "./train-valid/GSM8K/valid.json",
    "OpenBookQA":       "./train-valid/OpenBookQA/valid.json",
    "CommonsenseQA":    "./train-valid/CommonsenseQA/valid.json",
}

data_file_dict = {
    "ASDIV":            "Your/ASDIV/Data/Path",
    "GSM8K":            "Your/GSM8K/Data/Path",
    "OpenBookQA":       "Your/OpenBookQA/Data/Path",
    "CommonsenseQA":    "Your/CommonsenseQA/Data/Path",
}


save_path_dict = {
    "ASDIV":            "Your/ASDIV/Save/Path",
    "GSM8K":            "Your/GSM8K/Save/Path",
    "OpenBookQA":       "Your/OpenBookQA/Save/Path",
    "CommonsenseQA":    "Your/CommonsenseQA/Save/Path" 
}

data_names = ["ASDIV", "GSM8K"]

for data_name in data_names:
    valid_json_file =  valid_json_dict[data_name]
    data_file  =  data_file_dict[data_name]
    valid_index = json_read(valid_json_file)

    if data_name == "GSM8K":
        valid_examples = []

        train_data = load_dataset(data_file, "main", split = "train")

        for ind,example in enumerate(train_data):
            if ind in valid_index:
                valid_examples.append(example)

        valid_MATH_dir  = Path(save_path_dict[data_name])
        valid_MATH_dir.mkdir(exist_ok=True)
        table = pa.Table.from_pandas(pd.DataFrame(valid_examples))
        pa.parquet.write_table(table, valid_MATH_dir / "valid.parquet")   
        print(len(valid_examples))

    else:
        valid_examples = []

        train_data = load_dataset(data_file,split = "train")

        for ind,example in enumerate(train_data):
            if ind in valid_index:
                valid_examples.append(example)

        valid_MATH_dir  = Path(save_path_dict[data_name])
        valid_MATH_dir.mkdir(exist_ok=True)
        table = pa.Table.from_pandas(pd.DataFrame(valid_examples))
        pa.parquet.write_table(table, valid_MATH_dir / "valid.parquet")   
        print(len(valid_examples))







data_names = ["OpenBookQA", "CommonsenseQA"]

for data_name in data_names:
    valid_json_file =  valid_json_dict[data_name]
    data_file  =  data_file_dict[data_name]
    valid_index = json_read(valid_json_file)

    if data_name == "OpenBookQA":
        valid_examples = []

        valid_data = load_dataset(data_file, "main", split = "validation")
        for example in  valid_data:
            valid_examples.append(example)

        valid_MATH_dir  = Path(save_path_dict[data_name])
        valid_MATH_dir.mkdir(exist_ok=True)
        table = pa.Table.from_pandas(pd.DataFrame(valid_examples))
        pa.parquet.write_table(table, valid_MATH_dir / "valid.parquet")   
        print(len(valid_examples))

    else:
        valid_examples = []

        train_data = load_dataset(data_file, split = "train")

        for ind,example in enumerate(train_data):
            if ind in valid_index:
                valid_examples.append(example)
        
        valid_MATH_dir  = Path(save_path_dict[data_name])
        valid_MATH_dir.mkdir(exist_ok=True)
        table = pa.Table.from_pandas(pd.DataFrame(valid_examples))
        pa.parquet.write_table(table, valid_MATH_dir / "valid.parquet")   
        print(len(valid_examples))

