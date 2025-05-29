# -------------------------------------------------------------------
# OThink-R1
# Copyright (c) 2025 OPPO AgenticIR-Lab and contributors. All rights reserved.
# ------------------------------------------------------------------- 
# Licensed under the MIT license. 
# See LICENSE file in the project root for details.
# -------------------------------------------------------------------

from datasets import load_dataset
from pathlib import Path

import pandas  as pd
import pyarrow as pa
import os 
import json 
import re

datanames   = [  "CommonsenseQA","OpenBookQA","GSM8K","ASDIV" ]
model_sizes = [ "14B"]

qwen_dict        = dict()
r1___dict        = dict()
part1_index_dict = dict()
valid_index      = dict()
for data in datanames:
    qwen_dict[data]           = dict()
    r1___dict[data]           = dict()
    part1_index_dict[data]    = dict()
    valid_index[data]         = dict()
    for model_size in model_sizes:
        qwen_dict[data].update({
            model_size: f"../OThinkR1Data/{data}/{model_size}/{data}-QWen-WITH-CORRECT"
        })


        r1___dict[data].update({
            model_size: f"../OThinkR1Data/{data}/{model_size}/{data}-WITH-CORRECT"
        })


        part1_index_dict[data].update({
            model_size: f"../OThinkR1Data/{data}/{model_size}/part1.json"
        })

        valid_index[data].update({
            model_size: f"./train-valid/{data}/valid.json"
        })


def read_json(sr_path):
    with open(sr_path, 'r') as f:
        return json.load(f) 

for data in datanames:
    for model_size in model_sizes:
        SFT_data = []
        SFT_dir  = Path(f"../OThinkR1Data/{data}/{model_size}/Version-3-SPLIT")
        SFT_dir.mkdir(exist_ok=True)

        qwen_responses = load_dataset(qwen_dict[data][model_size],split="train" )
        r1___responses = load_dataset(r1___dict[data][model_size],split="train")
        part1__indexes = read_json(part1_index_dict[data][model_size])
        valid__indexes = read_json(valid_index[data][model_size])

        for ind, (r1_res, qwen_res) in enumerate(zip(r1___responses,qwen_responses)):
            if ind in valid__indexes:
                continue
            if r1_res["correctness"]:                                   # We only retain R1 model can develop right answers
                if qwen_res["correctness"] and ind in part1__indexes:   # We prune: 1) Qwen model is right; 2) The reponse is determined as redundant
                    chosen = r1_res['R1-Response']
                    think_content = re.search(r'<think>(.*?)</think>', chosen, re.DOTALL)
                    if think_content:
                        chosen_content = '<think>\\n</think>' + chosen.replace(think_content.group(0), '', 1)
                        SFT_data.append({
                        "prompt":      r1_res["question_prompt"],
                        "completion":  chosen_content
                        })
                    else:
                        SFT_data.append({
                        "prompt":      r1_res["question_prompt"],
                        "completion":  chosen
                        })
                else:
                    SFT_data.append({
                       "prompt":      r1_res["question_prompt"],
                       "completion":  r1_res["R1-Response"]
                    })
        
        


        table = pa.Table.from_pandas(pd.DataFrame(SFT_data))
        pa.parquet.write_table(table, SFT_dir / "train.parquet")   















