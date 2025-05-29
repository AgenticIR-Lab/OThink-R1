# -------------------------------------------------------------------
# OThink-R1
# Copyright (c) 2025 OPPO AgenticIR-Lab and contributors. All rights reserved.
# ------------------------------------------------------------------- 
# Licensed under the MIT license. 
# See LICENSE file in the project root for details.
# -------------------------------------------------------------------

import json


datanames   = [ "GSM8K", "ASDIV", "CommonsenseQA", "OpenBookQA" ]
model_sizes = [  "1.5B", "7B", "14B" ]

log_dict = {}

for model_size in model_sizes:
    log_dict[model_size] = dict()
    for data in datanames:
        log_dict[model_size].update({
            data:  f"../OThinkR1LLMJudge/{data}/{model_size}/class.log"
        })

save_dict = {}

for model_size in model_sizes:
    save_dict[model_size] = dict()
    for data in datanames:
        save_dict[model_size].update({
            data:  f"../OThinkR1Data/{data}/{model_size}/part1.json"
        })

map_index = {
    "Essential": False,
    "Essential \\": False,
    "Redundant": True,
    "": True
}
print(log_dict)
for model_size in model_sizes:
    for data in datanames:
        cut_index = []
        LLM_judge_file = log_dict[model_size][data]
        save_json_file = save_dict[model_size][data]

        read_f = open(LLM_judge_file, 'r')
        save_f = open(save_json_file, 'w')

        for line in read_f:
            record = json.loads(line)

            response_type = record["type"]
            if map_index[response_type]:
                cut_index.append(int(record["index"]))
        
        json.dump(cut_index, save_f, indent=4)
        read_f.close()
        save_f.close()
