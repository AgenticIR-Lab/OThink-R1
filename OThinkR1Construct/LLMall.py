# -------------------------------------------------------------------
# OThink-R1
# Copyright (c) 2025 OPPO AgenticIR-Lab and contributors. All rights reserved.
# ------------------------------------------------------------------- 
# Licensed under the MIT license. 
# See LICENSE file in the project root for details.
# -------------------------------------------------------------------

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tools.utils import load_and_combine_datasets,load_tokenizer
from omegaconf import DictConfig, OmegaConf
from core.verifier import AnswerVerifier
from pathlib import Path
from datasets import load_dataset

import pandas  as pd
import pyarrow as pa
import hydra
import os 
import re 

SYSTEM_PROMPT = ''''''  
os.environ["HF_HUB_OFFLINE"] = "1"
think_pattern = re.compile(r'<think>.*?</think>', flags=re.DOTALL)

@hydra.main(version_base=None, config_path="./config", config_name="CONSTRUCT")
def main(config: DictConfig) -> None:
    config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    print(f"config = \n{config}\ntype = {type(config)}")
    

    data_name   = list(config.data.datasets.keys())[0]

    train_data   = load_dataset(f'./OThinkR1Data/{data_name}/{config.model.model_size}/{data_name}-WITH-CORRECT',split="train")
    verify_type = config.data.datasets[data_name]['verify']

    tokenizer    = load_tokenizer(config)

    llm = LLM(
        model                   = config.model.path,
        tensor_parallel_size    = config.model.inference.tensor_parallel_size,
        enable_prefix_caching   = config.model.inference.enable_prefix_caching,
        gpu_memory_utilization  = config.model.inference.gpu_memory_utilization
    )

    sampling_params = SamplingParams(
        temperature         =config.model.inference.temperature,
        top_p               =config.model.inference.top_p,
        max_tokens          =config.model.inference.max_tokens,
        skip_special_tokens =config.model.inference.skip_special_tokens
    )

    prompts = [
        tokenizer.apply_chat_template(
            example["prompt"], 
            tokenize=False, 
            add_generation_prompt=True
        ) 
        for example in train_data
    ]

    requests = llm.generate(prompts, sampling_params, use_tqdm=True)
    # requests: RequestOutput a dict
    # key: 
    # - request_id
    # - prompt
    # - prompt_token_ids: List
    # - outputs: List
    #   - index
    #   - text
    #   - token_ids
    QWen_responses = []
    QWen = 0
    R1 = 0
    for idx, (data, request) in  enumerate(zip(train_data, requests)):
        for completion_output in request.outputs:
            QWen_Answer = think_pattern.sub('', completion_output.text)
            solution    = data['solution']

            try:
                correctness = bool(AnswerVerifier.answer_verify(
                    content     =  QWen_Answer,
                    solution    =  solution,
                    verify_type =  verify_type  
                ))
            except:
                correctness = False
            QWen_responses.append({
                "prompt":           data["prompt"],              
                "question_prompt":  data["question_prompt"],     
                "solution":         data["solution"],
                "correctness":      correctness,
                "QWen-Response":    QWen_Answer
            })
   

    save_dir = Path(f'./OThinkR1Data/{data_name}/{config.model.model_size}/{data_name}-QWen-WITH-CORRECT')
    save_dir.mkdir(exist_ok=True)

    table = pa.Table.from_pandas(pd.DataFrame(QWen_responses))
    pa.parquet.write_table(table, save_dir / "train.parquet")   
        


if __name__ == "__main__":
    main()