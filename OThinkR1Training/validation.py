# -------------------------------------------------------------------
# OThink-R1
# Copyright (c) 2025 OPPO AgenticIR-Lab and contributors. All rights reserved.
# ------------------------------------------------------------------- 
# Licensed under the MIT license. 
# See LICENSE file in the project root for details.
# -------------------------------------------------------------------


# pip install hydra-core omegaconf
import os
import hydra
from omegaconf import DictConfig, OmegaConf


from vllm import LLM, SamplingParams
from tools.eval_utils import write_responses,write_valid_responses
from tools.utils import load_and_combine_datasets
from transformers import AutoTokenizer

import logging

SYSTEM_PROMPT = ''''''

os.environ["HF_HUB_OFFLINE"] = "1"

@hydra.main(version_base=None, config_path="./config", config_name="eval")
def eval(config: DictConfig) -> None:
    config = OmegaConf.create(OmegaConf.to_container(config, resolve=True)) 
    print(f"config = \n{config}\ntype = {type(config)}")


    eval_split = "validation"
    test_dataset = load_and_combine_datasets(config, split_name=eval_split, system_prompt=SYSTEM_PROMPT)

    
    tokenizer = AutoTokenizer.from_pretrained(config.model.path)

    inference_model = LLM(
        model                  = config.model.path,
        tensor_parallel_size   = config.model.inference.tensor_parallel_size,
        enable_prefix_caching  = config.model.inference.enable_prefix_caching,
        gpu_memory_utilization = config.model.inference.gpu_memory_utilization,
        repetition_penalty     = config.model.inference.repetition_penalty
    )

    sampling_params = SamplingParams(
        temperature             = config.model.inference.temperature,
        top_p                   = config.model.inference.top_p,
        max_tokens              = config.model.inference.max_tokens,
        skip_special_tokens     = config.model.inference.skip_special_tokens
    )


    prompts = [
        tokenizer.apply_chat_template(
            example["prompt"], 
            tokenize=False, 
            add_generation_prompt=True
        ) 
        for example in test_dataset
    ]


    outputs = inference_model.generate(prompts, sampling_params, use_tqdm=True)

    write_valid_responses(
        config = config, 
        outputs = outputs,
        test_dataset = test_dataset
    )




if __name__ == "__main__":
    eval()