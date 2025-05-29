# -------------------------------------------------------------------
# OThink-R1
# Copyright (c) 2025 OPPO AgenticIR-Lab and contributors. All rights reserved.
# ------------------------------------------------------------------- 
# Licensed under the MIT license. 
# See LICENSE file in the project root for details.
# -------------------------------------------------------------------

import hydra
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer



def load_and_combine_datasets(config, split_name="train", system_prompt=""):
    dataset_list = []
    for ind, ds in enumerate(config.data.datasets.values()):
        ds_name = str(list(config.data.datasets.keys())[ind])
        processor = hydra.utils.instantiate(ds)
        processor.set_name(ds_name)
        processor.set_system_prompt(system_prompt)
        for split in ds.splits.values():
            if split.name == split_name:
                if ds.subset is not None:
                    dataset = load_dataset(
                        ds.path,
                        ds.subset,
                        split=f"{split.name}{split.get('slice', '')}"
                    )
                else:
                    dataset = load_dataset(
                        ds.path, 
                        split=f"{split.name}{split.get('slice', '')}"
                    )
                processed = dataset.map(processor.make_conversation,load_from_cache_file=False)
                dataset_list.append(processed.remove_columns(split.columns_to_remove))
                print(f"{split_name}: {dataset}")

    # 拼接
    if len(dataset_list) > 1:
        combined_dataset = concatenate_datasets(dataset_list)
        shuffle_dataset = combined_dataset.shuffle(seed=config.utils.seed)
        return shuffle_dataset
    else:
        return dataset_list[0]

    


def load_tokenizer(config):
    if config.model.get('tokenizer', None):
        tokenizer = AutoTokenizer.from_pretrained(config.model.tokenizer)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.model.path)
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_reward_functions(config, tokenizer):
    reward_setting = config.reward
    reward_processor = hydra.utils.instantiate(reward_setting)
    reward_funcs = reward_processor.get_reward_funcs(config, tokenizer)
    return reward_funcs



