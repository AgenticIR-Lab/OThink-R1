# OThink-R1 Official Code

This repository contains the code for the submission to the NeurIPS 2025.


# Conda Environment

We provide the environment requirements, you can construct the environment as:
```
conda env create -f environment.yaml
```
This will create a conda environment named `OThink-R1`. To activate the environment, run:
```
conda activate OThink-R1
```

# Code Structure

The code is organized as follows:

```
├── OThinkR1Construct       # OThink-R1: Training
│   ├── config                                      # The Hydra Configh
│   ├── construct                                   # The data construct scripts
│   │   ├── DataConstruct_0_Split.py                # Data construction step 1: Get the data part:
│   │   │                                            (R1: right, Qwen: right). (R1: right, Qwen: wrong)
│   │   ├── DataConstruct_1_LLM_Judge.py            # Data construction step 2: LLM-Judge to determing which is redundant 
│   │   ├── DataConstruct_2_ExtractRedundant.py     # Data construction step 3: Store redundant to json
│   │   ├── DataConstruct_3_Prune.py                # Data construction step 4: Prune redundant reasoning trajectories
│   │   ├── DataConstruct_4_DataFuse.py             # Data construction step 5: Fuse ASDIV, GSM8K to MATH (OpenBookQA, CommonsenseQA to QA)
│   │   ├── JudgeScripts                            # LLM Judge Scripts
│   │   ├── SamplingScripts                         # Sampling Response Scripts
│   │   └── train-valid                             # Valid Index
│   ├── core
│   │   ├── dataset_processor.py                    # The dataprocessor: For construct the corresponding question prompt
│   │   ├── __init__.py                 
│   │   └── verifier.py                             # The correctness verifier
│   ├── LLMall.py                                   # Sampling non-reasoning LLM's responses
│   ├── LRMall.py                                   # Sampling large reasoning model's responses
│   ├── OThinkR1Data                                # OThinkR1-Data
│   ├── OThinkR1LLMJudge                            # LLM-Judge Labels: Essential and Redundant
│   ├── OThinkR1Parts                               # Four parts (R1: right, Qwen: right), ...
│   └── tools                                       # Evaludation and training tools 
├── OThinkR1Training    # OThink-R1: Training
│   ├── config                                      # The Hydra Configh
│   ├── core                                        # The dataprocessor: For construct the corresponding question prompt
│   ├── scripts                                     # The training scripts
│   ├── tools                                       # Evaludation and training tools 
│   ├── eval.py                                     # The evaludation scripts
│   ├── training.py                                 # The training scripts
│   └── validation.py                               # The validation scripts
└── READEME.md                                      # This file

```

# Running Example

The OThink-R1 is divided into two parts: 1) The dataset construction; 2) The model training

## Dataset construction

**Sampling LRM and LLM responses**: We firstly sample R1 and Qwen responses. The sampling scripts are in "./OThinkR1Construct/construct/SamplingScripts". Note that, you should replace the model_id in "./OThinkR1Construct/config/model/" with your own R1 models or Qwen models.

**Split (R1: right, Qwen: wrong), (R1: right, Qwen: right)**: By runing "./OThinkR1Construct/construct/DataConstruct_0_Split.py"

**LLM-Judge**: By runing all scripts in "./OThinkR1Construct/construct/JudgeScripts"

**Extract Redundant Reasoning Index**: By runing "./OThinkR1Construct/construct/DataConstruct_2_ExtractRedundant.py"

**Prune Redundant Reasoning Trajectories**: By runing "./OThinkR1Construct/construct/DataConstruct_3_Prune.py"

**Data Fuse**: By runing "./OThinkR1Construct/construct/DataConstruct_4_DataFuse.py"


## Model Training

By runing scripts in "./OThinkR1Training/scripts"