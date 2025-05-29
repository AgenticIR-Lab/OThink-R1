# -------------------------------------------------------------------
# OThink-R1
# Copyright (c) 2025 OPPO AgenticIR-Lab and contributors. All rights reserved.
# ------------------------------------------------------------------- 
# Licensed under the MIT license. 
# See LICENSE file in the project root for details.
# -------------------------------------------------------------------

from datasets import load_dataset
import os


R1_dataset_path = {
    "1.5B": {
        "ASDIV": "../OThinkR1Data/ASDIV/1.5B/ASDIV-R1-WITH-CORRECT",
        "GSM8K": "../OThinkR1Data/GSM8K/1.5B/GSM8K-R1-WITH-CORRECT",
        "CommonsenseQA": "../OThinkR1Data/CommonsenseQA/1.5B/CommonsenseQA-R1-WITH-CORRECT",
        "OpenBookQA": "../OThinkR1Data/OpenBookQA/1.5B/OpenBookQA-R1-WITH-CORRECT",
    },
    "7B": {
        "ASDIV": "../OThinkR1Data/ASDIV/7B/ASDIV-R1-WITH-CORRECT",
        "GSM8K": "../OThinkR1Data/GSM8K/7B/GSM8K-R1-WITH-CORRECT",
        "CommonsenseQA": "../OThinkR1Data/CommonsenseQA/7B/CommonsenseQA-R1-WITH-CORRECT",
        "OpenBookQA": "../OThinkR1Data/OpenBookQA/7B/OpenBookQA-R1-WITH-CORRECT",
    },
    "14B": {
        "ASDIV": "../OThinkR1Data/ASDIV/14B/ASDIV-R1-WITH-CORRECT",
        "GSM8K": "../OThinkR1Data/GSM8K/14B/GSM8K-R1-WITH-CORRECT",
        "CommonsenseQA": "../OThinkR1Data/CommonsenseQA/14B/CommonsenseQA-R1-WITH-CORRECT",
        "OpenBookQA": "../OThinkR1Data/OpenBookQA/14B/OpenBookQA-R1-WITH-CORRECT",
    }
}


QWen_dataset_path = {
    "1.5B": {
        "ASDIV":            "../OThinkR1Data/ASDIV/1.5B/ASDIV-QWen-WITH-CORRECT",
        "GSM8K":            "../OThinkR1Data/GSM8K/1.5B/GSM8K-QWen-WITH-CORRECT",
        "CommonsenseQA":    "../OThinkR1Data/CommonsenseQA/1.5B/CommonsenseQA-QWen-WITH-CORRECT",
        "OpenBookQA":       "../OThinkR1Data/OpenBookQA/1.5B/OpenBookQA-QWen-WITH-CORRECT",
    },
    "7B": {
        "ASDIV":            "../OThinkR1Data/ASDIV/7B/ASDIV-QWen-WITH-CORRECT",
        "GSM8K":            "../OThinkR1Data/GSM8K/7B/GSM8K-QWen-WITH-CORRECT",
        "CommonsenseQA":    "../OThinkR1Data/CommonsenseQA/7B/CommonsenseQA-QWen-WITH-CORRECT",
        "OpenBookQA":       "../OThinkR1Data/OpenBookQA/7B/OpenBookQA-QWen-WITH-CORRECT",
    },
    "14B": {
        "ASDIV":            "../OThinkR1Data/ASDIV/14B/ASDIV-QWen-WITH-CORRECT",
        "GSM8K":            "../OThinkR1Data/GSM8K/14B/GSM8K-QWen-WITH-CORRECT",
        "CommonsenseQA":    "../OThinkR1Data/CommonsenseQA/14B/CommonsenseQA-QWen-WITH-CORRECT",
        "OpenBookQA":       "../OThinkR1Data/OpenBookQA/14B/OpenBookQA-QWen-WITH-CORRECT",
    },
}

model_sizes = [ "1.5B", "7B", "14B"] 
data_names  = [ "GSM8K", "CommonsenseQA", "OpenBookQA", "ASDIV" ]




def qwen_write_format_context(writer,idx, verify_result, question_answer, generated_text):

    writer.write("------------------------------------------\n")
    writer.write(f"Index: {idx}\n")
    writer.write(f"Verify result: {verify_result}\n")
    writer.write(f"Gold solution: {question_answer}\n")
    writer.write(f"verify_result = {verify_result}\n")
    writer.write(f"Generated text:\n{generated_text}\n") 


def r1_write_format_context(writer,idx, verify_result, question_answer, generated_text):

    writer.write("------------------------------------------\n")
    writer.write(f"Index: {idx}\n")
    writer.write(f"Verify result: {verify_result}\n")
    writer.write(f"Gold solution: {question_answer}\n")
    writer.write(f"verify_result = {verify_result}\n")
    writer.write(f"Generated text:\n{generated_text}\n") 


for model_size in model_sizes:
    for data_name in data_names:
        qwen_path = QWen_dataset_path[model_size][data_name]
        r1___path = R1_dataset_path[model_size][data_name]


        qwen_all = load_dataset(qwen_path)
        r1___all = load_dataset(r1___path)

        path_folders = []
        for part in range(1,5):
            pre_folder = f"../OThinkR1Parts/{data_name}/{model_size}/part{part}/"
            os.makedirs(pre_folder, exist_ok=True)
            path_folders.append({
                "qwen":     pre_folder + "qwen.log",
                "r1":       pre_folder + "r1.log", 
                "question": pre_folder + "question.log"
            })


        conditions      = [ (True, True), (True, False), (False, True),  (False, False) ]
        writer_dict  = {}
        for condition,path_folder in zip(conditions,path_folders):
            writer_dict[condition] = {
                "qwen":     open(path_folder["qwen"], "w"),
                "r1":       open(path_folder["r1"], "w"),
                "question": open(path_folder["question"], "w")
            }


        for ind, (qwen_example, r1___example) in enumerate(zip(qwen_all,r1___all)):
            qwen_correctness = qwen_example["correctness"]
            r1___correctness = r1___example["correctness"]


            qwen_write_format_context(
                writer          = writer_dict[(qwen_correctness,r1___correctness)]["qwen"],
                idx             = ind,
                verify_result   = qwen_example["correctness"],
                question_answer = qwen_example["solution"],
                generated_text  = qwen_example["QWen-Response"]
            )


            r1_write_format_context(
                writer              = writer_dict[(qwen_correctness,r1___correctness)]["r1"],
                idx                 = ind,
                verify_result       = r1___example["correctness"],
                question_answer     = r1___example["solution"],
                generated_text      = r1___example["R1-Response"]
            )
            
            writer_dict[(qwen_correctness,r1___correctness)]["question"].write("------------------------------------------\n")
            writer_dict[(qwen_correctness,r1___correctness)]["question"].write(f"Index: {ind}\n")
            if data_name == "GSM8K":
                writer_dict[(qwen_correctness,r1___correctness)]["question"].write(f"Question: {r1___example["question_prompt"]}\n")
            else:
                writer_dict[(qwen_correctness,r1___correctness)]["question"].write(f"Question-Prompt: {r1___example["question_prompt"]}\n")


