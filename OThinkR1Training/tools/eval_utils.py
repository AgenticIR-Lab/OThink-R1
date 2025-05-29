# -------------------------------------------------------------------
# OThink-R1
# Copyright (c) 2025 OPPO AgenticIR-Lab and contributors. All rights reserved.
# ------------------------------------------------------------------- 
# Licensed under the MIT license. 
# See LICENSE file in the project root for details.
# -------------------------------------------------------------------


from core.verifier import AnswerVerifier
import logging
import os

def create_eval_logger(level, handlers, format_string):
    logger = logging.getLogger("evaluation_logger")
    logger.setLevel(level)

    formatter = logging.Formatter(format_string)
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

def write_responses(config, outputs, test_dataset):
    
    data_name = list(config.data.datasets.keys())[0]
    data_verify = config.data.datasets[data_name]['verify']

    model_path = config.model.path
    parts = model_path.split('/')
    model_string = parts[-2] + '_' + parts[-1].replace('-', '')  

    log_filename = f"log/{data_name}/{config.model.model_size}/{config.model.mode}/{model_string}-parallel-{config.model.inference.tensor_parallel_size}-tmp-{config.model.inference.temperature}-topp-{config.model.inference.top_p}.log"
    log_dir = os.path.dirname(log_filename)
    os.makedirs(log_dir, exist_ok=True)   
 
    logger = create_eval_logger(
        level           = logging.INFO,
        format_string   = '%(message)s',
        handlers        = [
            logging.FileHandler(log_filename, mode='w'),    
            logging.StreamHandler()                         
        ]
    )




    num_generated_tokens_list = []
    verify_result_list = []
 
    logger.info(f"\n======== Evaluation Start ========")
    logger.info(f"Model:       {config.model.name}")


    failed_parsed = 0

    for idx, llm_response in enumerate(outputs):

        completion_output       = llm_response.outputs[0] 

        
        generated_text          = completion_output.text
        num_generated_tokens    = len(completion_output.token_ids) 
        solution                = test_dataset[idx]["solution"]

        verify_result               = False                        
        verify_result, answers = AnswerVerifier.answer_verify_and_parse(
            content         = generated_text,
            solution        = solution,
            verify_type     = data_verify,
        )
        
        model_answer    = answers[0]
        question_answer = answers[1]

        if question_answer is None:
            failed_parsed = failed_parsed + 1

        # # level = test_dataset[idx]["level"]
        num_generated_tokens_list.append(num_generated_tokens)
        verify_result_list.append(verify_result)
 
        logger.info("------------------------------------------")
        logger.info(f"Index: {idx}")
        logger.info(f"Verify result: {verify_result}")
        logger.info(f"Gold solution: {question_answer}")
        logger.info(f"Model Answer: {model_answer}")
        logger.info(f"verify_result = {verify_result}")
        logger.info(f"Generated tokens: {num_generated_tokens}\n")
        logger.info(f"Generated text:\n{generated_text}") 

    print("")
    print(f"failed_parsed = {failed_parsed}")

    logger.info("\n============= Summary =============")
    logger.info(f"Total cases: {len(outputs)}")
    logger.info(f"Average tokens: {sum(num_generated_tokens_list)/len(num_generated_tokens_list):.3f}")
    logger.info(f"Correct rate: {sum(verify_result_list)/len(verify_result_list):.3f}")
    logger.info(f"Log saved to: {log_filename}")



def write_valid_responses(config, outputs, test_dataset):
    
    data_name = list(config.data.datasets.keys())[0]
    data_verify = config.data.datasets[data_name]['verify']

    model_path = config.model.path
    parts = model_path.split('/')
    model_string = parts[-2] + '_' + parts[-1].replace('-', '')

    log_filename = f"valid/{data_name}/{config.model.model_size}/{config.model.mode}/{model_string}-parallel-{config.model.inference.tensor_parallel_size}-tmp-{config.model.inference.temperature}-topp-{config.model.inference.top_p}.log"
    log_dir = os.path.dirname(log_filename)
    os.makedirs(log_dir, exist_ok=True) 

    logger = create_eval_logger(
        level           = logging.INFO,
        format_string   = '%(message)s',
        handlers        = [
            logging.FileHandler(log_filename, mode='w'),  
            logging.StreamHandler()                      
        ]
    )




    num_generated_tokens_list = []
    verify_result_list = []
 
    logger.info(f"\n======== Evaluation Start ========")
    logger.info(f"Model:       {config.model.name}")


    failed_parsed = 0

    for idx, llm_response in enumerate(outputs):

        completion_output       = llm_response.outputs[0] 

        
        generated_text          = completion_output.text
        num_generated_tokens    = len(completion_output.token_ids) 
        solution                = test_dataset[idx]["solution"]

        verify_result               = False                        
        verify_result, answers = AnswerVerifier.answer_verify_and_parse(
            content         = generated_text,
            solution        = solution,
            verify_type     = data_verify,
        )
        
        model_answer    = answers[0]
        question_answer = answers[1]

        if question_answer is None:
            failed_parsed = failed_parsed + 1

        # # level = test_dataset[idx]["level"]
        num_generated_tokens_list.append(num_generated_tokens)
        verify_result_list.append(verify_result)
 
        logger.info("------------------------------------------")
        logger.info(f"Index: {idx}")
        logger.info(f"Verify result: {verify_result}")
        logger.info(f"Gold solution: {question_answer}")
        logger.info(f"Model Answer: {model_answer}")
        logger.info(f"verify_result = {verify_result}")
        logger.info(f"Generated tokens: {num_generated_tokens}\n")
        logger.info(f"Generated text:\n{generated_text}")  

    print("")
    print(f"failed_parsed = {failed_parsed}")

    logger.info("\n============= Summary =============")
    logger.info(f"Total cases: {len(outputs)}")
    logger.info(f"Average tokens: {sum(num_generated_tokens_list)/len(num_generated_tokens_list):.3f}")
    logger.info(f"Correct rate: {sum(verify_result_list)/len(verify_result_list):.3f}")
    logger.info(f"Log saved to: {log_filename}")





def write_repeat_responses(config, outputs, test_dataset):
    
    data_name = list(config.data.datasets.keys())[0]
    data_verify = config.data.datasets[data_name]['verify']

    model_path = config.model.path
    parts = model_path.split('/')
    model_string = parts[-2] + '_' + parts[-1].replace('-', '') 

    log_filename = f"repeatLog/{data_name}/{config.model.model_size}/{config.model.mode}/{model_string}-parallel-{config.model.inference.tensor_parallel_size}-tmp-{config.model.inference.temperature}-topp-{config.model.inference.top_p}-reapeat-{config.model.inference.repetition_penalty}.log"
    log_dir = os.path.dirname(log_filename)
    os.makedirs(log_dir, exist_ok=True)

    logger = create_eval_logger(
        level           = logging.INFO,
        format_string   = '%(message)s',
        handlers        = [
            logging.FileHandler(log_filename, mode='w'), 
            logging.StreamHandler()                       
        ]
    )




    num_generated_tokens_list = []
    verify_result_list = []
 
    logger.info(f"\n======== Evaluation Start ========")
    logger.info(f"Model:       {config.model.name}")
    # logging.info(f"Dataset:     {config.data.}")


    failed_parsed = 0

    for idx, llm_response in enumerate(outputs):

        completion_output       = llm_response.outputs[0]

        
        generated_text          = completion_output.text
        num_generated_tokens    = len(completion_output.token_ids)
        solution                = test_dataset[idx]["solution"]

        verify_result               = False                        
        verify_result, answers = AnswerVerifier.answer_verify_and_parse(
            content         = generated_text,
            solution        = solution,
            verify_type     = data_verify,
        )
        
        model_answer    = answers[0]
        question_answer = answers[1]

        if question_answer is None:
            failed_parsed = failed_parsed + 1

        # # level = test_dataset[idx]["level"]
        num_generated_tokens_list.append(num_generated_tokens)
        verify_result_list.append(verify_result)
 
        logger.info("------------------------------------------")
        logger.info(f"Index: {idx}")
        logger.info(f"Verify result: {verify_result}")
        logger.info(f"Gold solution: {question_answer}")
        logger.info(f"Model Answer: {model_answer}")
        logger.info(f"verify_result = {verify_result}")
        logger.info(f"Generated tokens: {num_generated_tokens}\n")
        logger.info(f"Generated text:\n{generated_text}") 

    print("")
    print(f"failed_parsed = {failed_parsed}")

    logger.info("\n============= Summary =============")
    logger.info(f"Total cases: {len(outputs)}")
    logger.info(f"Average tokens: {sum(num_generated_tokens_list)/len(num_generated_tokens_list):.3f}")
    logger.info(f"Correct rate: {sum(verify_result_list)/len(verify_result_list):.3f}")
    logger.info(f"Log saved to: {log_filename}")