# -------------------------------------------------------------------
# OThink-R1
# Copyright (c) 2025 OPPO AgenticIR-Lab and contributors. All rights reserved.
# ------------------------------------------------------------------- 
# Licensed under the MIT license. 
# See LICENSE file in the project root for details.
# -------------------------------------------------------------------

from abc import ABC, abstractmethod
import re

class BaseDatasetProcessor(ABC):
    def __init__(self, **kwargs):
        pass

    def set_name(self, name):
        self.name = name

    def set_system_prompt(self, system_prompt):
        self.system_prompt = system_prompt

    @abstractmethod
    def make_conversation(self, example):
        pass


class CommonsenseQAProcessor(BaseDatasetProcessor):
    def make_conversation(self, example):
        question = example["question"]
        choices = example["choices"]
        choices_text = choices['text']
        answer = example['answerKey']
        question_prompt = f'''
        I am providing a question with multiple choices. The answer is one of the choice listed in the choices. Please answer the correct one. \n
        Question: {question} \n
        CHOICES:
        (A) {choices_text[0]} \n
        (B) {choices_text[1]} \n
        (C) {choices_text[2]} \n
        (D) {choices_text[3]} \n
        (E) {choices_text[4]} \n
        '''
        return {
            "prompt": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": question_prompt},
            ],
            "question_prompt": question_prompt,
            "solution": answer,
            "type": self.name
        }

class OpenBookQAProcessor(BaseDatasetProcessor):
    def make_conversation(self, example):
        question = example["question_stem"]
        choices = example["choices"]
        choices_text = choices['text']
        answer = example['answerKey']
        question_prompt = f'''
        I am providing a question with multiple choices. The answer is one of the choice listed in the choices. Please answer the correct one. \n
        Question: {question} \n
        CHOICES:
        (A) {choices_text[0]} \n
        (B) {choices_text[1]} \n
        (C) {choices_text[2]} \n
        (D) {choices_text[3]} \n
        '''
        return {
            "prompt": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": question_prompt},
            ],
            "question_prompt": question_prompt,
            "solution": answer,
            "type": self.name
        }

class ASDIVProcessor(BaseDatasetProcessor):
    def make_conversation(self, example):
        return {
           "prompt": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": example['text']},
            ],
            "question_prompt": example['text'],
            "solution": example['label'],
            "type": self.name
        }

class GSM8KProcessor(BaseDatasetProcessor):
    def make_conversation(self, example):
        origin_solution = example["answer"]
        match = re.search(r'####\s*(.*)', origin_solution)
        if match:
            result = match.group(1)
        else:
            result = ""
        solution = f"${result}$"
        return {
            "prompt": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": example["question"]},
            ],
            "solution": solution,
            "question_prompt": example['question'],
            "type": self.name
        }

class BigMathRLVerifiedProcessor(BaseDatasetProcessor):
    def make_conversation(self, example):
        origin_solution = example["answer"]
        solution = f"${origin_solution}$"
        return {
            "prompt": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": example["problem"]},
            ],
            "solution": solution,
            "difficulty": example["source"],
            "type": self.name
        }