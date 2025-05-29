# -------------------------------------------------------------------
# OThink-R1
# Copyright (c) 2025 OPPO AgenticIR-Lab and contributors. All rights reserved.
# ------------------------------------------------------------------- 
# Licensed under the MIT license. 
# See LICENSE file in the project root for details.
# -------------------------------------------------------------------

import re
from math_verify import LatexExtractionConfig, ExprExtractionConfig, parse, verify
import random

class AnswerVerifier(object):
    @classmethod
    def answer_string_matching(cls, content, solution, answer_pattern):
        match = re.finditer(answer_pattern, content)
        matches = list(match)
        if matches:
            answer_parsed = str(matches[-1].group(1).strip())
        else:
            answer_parsed = None
        gold_parsed = str(solution)
        if answer_parsed is None:
            return False
        else:
            try:
                if gold_parsed.strip() == answer_parsed.strip():
                    return True,[answer_parsed,gold_parsed]
                else:
                    return False,[answer_parsed,gold_parsed]
            except Exception as e:
                print(f"answer_string_matching failed, because of {e}")
                return False,[None,None]
    
    @classmethod
    def answer_math_verify(cls, content, solution, extraction_mode, 
                           use_answer_pattern=True, answer_pattern=r'.*<answer>\s*([^<]*?)\s*</answer>.*'):
        # gold_parsed
        gold_parsed = parse(solution, 
                       extraction_mode=extraction_mode,
                       extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])

        # answer_parsed
        if use_answer_pattern:
            re_pattern = re.compile(answer_pattern, flags=re.DOTALL | re.IGNORECASE)
            match = re_pattern.fullmatch(content)
            if match:
                response_text = match.group(1)
            else:
                response_text = content
        else:
            response_text = content
        answer_parsed = parse(response_text,
                         extraction_mode=extraction_mode,
                         extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])

        if len(gold_parsed) != 0:
            try:
                verify_result = bool(verify(answer_parsed, gold_parsed))
                return verify_result,[answer_parsed,gold_parsed]
            except Exception as e:
                print(f"answer_math_verify failed, because of {e}")
                return False, [None,None]
        else:
            return True,[answer_parsed,gold_parsed]
    
    @classmethod
    def answer_choice_selecting(cls, content, solution, answer_pattern):
        '''The defined class method'''
        def safe_regex_search(pattern, text, flags=0):
            """
            TODO: The optimal solution for timeout detection is to use the 'regex' library instead of 're' for regular expression matching.
            However, since the 'regex' and 're' libraries handle regex parsing differently, it has not been adopted for now.
            
            Issue: The current implementation using 'timeout_decorator' does not work on Windows platforms.
            Reason: 'timeout_decorator' relies on signal-based timeouts, which are only supported on Unix-based systems and do not work on Windows.
            """
            try:
                return re.search(pattern, text, flags)
            except timeout_decorator.TimeoutError:
                # print(f"Regex match timeout: pattern={pattern}, text={text[:100]}...")
                return None
            except Exception as e:
                # print(f"Regex match error: {str(e)}")
                return None

        def extract_option_labels(text, options='ABCDE'): # options='ABCDEFGHIJ'
            if not isinstance(text, str) or not isinstance(options, str):
                return 'error'
            
            text = text.rstrip()
            last_line = text.split('\n')[-1]
            
            option_str = ''.join([chr(65 + i) for i in range(len(options))]) if options else 'ABCDEFGHIJ'
            
            patterns = [
        
                rf'[Tt]he\s+(?:\w+\s+)?(?:answer|option)(?:\w+\s+)?\s+is?:?\s*(?:[\*\$\\{{(\[\\\\(]*?(?:(?:\\\\boxed|\\\\mathbf|\\\\mathrm|\\\\text){{)?)*\s*([{option_str}])(?:\\\\?\}}?\$?\)?\]?\}}?)*(?:[\s:\.\*)]|$)',
        
                rf'(?i:Answer)[\*\s]*:\s*(?:[\*\$\\{{(\[\\\\(]*?(?:(?:\\\\boxed|\\\\mathbf|\\\\mathrm|\\\\text){{)?)*\s*([{option_str}])(?:\\\\?\}}?\$?\)?\]?\}}?)*(?:[\s:\.\*)]|$)',
        
                rf'^[^\w\r\n]*(?:[\*\$\\{{(\[\\\\(]*?(?:(?:\\\\boxed|\\\\mathbf|\\\\mathrm|\\\\text){{)?)*\s*([{option_str}])(?:\\\\?\}}?\$?\)?\]?\}}?)*(?:[\s:\.\*)]|$)'
            ]

            for pattern in patterns:
                match = safe_regex_search(pattern, last_line, re.IGNORECASE)
                if match:
                    return match.group(1)
            
            for pattern in patterns:
                match = safe_regex_search(pattern, text, re.IGNORECASE)
                if match:
                    return match.group(1)
            
            # Backup mechanism: scanning the entire text from back to front for option characters
            reverse_text = text[::-1]
            reverse_option_pattern = re.compile(rf'([{option_str}])[\.\s\)\}}]*(\b|$)')
            reverse_match = reverse_option_pattern.search(reverse_text)
            if reverse_match:
                candidate = reverse_match.group(1)[::-1]  # Reverse the character recovery sequence
                if candidate in option_str:
                    return candidate
            
            return None

        gold_parsed = str(solution)
        answer_parsed = None
        answer_parsed = extract_option_labels(content)
        if answer_parsed is not None:
            if answer_parsed.strip() in gold_parsed.strip(): # The answer is right
                return True, [answer_parsed,gold_parsed]
            else: # The answer is wrong
                return False, [answer_parsed,gold_parsed]
        else:     # The verifier does not prase the right answer
            return False, [None,gold_parsed]


    @classmethod   
    def answer_verify(
        cls,
        content, 
        solution, 
        verify_type,
        extraction_mode = "first_match",
        answer_pattern = r'<answer>(.*?)</answer>'
    ):
        if verify_type == 'math_verify':
            return cls.answer_math_verify(content, solution, extraction_mode)[0]
        elif verify_type == 'string_matching':
            return cls.answer_string_matching(content, solution, answer_pattern)[0]
        elif verify_type == 'choice_selecting':
            return cls.answer_choice_selecting(content, solution, answer_pattern)[0]
        else:
            return False
    
    @classmethod   
    def answer_verify_and_parse(
        cls,
        content, 
        solution, 
        verify_type,
        extraction_mode = "first_match",
        answer_pattern = r'<answer>(.*?)</answer>'
    ):
        if verify_type == 'math_verify':
            return cls.answer_math_verify(content, solution, extraction_mode)
        elif verify_type == 'string_matching':
            return cls.answer_string_matching(content, solution, answer_pattern)
        elif verify_type == 'choice_selecting':
            return cls.answer_choice_selecting(content, solution, answer_pattern)
        else:
            return False