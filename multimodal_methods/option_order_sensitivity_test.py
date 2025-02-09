import numpy as np
from tqdm import tqdm
from mm_detect.utils.logger import get_child_logger
from mm_detect.utils.dataset_utils import get_answers_list, get_answer_index

from mm_detect.mllms.qwen_vl import QwenVL
from mm_detect.mllms.llava import LLaVA
from mm_detect.mllms.vila import VILA
from mm_detect.mllms.fuyu import Fuyu
from mm_detect.mllms.internvl2 import InterVL2
from mm_detect.mllms.yi_vl import YiVL
from mm_detect.mllms.phi3_vision import Phi3
from mm_detect.mllms.idefics2 import Idefics2
from mm_detect.mllms.gpt import GPT
from mm_detect.mllms.deepseek_vl2 import DeepseekVL2

import os
import json
import random

logger = get_child_logger("option_order_sensitivity_test")

def shuffle_choices(choices, answer_index):
    answer = choices[answer_index]
    shuffled_choices = choices[:]
    while True:
        random.shuffle(shuffled_choices)
        new_answer_index = shuffled_choices.index(answer)
        if new_answer_index != answer_index:
            break
    return answer, shuffled_choices, new_answer_index

def build_prompt(example, eval_data_name):
    choices = get_answers_list(example, eval_data_name)
    if not choices:
        return "failed", "failed", "failed", "failed", "failed"

    text = example["text"]

    if eval_data_name == "MMMU/MMMU_Pro":
        text = text.replace("<image 1>", "image")
    
    answer_index = get_answer_index(example, eval_data_name)
    answer, shuffled_choices, new_answer_index = shuffle_choices(choices, answer_index)

    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    option = alphabet[answer_index]
    new_option = alphabet[new_answer_index]

    prompt = "Please answer the following multichoice question."
    prompt += f"\n\nQuestion: {text}"
    prompt += "\nOptions:"
    for i in range(len(choices)):
        letter = alphabet[i]
        choice = choices[i]
        prompt += f"\n{letter}: [{choice}]"
    prompt += "\n\nReply with answer only."

    new_prompt = "Please answer the following multichoice question."
    new_prompt += f"\n\nQuestion: {text}"
    new_prompt += "\nOptions:"
    for i in range(len(shuffled_choices)):
        letter = alphabet[i]
        choice = shuffled_choices[i]
        new_prompt += f"\n{letter}: [{choice}]"
    new_prompt += "\n\nReply with the answer only."

    return prompt, new_prompt, option, new_option, answer

def inference(data_points, eval_data_name, llm):
    responses = []
    new_responses = []
    options = []
    new_options = []
    answers = []

    results_file = "/home/leo/workspace/log/deepseek_vl2/results.json"
    
    if os.path.exists(results_file) and os.path.getsize(results_file) > 0:
        with open(results_file, "r") as f:
            results = json.load(f)
        last_id = results[-1]["id"] if results else 0
        start_index = len(results)
        print(f"Resuming from index {start_index}, last id {last_id}")
        id_counter = last_id + 1
    else:
        results = []
        start_index = 0
        id_counter = 1

    for i in tqdm(range(start_index, len(data_points))):
        example = data_points[i]
        prompt, new_prompt, option, new_option, answer = build_prompt(example, eval_data_name)
        if prompt == "failed" or new_prompt == "failed":
            print(f"Skipping data point index {i} due to build_prompt failure.")
            continue

        try:
            if isinstance(llm, QwenVL):
                response, _ = llm.eval_model(data_point=example, prompt=prompt, id=id_counter)
                new_response, _ = llm.eval_model(data_point=example, prompt=new_prompt, id=id_counter)
            else:
                response, _ = llm.eval_model(data_point=example, prompt=prompt)
                new_response, _ = llm.eval_model(data_point=example, prompt=new_prompt)
        except Exception as e:
            print(f"LLM evaluation error at index {i}: {e}")
            continue

        entry = {
            "id": id_counter,
            "question": example["text"],
            "original_prompt": prompt,
            "response": response,
            "option": option,
            "shuffled_prompt": new_prompt,
            "shuffled_response": new_response,
            "new_option": new_option,
            "answer": answer
        }
        results.append(entry)
        id_counter += 1

        if (i - start_index + 1) % 10 == 0:
            with open(results_file, "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"Saved intermediate results up to data point index {i}")

    for entry in results:
        responses.append(entry["response"])
        new_responses.append(entry["shuffled_response"])
        options.append(entry["option"])
        new_options.append(entry["new_option"])
        answers.append(entry["answer"])

    return responses, new_responses, options, new_options, answers

def main_option_order_sensitivity_test(
    eval_data,
    eval_data_name,
    n_eval_data_points,
    # model parameters
    model_name: str = None,
    max_output_tokens: int = 128,
    temperature: float = 0.0,
):
    if "llava" in model_name.lower():
        llm = LLaVA(
            model_name=model_name,
            max_output_tokens=10,
            temperature=0.0
        )
        responses, new_responses, options, new_options, answers = inference(
            eval_data,
            eval_data_name,
            llm
        )
    elif "vila" in model_name.lower():
        llm = VILA(
            model_name=model_name,
            max_output_tokens=10,
            temperature=0.0
        )
        responses, new_responses, options, new_options, answers = inference(
            eval_data,
            eval_data_name,
            llm
        )
    elif "internvl2" in model_name.lower(): 
        llm = InterVL2(
            model_name=model_name,
            max_output_tokens=10,
            temperature=0.0
        )
        responses, new_responses, options, new_options, answers = inference(
            eval_data,
            eval_data_name,
            llm
        )
    elif "idefics" in model_name.lower(): 
        llm = Idefics2(
            model_name=model_name,
            max_output_tokens=10,
            temperature=0.0
        )
        responses, new_responses, options, new_options, answers = inference(
            eval_data,
            eval_data_name,
            llm
        )
    elif "qwen" in model_name.lower(): 
        llm = QwenVL(
            model_name=model_name,
            max_output_tokens=10,
            temperature=0.0
        )
        responses, new_responses, options, new_options, answers = inference(
            eval_data,
            eval_data_name,
            llm
        )
    elif "fuyu" in model_name.lower(): 
        llm = Fuyu(
            model_name=model_name,
            max_output_tokens=10,
            temperature=0.0
        )
        responses, new_responses, options, new_options, answers = inference(
            eval_data,
            eval_data_name,
            llm
        )
    elif "yi-vl" in model_name.lower(): 
        llm = YiVL(
            model_name=model_name,
            max_output_tokens=10,
            temperature=0.0
        )
        responses, new_responses, options, new_options, answers = inference(
            eval_data,
            eval_data_name,
            llm
        )
    elif "phi-3-vision" in model_name.lower(): 
        llm = Phi3(
            model_name=model_name,
            max_output_tokens=10,
            temperature=0.0
        )
        responses, new_responses, options, new_options, answers = inference(
            eval_data,
            eval_data_name,
            llm
        )
    elif "gpt" in model_name.lower() or "gemini" in model_name.lower() or "claude" in model_name.lower(): 
        llm = GPT(
            model_name=model_name,
            max_output_tokens=10,
            temperature=0.0
        )
        responses, new_responses, options, new_options, answers = inference(
            eval_data,
            eval_data_name,
            llm
        )
    elif "deepseek" in model_name.lower(): 
        llm = DeepseekVL2(
            model_name=model_name,
            max_output_tokens=10,
            temperature=0.0
        )
        responses, new_responses, options, new_options, answers = inference(
            eval_data,
            eval_data_name,
            llm
        )

    responses = [x.lower() for x in responses]
    new_responses = [x.lower() for x in new_responses]
    options = [x.lower() for x in options]
    new_options = [x.lower() for x in new_options]
    answers = [x.lower() for x in answers]

    correct_count = 0
    for i in range(len(responses)):
        if responses[i] == options[i] or answers[i] in responses[i]:
            correct_count += 1
    correct_rate = correct_count / len(responses)

    new_correct_count = 0
    for i in range(len(new_responses)):
        if new_responses[i] == new_options[i] or answers[i] in new_responses[i]:
            new_correct_count += 1
    new_correct_rate = new_correct_count / len(new_responses)

    delta = new_correct_rate - correct_rate
    normalized_delta = delta / correct_rate

    drops = 0
    for i in range(len(new_responses)):
        if responses[i] == options[i] or answers[i] in responses[i]:
            if new_responses[i] != new_options[i] and answers[i] not in new_responses[i]:
                drops += 1
    instance_leakage = drops / len(new_responses)

    logger.info(f"Correct Rate: {correct_rate:.2%}")
    logger.info(f"Correct Rate after shuffling: {new_correct_rate:.2%}")
    logger.info(f"Difference of the Correct Rate: {delta:.2%}")
    logger.info(f"Normalized Difference of the Correct Rate: {normalized_delta:.2%}")
    logger.info(f"Instance Leakage: {instance_leakage:.2%}")
