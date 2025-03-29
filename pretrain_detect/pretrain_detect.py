import numpy as np
from tqdm import tqdm
from mm_detect.utils.logger import get_child_logger
from mm_detect.utils.dataset_utils import get_answers_list, get_answer_index

from nltk.tokenize import word_tokenize
from mm_detect.mllms.pretrained_llms.llama2 import LLaMA
from mm_detect.mllms.pretrained_llms.qwen import Qwen
from mm_detect.mllms.pretrained_llms.mistral import Mistral
from mm_detect.mllms.pretrained_llms.phi3 import Phi3
from mm_detect.mllms.pretrained_llms.internlm2 import Internlm2
from mm_detect.mllms.pretrained_llms.yi import Yi
from mm_detect.mllms.pretrained_llms.deepseek_moe import DeepSeek_MOE

import os
import json
import random

logger = get_child_logger("pretrain-detect")

def build_prompt(
    example,
    eval_data_name,
):
    choices = get_answers_list(example, eval_data_name)
    text = example["text"]

    answer_index = get_answer_index(example, eval_data_name)
    answer = choices[answer_index]

    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    option = alphabet[answer_index]

    prompt = f"Please answer the following multichoice question."
    prompt += f"\n\nQuestion: {text}"
    prompt += "\nOptions:"
    for i in range(len(choices)):
        letter = alphabet[i]
        choice = choices[i]
        prompt += f"\n{letter}: {choice}"
    prompt += "\nIf you do not know the answer, output '''I don't know'''.\nOnly reply with your answer."

    return prompt, option, answer

def inference(data_points, eval_data_name, llm):
    results_file = "results.json"
    
    if os.path.exists(results_file) and os.path.getsize(results_file) > 0:
        with open(results_file, "r", encoding="utf-8") as f:
            results = json.load(f)
        last_id = results[-1]["id"] if results else 0
        start_index = len(results)
        id_counter = last_id + 1
        print(f"Resuming from index {start_index}, last id {last_id}")
    else:
        results = []
        start_index = 0
        id_counter = 1

    for i in tqdm(range(start_index, len(data_points))):
        example = data_points[i]
        prompt, option, answer = build_prompt(example, eval_data_name)
        if prompt == "failed":
            print(f"Skipping data point index {i} due to build_prompt failure.")
            continue

        try:
            response, _ = llm.eval_model(prompt=prompt)
        except Exception as e:
            print(f"LLM evaluation error at index {i}: {e}")
            continue

        entry = {
            "id": id_counter,
            "prompt": prompt,
            "response": response,
            "answer": answer
        }
        results.append(entry)
        id_counter += 1

        if (i - start_index + 1) % 10 == 0:
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"Saved intermediate results up to data point index {i}")

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("Final results saved.")

    responses = [entry["response"] for entry in results]
    options = [entry["option"] for entry in results]
    answers = [entry["answer"] for entry in results]

    return responses, options, answers

def main_pretrain_detect(
    eval_data,
    eval_data_name,
    n_eval_data_points,
    # model parameters
    model_name: str = None,
    max_output_tokens: int = 128,
    temperature: float = 0.0,
):
    if "llama" in model_name.lower():
        llm = LLaMA(
            model_name=model_name,
            max_output_tokens=10,
            temperature=0.0
        )
        responses, options, answers = inference(
            eval_data,
            eval_data_name,
            llm
        )
    elif "intern" in model_name.lower(): 
        llm = Internlm2(
            model_name=model_name,
            max_output_tokens=10,
            temperature=0.0
        )
        responses, options, answers = inference(
            eval_data,
            eval_data_name,
            llm
        )
    elif "mistral" in model_name.lower(): 
        llm = Mistral(
            model_name=model_name,
            max_output_tokens=10,
            temperature=0.0
        )
        responses, options, answers = inference(
            eval_data,
            eval_data_name,
            llm
        )
    elif "qwen" in model_name.lower(): 
        llm = Qwen(
            model_name=model_name,
            max_output_tokens=10,
            temperature=0.0
        )
        responses, options, answers = inference(
            eval_data,
            eval_data_name,
            llm
        )
    elif "yi" in model_name.lower(): 
        llm = Yi(
            model_name=model_name,
            max_output_tokens=10,
            temperature=0.0
        )
        responses, options, answers = inference(
            eval_data,
            eval_data_name,
            llm
        )
    elif "phi-3" in model_name.lower(): 
        llm = Phi3(
            model_name=model_name,
            max_output_tokens=10,
            temperature=0.0
        )
        responses, options, answers = inference(
            eval_data,
            eval_data_name,
            llm
        )
    elif "deepseek" in model_name.lower(): 
        llm = DeepSeek_MOE(
            model_name=model_name,
            max_output_tokens=10,
            temperature=0.0
        )
        responses, options, answers = inference(
            eval_data,
            eval_data_name,
            llm
        )

    responses = [x.lower() for x in responses]
    options = [x.lower() for x in options]
    answers = [x.lower() for x in answers]

    num_cont = 0
    for i in range(len(responses)):
        if responses[i] == options[i] or answers[i] in responses[i]:
            num_cont += 1
    cont_rate = num_cont / len(responses)

    logger.info(f"Contamination Rate: {cont_rate:.2%}")
