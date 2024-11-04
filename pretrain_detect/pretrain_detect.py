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
    prompt += "\n\nReply with answer only."

    return prompt, option, answer

def inference(
    data_points,
    eval_data_name,
    llm,
):
    responses, options, answers = [], [], []

    for example in tqdm(data_points):
        prompt, option, answer = build_prompt(
            example,
            eval_data_name,
        )
        response, _ = llm.eval_model(prompt=prompt)
        responses.append(response)
        options.append(option)
        answers.append(answer)

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

    responses = [x.lower() for x in responses]
    options = [x.lower() for x in options]
    answers = [x.lower() for x in answers]

    num_cont = 0
    for i in range(len(responses)):
        if responses[i] == options[i] or answers[i] in responses[i]:
            num_cont += 1
    cont_rate = num_cont / len(responses)

    logger.info(f"Contamination Rate: {cont_rate:.2%}")
