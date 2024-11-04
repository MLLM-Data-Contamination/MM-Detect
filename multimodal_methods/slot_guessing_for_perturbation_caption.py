import os
import numpy as np
from tqdm import tqdm
import random
from rouge_score import rouge_scorer
from nltk.tokenize import word_tokenize
from nltk.tag import StanfordPOSTagger
from mm_detect.utils.logger import get_child_logger, suspend_logging
from mm_detect.utils.dataset_utils import get_answers_list

from mm_detect.mllms.llava import LLaVA
from mm_detect.mllms.vila import VILA
from mm_detect.mllms.qwen_vl import QwenVL
from mm_detect.mllms.fuyu import Fuyu
from mm_detect.mllms.internvl2 import InterVL2
from mm_detect.mllms.yi_vl import YiVL
from mm_detect.mllms.phi3_vision import Phi3
from mm_detect.mllms.idefics2 import Idefics2
from mm_detect.mllms.gpt import GPT

import requests
from io import BytesIO
from PIL import Image

from googletrans import Translator
translator = Translator()

logger = get_child_logger("slot_guessing_for_perturbation_caption")

def get_stanford_tagger():
    if not("CLASSPATH" in os.environ and "STANFORD_MODELS" in os.environ):
        logger.info("You are using a model contamination detection method which requires Stanford's Part-of-Speech tagger.")
        logger.info("You need to setup global variables CLASSPATH and STANFORD_MODELS specifying the path to the tagger.")
        logger.info("First download the tagger here: https://nlp.stanford.edu/software/tagger.html#Download")
        logger.info("Then place it into some directory.")
        # home_dir = input("Please specify the directory where you place the tagger (Example: /home/leo/stanford-postagger-full-2020-11-17): ")
        os.environ["CLASSPATH"] = "stanford-postagger-full-2020-11-17"
        os.environ["STANFORD_MODELS"] = "stanford-postagger-full-2020-11-17/models"
    st = StanfordPOSTagger('english-bidirectional-distsim.tagger')

    return st

def build_prompt(
    example, 
    tagger,
    eval_data_name
):
    if example.get("caption"):
        if isinstance(example["caption"], str):
            text = example["caption"]
        else:
            text = random.choice(example["caption"])
    else:
        text = example["text"]

    caption = text

    zh_text = translator.translate(text, dest='zh-CN').text
    trans_text = translator.translate(zh_text, dest='en').text

    tags = tagger.tag(text.split())
    words = [x for x in tags if x[1] in ['NN', 'JJ', 'VB']]
    if len(words) == 0:
        return "failed", "", "", "", ""
    idx = np.random.randint(len(words))
    word = words[idx][0].rstrip('.')
    for i in range(len(text)-len(word)+1):
        if text[i:(i+len(word))] == word:
            text = text[:i] + "[MASK]" + text[(i+len(word)):]
            break

    prompt = "Fill the '[MASK]' of the following sentence in one word:"
    prompt += f"\n\n{text}"
    prompt += "\nOnly reply the word you fill in the [MASK]."

    tags = tagger.tag(trans_text.split())
    new_words = [x for x in tags if x[1] in ['NN', 'JJ', 'VB']]
    if len(new_words) == 0:
        return "", "", "failed", "", ""
    idx = np.random.randint(len(new_words))
    new_word = new_words[idx][0].rstrip('.')
    for i in range(len(trans_text)-len(new_word)+1):
        if trans_text[i:(i+len(new_word))] == new_word:
            trans_text = trans_text[:i] + "[MASK]" + trans_text[(i+len(new_word)):]
            break

    new_prompt = "Fill in the [MASK] in one word:"
    new_prompt += f"\n\n{trans_text}"
    new_prompt += "\nOnly reply the word you fill in the [MASK]."

    return prompt, word, new_prompt, new_word, caption

def mllm_inference(
    data_points, 
    n_eval, 
    eval_data_name,
    llm, 
):
    tagger = get_stanford_tagger()

    responses, masked_words, new_responses, new_masked_words = [], [], [], []
    for example in tqdm(data_points):
        prompt, masked_word, new_prompt, new_masked_word, caption = build_prompt(
            example, 
            tagger,
            eval_data_name,
        )
        if prompt == "failed" or new_prompt == "failed":
            continue

        response, _ = llm.eval_model(data_point=example, prompt=prompt)
        new_response, _ = llm.eval_model(data_point=example, prompt=new_prompt)

        responses.append(response)
        masked_words.append(masked_word)
        new_responses.append(new_response)
        new_masked_words.append(new_masked_word)

    return responses, masked_words, new_responses, new_masked_words

def qwen_inference(
    data_points, 
    n_eval, 
    eval_data_name,
    llm, 
):
    tagger = get_stanford_tagger()

    responses, masked_words, new_responses, new_masked_words = [], [], [], []
    id = 1
    for example in tqdm(data_points):
        # Skip Nocaps' 117 item
        if eval_data_name == "lmms-lab/NoCaps" and id == 117:
            id += 1
            continue

        prompt, masked_word, new_prompt, new_masked_word, caption = build_prompt(
            example, 
            tagger,
            eval_data_name,
        )
        if prompt == "failed" or new_prompt == "failed":
            continue

        response, _ = llm.eval_model(example, prompt=prompt, id=id)
        new_response, _ = llm.eval_model(example, prompt=new_prompt, id=id)
        id += 1

        responses.append(response)
        masked_words.append(masked_word)
        new_responses.append(new_response)
        new_masked_words.append(new_masked_word)

    return responses, masked_words, new_responses, new_masked_words

def main_slot_guessing_for_perturbation_caption(
    eval_data,
    eval_data_name,
    n_eval_data_points,
    # model parameters
    model_name: str = None,
    max_output_tokens: int = 128,
    temperature: float = 0.0,
):
    data_points = []
    for i in range(n_eval_data_points):
        data_points.append(eval_data[i])

    if "llava" in model_name.lower():
        llm = LLaVA(
            model_name=model_name,
            max_output_tokens=max_output_tokens,
            temperature=temperature
        )

        responses, masked_words, new_responses, new_masked_words = mllm_inference(
            data_points, 
            n_eval_data_points,
            eval_data_name,
            llm
        )
    elif "vila" in model_name.lower():
        llm = VILA(
            model_name=model_name,
            temperature=0.0
        )
        responses, masked_words, new_responses, new_masked_words = mllm_inference(
            data_points, 
            n_eval_data_points,
            eval_data_name,
            llm
        )
    elif "internvl2" in model_name.lower(): 
        llm = InterVL2(
            model_name=model_name,
            temperature=0.0
        )
        responses, masked_words, new_responses, new_masked_words = mllm_inference(
            data_points, 
            n_eval_data_points,
            eval_data_name,
            llm
        )
    elif "idefics" in model_name.lower(): 
        llm = Idefics2(
            model_name=model_name,
            temperature=0.0
        )
        responses, masked_words, new_responses, new_masked_words = mllm_inference(
            data_points, 
            n_eval_data_points,
            eval_data_name,
            llm
        )
    elif "qwen" in model_name.lower(): 
        llm = QwenVL(
            model_name=model_name,
            temperature=0.0
        )
        responses, masked_words, new_responses, new_masked_words = qwen_inference(
            data_points, 
            n_eval_data_points,
            eval_data_name,
            llm
        )
    elif "fuyu" in model_name.lower(): 
        llm = Fuyu(
            model_name=model_name,
            temperature=0.0
        )
        responses, masked_words, new_responses, new_masked_words = mllm_inference(
            data_points, 
            n_eval_data_points,
            eval_data_name,
            llm
        )
    elif "yi-vl" in model_name.lower(): 
        llm = YiVL(
            model_name=model_name,
            temperature=0.0
        )
        responses, masked_words, new_responses, new_masked_words = mllm_inference(
            data_points, 
            n_eval_data_points,
            eval_data_name,
            llm
        )
    elif "phi-3-vision" in model_name.lower(): 
        llm = Phi3(
            model_name=model_name,
            temperature=0.0
        )
        responses, masked_words, new_responses, new_masked_words = mllm_inference(
            data_points, 
            n_eval_data_points,
            eval_data_name,
            llm
        )

    elif "gpt" in model_name.lower() or "gemini" in model_name.lower() or "claude" in model_name.lower(): 
        llm = GPT(
            model_name=model_name,
            temperature=0.0
        )
        responses, masked_words, new_responses, new_masked_words = mllm_inference(
            data_points, 
            n_eval_data_points,
            eval_data_name,
            llm
        )
    
    responses = [x.lower() for x in responses]
    masked_words = [x.lower() for x in masked_words]
    new_responses = [x.lower() for x in new_responses]
    new_masked_words = [x.lower() for x in new_masked_words]

    em = len([i for i in range(len(responses)) if masked_words[i] in responses[i]]) / len(responses)
    new_em = len([i for i in range(len(new_responses)) if new_masked_words[i] in new_responses[i]]) / len(new_responses)

    scorer = rouge_scorer.RougeScorer(['rougeLsum'], use_stemmer=True)
    rl = np.mean(np.array([scorer.score(responses[i], masked_words[i])["rougeLsum"].fmeasure for i in range(len(responses))]))
    new_rl = np.mean(np.array([scorer.score(new_responses[i], new_masked_words[i])["rougeLsum"].fmeasure for i in range(len(new_responses))]))

    delta = new_em - em
    normalized_delta = delta / em

    if delta > -1:
        drops = 0
        for i in range(len(new_responses)):
            if masked_words[i] in responses[i] and new_masked_words[i] not in new_responses[i]:
                    drops += 1
        instance_leakage = drops / len(new_responses)
    else:
        instance_leakage = None

    logger.info(f"Exact Match (EM): {em:.2%}, ROUGE-L F1: {rl:.3f}")
    logger.info(f"Exact Match after Interruption: {new_em:.2%}, ROUGE-L F1 after Interruption: {new_rl:.3f}")
    logger.info(f"EM Difference: {delta:.2%}")
    logger.info(f"Normalized EM Difference: {normalized_delta:.2%}")
    if instance_leakage != None:
        logger.info(f"Instance Leakage: {instance_leakage:.2%}")
