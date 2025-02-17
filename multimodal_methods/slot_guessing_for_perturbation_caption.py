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
from mm_detect.mllms.deepseek_vl2 import DeepseekVL2

import requests
import io
from io import BytesIO
from PIL import Image
import json

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
        os.environ["CLASSPATH"] = "MM-Detect/stanford-postagger-full-2020-11-17"
        os.environ["STANFORD_MODELS"] = "MM-Detect/stanford-postagger-full-2020-11-17/models"
    st = StanfordPOSTagger('english-bidirectional-distsim.tagger')

    return st

def build_prompt(example, tagger, eval_data_name):
    if example.get("caption"):
        if isinstance(example["caption"], str):
            text = example["caption"]
        else:
            text = random.choice(example["caption"])
    else:
        text = example["text"]

    caption = text

    try:
        zh_text = translator.translate(text, dest='zh-CN').text
        trans_text = translator.translate(zh_text, dest='en').text
        back_caption = trans_text
    except Exception as e:
        print(f"Translation error for text [{text}]: {e}")
        return "failed", "", "", "", "", ""
        
    print(f"Original: {text}")
    print(f"Translated: {zh_text}")
    print(f"Back to English: {trans_text}")

    tags = tagger.tag(text.split())
    words = [x for x in tags if x[1] in ['NN', 'JJ', 'VB']]
    if len(words) == 0:
        return "failed", "", "", "", "", ""
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
        return "", "", "failed", "", "", ""
    idx = np.random.randint(len(new_words))
    new_word = new_words[idx][0].rstrip('.')
    for i in range(len(trans_text)-len(new_word)+1):
        if trans_text[i:(i+len(new_word))] == new_word:
            trans_text = trans_text[:i] + "[MASK]" + trans_text[(i+len(new_word)):]
            break

    new_prompt = "Fill in the [MASK] in one word:"
    new_prompt += f"\n\n{trans_text}"
    new_prompt += "\nOnly reply the word you fill in the [MASK]."

    return prompt, word, new_prompt, new_word, caption, back_caption

def mllm_inference(data_points, n_eval, eval_data_name, llm):
    tagger = get_stanford_tagger()
    
    results_file = "/home/leo/workspace/log/gemini/results.json"
    if os.path.exists(results_file) and os.path.getsize(results_file) > 0:
        with open(results_file, "r") as f:
            results = json.load(f)
        last_id = max(item.get("id", 0) for item in results) if results else 0
        print(f"Resuming from last saved id {last_id}")
        start_index = last_id
    else:
        results = []
        last_id = 0
        start_index = 0

    # For the vintage dataset, ensure that the local image save directory exists
    if "vintage" in eval_data_name:
        save_dir = "/home/leo/workspace/vintage_images/"
        os.makedirs(save_dir, exist_ok=True)
    
    id_counter = last_id + 1 

    for i in tqdm(range(start_index, len(data_points))):
        example = data_points[i]
        prompt, masked_word, new_prompt, new_masked_word, caption, trans_text = build_prompt(
            example, 
            tagger,
            eval_data_name,
        )
        if prompt == "failed" or new_prompt == "failed":
            print(f"Skipping data point index {i} due to translation error.")
            continue

        # If the data point is from the Vintage dataset, download the image and add it to the data point
        if "vintage" in eval_data_name:
            try:
                filename = os.path.join(save_dir, f"{id_counter}.jpg")
                if os.path.exists(filename):
                    with open(filename, "rb") as f:
                        image_bytes = f.read()
                    print(f"Using cached image for id {id_counter}")
                else:
                    img_response = requests.get(example["image_url"])
                    img_response.raise_for_status()
                    image_bytes = img_response.content
                    with open(filename, "wb") as f:
                        f.write(image_bytes)
                    print(f"Saved image for id {id_counter}")
                image = Image.open(io.BytesIO(image_bytes))
                example["image"] = image
            except Exception as e:
                print(f"Exception raised while processing image at index {i}: {e}")
                continue

        id_counter += 1
        
        try:
            response, _ = llm.eval_model(data_point=example, prompt=prompt)
            new_response, _ = llm.eval_model(data_point=example, prompt=new_prompt)
        except Exception as e:
            print(f"LLM evaluation error at index {i}: {e}")
            continue

        entry = {
            "id": id_counter - 1,
            "original_caption": caption,
            "original_mask_word": masked_word,
            "original_model_output": response,
            "back_translated_caption": trans_text,
            "back_translated_mask_word": new_masked_word,
            "back_translated_model_output": new_response
        }
        results.append(entry)

        # Save intermediate results every 10 iterations
        with open(results_file, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Saved intermediate results up to index {i}")

    responses = [entry["original_model_output"] for entry in results]
    masked_words = [entry["original_mask_word"] for entry in results]
    new_responses = [entry["back_translated_model_output"] for entry in results]
    new_masked_words = [entry["back_translated_mask_word"] for entry in results]

    return responses, masked_words, new_responses, new_masked_words

def qwen_inference(data_points, n_eval, eval_data_name, llm):
    # Set the results file path (using a different filename to avoid conflicts)
    results_file = "/home/leo/workspace/log/qwen_results.json"
    
    # If the results file exists and is non-empty, resume from the last saved point
    if os.path.exists(results_file) and os.path.getsize(results_file) > 0:
        with open(results_file, "r", encoding="utf-8") as f:
            results = json.load(f)
        # Get the maximum id from the saved results (or 0 if none found)
        last_id = max(item.get("id", 0) for item in results) if results else 0
        print(f"Resuming from last saved id {last_id}")
        start_index = last_id
    else:
        results = []
        last_id = 0
        start_index = 0

    # For the vintage dataset, ensure that the local image save directory exists
    if "vintage" in eval_data_name:
        save_dir = "/home/leo/workspace/vintage_images/"
        os.makedirs(save_dir, exist_ok=True)

    # Retrieve the Stanford tagger (assuming this function is defined elsewhere)
    tagger = get_stanford_tagger()

    # Initialize id_counter starting from the next id after the last saved id
    id_counter = last_id + 1

    # Iterate over the data points starting from the last saved index
    for i in tqdm(range(start_index, len(data_points))):
        example = data_points[i]
        # Build prompt and related information (masked word, translated prompt, etc.)
        prompt, masked_word, new_prompt, new_masked_word, caption, trans_text = build_prompt(
            example, 
            tagger,
            eval_data_name,
        )
        if prompt == "failed" or new_prompt == "failed":
            print(f"Skipping data point index {i} due to translation error.")
            continue

        # For vintage data, process the image: use cached image if available or download and save locally.
        local_image_path = None
        if "vintage" in eval_data_name:
            try:
                filename = os.path.join(save_dir, f"{id_counter}.jpg")
                if os.path.exists(filename):
                    with open(filename, "rb") as f:
                        image_bytes = f.read()
                    print(f"Using cached image for id {id_counter}")
                else:
                    img_response = requests.get(example["image_url"])
                    img_response.raise_for_status()
                    image_bytes = img_response.content
                    with open(filename, "wb") as f:
                        f.write(image_bytes)
                    print(f"Saved image for id {id_counter}")
                # Set the local image path to pass to the llm.eval_model as the 'id' parameter
                local_image_path = filename
                image = Image.open(io.BytesIO(image_bytes))
                example["image"] = image
            except Exception as e:
                print(f"Exception raised while processing image at index {i}: {e}")
                continue

        # Increase the id_counter after processing the image
        id_counter += 1

        try:
            # Evaluate the model using the original prompt and pass the local image path as 'id'
            response, _ = llm.eval_model(data_point=example, prompt=prompt, id=local_image_path)
            # Evaluate the model using the back-translated prompt
            new_response, _ = llm.eval_model(data_point=example, prompt=new_prompt, id=local_image_path)
        except Exception as e:
            print(f"LLM evaluation error at index {i}: {e}")
            continue

        # Create an entry with the results
        entry = {
            "id": id_counter - 1,
            "original_caption": caption,
            "original_mask_word": masked_word,
            "original_model_output": response,
            "back_translated_caption": trans_text,
            "back_translated_mask_word": new_masked_word,
            "back_translated_model_output": new_response
        }
        results.append(entry)

        # Save intermediate results every 10 iterations
        if (i + 1) % 10 == 0:
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"Saved intermediate results up to index {i}")

    # Save the final results after processing all data points
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Extract and return outputs
    responses = [entry["original_model_output"] for entry in results]
    masked_words = [entry["original_mask_word"] for entry in results]
    new_responses = [entry["back_translated_model_output"] for entry in results]
    new_masked_words = [entry["back_translated_mask_word"] for entry in results]

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
    elif "deepseek" in model_name.lower(): 
        llm = DeepseekVL2(
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

    drops = 0
    for i in range(len(new_responses)):
        if masked_words[i] in responses[i] and new_masked_words[i] not in new_responses[i]:
                drops += 1
    instance_leakage = drops / len(new_responses)

    logger.info(f"Exact Match (EM): {em:.2%}, ROUGE-L F1: {rl:.3f}")
    logger.info(f"Exact Match after Interruption: {new_em:.2%}, ROUGE-L F1 after Interruption: {new_rl:.3f}")
    logger.info(f"EM Difference: {delta:.2%}")
    logger.info(f"Normalized EM Difference: {normalized_delta:.2%}")
    logger.info(f"Instance Leakage: {instance_leakage:.2%}")
