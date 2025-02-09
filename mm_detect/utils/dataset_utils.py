import re
import ast

def get_answers_list(data_point, dataset_name):
    choices = []

    # derek-thomas/ScienceQA
    if "ScienceQA" in dataset_name:
        choices = data_point["choices"]
    # Lin-Chen/MMStar
    elif "MMStar" in dataset_name:
        text = data_point["text"]
        # Extract question
        question, options_part  = text.split('Options:', 1)
        question = question.strip()
        data_point["text"] = question
        # Extract options
        option_matches = re.findall(r'[A-Z]:\s*([\s\S]*?)(?=,\s*[A-Z]:|$)', options_part)
        choices = [match.strip().rstrip(',.') for match in option_matches]
    elif dataset_name == "MMMU/MMMU_Pro":
        choices = data_point["options"]
        if isinstance(choices, str):
            choices = ast.literal_eval(choices)

    return choices

def get_answer_index(data_point, dataset_name):
    alphabet = "abcdefghijklmnopqrstuvwxyz"

    # derek-thomas/ScienceQA
    if "ScienceQA" in dataset_name:
        answer_index = data_point["answer"]
    # Lin-Chen/MMStar
    elif "MMStar" in dataset_name:
        answer = data_point["answer"].lower()
        answer_index = alphabet.index(answer)
    elif dataset_name == "MMMU/MMMU_Pro":
        answer = data_point["answer"].lower()
        answer_index = alphabet.index(answer)

    return answer_index
