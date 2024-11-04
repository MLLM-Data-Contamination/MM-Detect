import re
import ast

def get_answers_list(data_point, dataset_name):
    choices = []
    if dataset_name == "derek-thomas/ScienceQA":
        choices = data_point["choices"]
    elif dataset_name == "Lin-Chen/MMStar":
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
    if dataset_name == "derek-thomas/ScienceQA":
        answer_index = data_point["answer"]
    elif dataset_name == "Lin-Chen/MMStar":
        answer = data_point["answer"].lower()
        answer_index = alphabet.index(answer)
    elif dataset_name == "MMMU/MMMU_Pro":
        answer = data_point["answer"].lower()
        answer_index = alphabet.index(answer)

    return answer_index
