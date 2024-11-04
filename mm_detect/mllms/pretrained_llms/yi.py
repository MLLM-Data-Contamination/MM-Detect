from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re

class Yi:
    def __init__(
        self,
        model_name: str = None,
        max_output_tokens: int = 30,
        temperature: float = 0.0,
    ):
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

        self.max_new_tokens = max_output_tokens
        self.temperature = temperature

    def eval_model(self, prompt: str = None):
        few_shot =  '''
Please answer the following multichoice question.

Question: What is in the image?
Options:
A: Big Ben
B: Leaning Tower of Pisa
C: Great Wall
D: Statue of Liberty

Reply with the answer only.
Assistant: Statue of Liberty
'''
        input_text = few_shot + prompt + "\nIf you do not know the answer, output I don't know."
        inputs = self.tokenizer(input_text, return_tensors="pt").to("cuda")
        outputs = self.model.generate(**inputs, max_new_tokens=128)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        matches = re.findall(r'Assistant:(.+)', response)
        if matches:
            answer = matches[1].rstrip('.').strip()
        print(answer)

        return answer, 0

