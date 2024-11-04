from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
import re

class Mistral:
    def __init__(
        self,
        model_name: str = None,
        max_output_tokens: int = 30,
        temperature: float = 0.0,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda")

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
        model_inputs = self.tokenizer([few_shot + prompt + "\nIf you do not know the answer, output I don't know."], return_tensors="pt").to("cuda")
        self.model.to("cuda")

        generated_ids = self.model.generate(**model_inputs, max_new_tokens=40, do_sample=False)
        response = self.tokenizer.batch_decode(generated_ids)[0]

        matches = re.findall(r'Assistant:(.+)', response)
        if matches:
            answer = matches[1].rstrip('.').strip()
        print(answer)
        
        return answer, 0

