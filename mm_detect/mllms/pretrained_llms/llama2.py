from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import torch
import re

class LLaMA:
    def __init__(
        self,
        model_name: str = None,
        max_output_tokens: int = 30,
        temperature: float = 0.0,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )

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
        sequences = self.pipeline(
            few_shot + prompt + "\nIf you do not know the answer, output I don't know.",
            do_sample=False,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=300,
        )

        response = sequences[0]['generated_text']
        answer = response.splitlines()[-1]
        print(answer)

        return answer, 0

