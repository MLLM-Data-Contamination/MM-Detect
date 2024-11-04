import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

class Internlm2:
    def __init__(
        self,
        model_name: str = None,
        max_output_tokens: int = 30,
        temperature: float = 0.0,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # Set `torch_dtype=torch.float16` to load model in float16, otherwise it will be loaded as float32 and might cause OOM Error.
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, trust_remote_code=True).cuda()
        self.model = model.eval()

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
        inputs = self.tokenizer([few_shot + prompt + "\nIf you do not know the answer, output I don't know." + "\nAssistant: "], return_tensors="pt")
        for k,v in inputs.items():
            inputs[k] = v.cuda()
        gen_kwargs = {"max_new_tokens": 300, "do_sample": False}
        output = self.model.generate(**inputs, **gen_kwargs)
        response = self.tokenizer.decode(output[0].tolist(), skip_special_tokens=True)

        matches = re.findall(r'Assistant:(.+)', response)
        if matches:
            answer = matches[1].rstrip('.').strip()
        print(answer)

        return answer, 0

