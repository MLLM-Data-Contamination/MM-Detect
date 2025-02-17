import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

class DeepSeek_MOE:
    def __init__(
        self,
        model_name: str = None,
        max_output_tokens: int = 30,
        temperature: float = 0.0,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        self.model.generation_config = GenerationConfig.from_pretrained(model_name)
        self.model.generation_config.pad_token_id = self.model.generation_config.eos_token_id

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
Assistant: D: Statue of Liberty
'''

        messages = [
            # {"role": "user", "content": few_shot},
            {"role": "user", "content": prompt + "\nIf you do not know the answer, output I don't know."},
        ]

        input_tensor = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")

        outputs = self.model.generate(input_tensor.to(self.model.device), max_new_tokens=self.max_new_tokens, do_sample=True if self.temperature > 0 else False)

        result = self.tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
        print("Model Response: ", result)

        return result, 0
