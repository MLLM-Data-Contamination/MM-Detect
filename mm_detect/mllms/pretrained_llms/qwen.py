from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch

class Qwen:
    def __init__(
        self,
        model_name: str = None,
        max_output_tokens: int = 30,
        temperature: float = 0.0,
    ):
        # Note: The default behavior now has injection attack prevention off.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # use bf16
        # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
        # use fp16
        # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
        # use cpu only
        # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="cpu", trust_remote_code=True).eval()
        # use auto mode, automatically select precision based on the device.
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda", trust_remote_code=True, fp32=True).eval()

        self.max_new_tokens = max_output_tokens
        self.temperature = temperature

    def eval_model(self, prompt: str = None):
        # Specify hyperparameters for generation. But if you use transformers>=4.32.0, there is no need to do this.
        # model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-7B", trust_remote_code=True)
        inputs = self.tokenizer(prompt + "\nOutput your answer at the last line. If you do not know the answer, output I don't know.", return_tensors='pt')
        inputs = inputs.to("cuda")
        pred = self.model.generate(**inputs)
        response = self.tokenizer.decode(pred.cpu()[0], skip_special_tokens=True).splitlines()[-1]
        print(response)
        
        return response, 0

