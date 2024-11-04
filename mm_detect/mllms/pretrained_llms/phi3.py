import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class Phi3:
    def __init__(
        self,
        model_name: str = None,
        max_output_tokens: int = 30,
        temperature: float = 0.0,
    ):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype="auto", 
            trust_remote_code=True,
        )
        self.model = self.model.to("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.max_new_tokens = max_output_tokens
        self.temperature = temperature

    def eval_model(self, prompt: str = None):
        messages = [
            {"role": "user", "content": prompt + "\nIf you do not know the answer, output I don't know."},
        ]

        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device="cuda"
        )

        generation_args = {
            "max_new_tokens": 100,
            "do_sample": False,
        }

        output = pipe(messages, **generation_args)
        response = output[0]['generated_text']
        answer = response[1]['content']
        print(answer)

        return response, 0

