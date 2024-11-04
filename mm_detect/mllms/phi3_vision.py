import argparse
import torch
import os
import json
from tqdm import tqdm

from transformers import AutoModelForCausalLM 
from transformers import AutoProcessor 


from PIL import Image
import requests
from io import BytesIO

class Phi3:
    def __init__(
        self,
        model_name: str = None,
        max_output_tokens: int = 30,
        temperature: float = 0.0,
    ):
        model_id = model_name
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            device_map="cuda", 
            trust_remote_code=True, 
            torch_dtype="auto", 
            # _attn_implementation='flash_attention_2',
            _attn_implementation='eager',
            ) # use _attn_implementation='eager' to disable flash attention

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True) 

        self.max_output_tokens = max_output_tokens
        self.temperature = temperature

    def eval_model(self, data_point, prompt: str = None, lower_case: bool = False):
        if prompt is None:
            text = "Please carefully observe the image and come up with a caption for the image."
            if lower_case:
                text = text.lower()
        else:
            text = prompt

        if data_point.get("image"):
            image = data_point["image"].convert("RGB")
        else:
            url = data_point["image_url"]
            response = requests.get(url)

            img_bytes = BytesIO(response.content)
            image = Image.open(img_bytes).convert("RGB")
            
        messages = [ 
            {"role": "user", "content": f"<|image_1|>\n{text}"}, 
        ] 

        prompt = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.processor(prompt, [image], return_tensors="pt").to("cuda") 

        generation_args = { 
            "max_new_tokens": self.max_output_tokens, 
            "do_sample": False, 
        } 

        generate_ids = self.model.generate(**inputs, eos_token_id=self.processor.tokenizer.eos_token_id, **generation_args) 

        # remove input tokens 
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 

        return response, 0
