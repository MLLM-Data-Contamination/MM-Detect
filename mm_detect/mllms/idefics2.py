import requests
import torch
from PIL import Image
from io import BytesIO

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from transformers.image_utils import load_image

import re

class Idefics2:
    def __init__(
        self,
        model_name: str = None,
        max_output_tokens: int = 30,
        temperature: float = 0.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        checkpoint = model_name
        self.model = AutoModelForVision2Seq.from_pretrained(
            checkpoint,
            torch_dtype=torch.float16,    
            # attn_implementation="flash_attention_2",
            ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(checkpoint)

        self.max_new_tokens = max_output_tokens
        self.temperature = temperature

    def eval_model(self, data_point, prompt: str = None, lower_case: bool = False):
        if prompt is None:
            text = "Please carefully observe the image and come up with a caption for the image."
            if lower_case:
                text = text.lower()
        else:
            text = prompt
        # image1 = load_image("https://images.unsplash.com/photo-1524099163253-32b7f0256868?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=3387&q=80")
        image1 = Image.open("scripts\mllms\slot_guessing_for_perturbation_caption\idefics2_few_shot\image1.png") # The path of the image you load from the URL above.
        
        if data_point.get("image"):
            image = data_point["image"].convert("RGB")
        else:
            url = data_point["image_url"]
            response = requests.get(url)

            img_bytes = BytesIO(response.content)
            image = Image.open(img_bytes).convert("RGB")
        
        if data_point.get("caption"):
            few_shot =  '''
Fill the '[MASK]' of the following sentence in one word:

Majestic view of the [MASK] of Liberty holding her torch high.
Only reply the word you fill in the [MASK].'''

            prompts = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": few_shot},
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Statue"},
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": text},
                    ]
                },  
            ]
            prompt = self.processor.apply_chat_template(prompts, add_generation_prompt=True)
            inputs = self.processor(text=prompt, images=[image1, image], return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        else:
            few_shot =  '''
Please answer the following multichoice question.

Question: What is in the image?
Options:
A: Big Ben
B: Leaning Tower of Pisa
C: Great Wall
D: Statue of Liberty

Reply with the answer only.'''

            # We feed to the model an arbitrary sequence of text strings and images. Images can be either URLs or PIL Images.
            prompts = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": few_shot},
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "D"},
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": text},
                    ]
                },  
            ]
            prompt = self.processor.apply_chat_template(prompts, add_generation_prompt=True)
            inputs = self.processor(text=prompt, images=[image1, image], return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate
        generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        matches = re.findall(r'Assistant:(.+)', generated_texts[0])
        if matches:
            answer = matches[1].rstrip('.').strip()

        return answer, 0
    
    

