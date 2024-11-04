from transformers import FuyuProcessor, FuyuForCausalLM
import torch
import re

from PIL import Image
import requests
from io import BytesIO

class Fuyu:
    def __init__(
        self,
        model_name: str = None,
        max_output_tokens: int = 30,
        temperature: float = 0.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        # load model and processor
        self.processor = FuyuProcessor.from_pretrained(model_name)
        self.model = FuyuForCausalLM.from_pretrained(model_name, device_map=device)

        self.device = device
        self.max_new_tokens = max_output_tokens
        self.temperature = temperature

    def eval_model(self, data_point, prompt: str = None, lower_case: bool = False):
        if prompt is None:
            text = "Please carefully observe the image and come up with a caption for the image."
            if lower_case:
                text = text.lower()
        else:
            text = prompt
        text += "\n"
        
        if data_point.get("image"):
            image = data_point["image"].convert("RGB")
        else:
            url = data_point["image_url"]
            response = requests.get(url)

            img_bytes = BytesIO(response.content)
            image = Image.open(img_bytes).convert("RGB")

        inputs = self.processor(text=text, images=image, return_tensors="pt").to(self.device)

        # autoregressively generate text
        generation_output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, temperature = self.temperature)
        generation_text = self.processor.batch_decode(generation_output[:, -7:], skip_special_tokens=True)
        
        # print(generation_text[0])
        if "\x04" in generation_text[0]:
            match = re.search(r'\x04(.*)', generation_text[0])
            if match:
                result = match.group(1).strip()
        else:
            result = generation_text[0]

        return result, 0
