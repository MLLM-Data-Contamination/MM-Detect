import torch
from openai import OpenAI

from PIL import Image
import io
import base64
import requests
from io import BytesIO

def encode_image(image):
    buffered = io.BytesIO()
    image_format = image.format 
    image.save(buffered, format=image_format)
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return image_format.lower(), img_str

class GPT:
    def __init__(
        self,
        model_name: str = None,
        max_output_tokens: int = 30,
        temperature: float = 0.0,
    ):
        # load model and processor
        self.client = OpenAI(
            api_key='YOUR_API_KEY', # Replace with your OpenAI API key.
        )

        self.model = model_name
        self.max_tokens = max_output_tokens
        self.temperature = temperature

    def request_gpt(self, gpt_prompt, image):
        image_format, base64_image = encode_image(image)
        gpt_response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                "role": "user",
                "content": [
                    {"type": "text", "text": gpt_prompt},
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{image_format};base64,{base64_image}",
                    },
                    },
                ],
                }
            ],
            max_tokens = self.max_tokens,
            temperature = self.temperature,
        )

        if gpt_response.choices is None: return ""
        response = gpt_response.choices[0].message.content

        return response

    def request_gpt_url(self, gpt_prompt, image_url):
        gpt_response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                "role": "user",
                "content": [
                    {"type": "text", "text": gpt_prompt},
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                    },
                    },
                ],
                }
            ],
            max_tokens = self.max_tokens,
            temperature = self.temperature,
        )

        if gpt_response.choices is None: return ""
        response = gpt_response.choices[0].message.content

        return response

    def eval_model(self, data_point, prompt: str = None, lower_case: bool = False):
        if prompt is None:
            text = "Please carefully observe the image and come up with a caption for the image."
            if lower_case:
                text = text.lower()
        else:
            text = prompt
        text += "\n"
        
        if data_point.get("image"):
            image = data_point["image"]
            response = self.request_gpt(text, image)
        else:
            url = data_point["image_url"]
            response = self.request_gpt_url(text, url)
        print(response)

        return response, 0

