from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch

from PIL import Image
import requests
from io import BytesIO

class QwenVL:
    def __init__(
        self,
        model_name: str = None,
        max_output_tokens: int = 30,
        temperature: float = 0.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        # Note: The default behavior now has injection attack prevention off.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # use bf16
        # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
        # use fp16
        # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
        # use cpu only
        # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cpu", trust_remote_code=True).eval()
        # use cuda device
        self.model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map=device, trust_remote_code=True, fp32=True).eval()

        self.max_new_tokens = max_output_tokens
        self.temperature = temperature

    def eval_model(self, data_point, prompt: str = None, lower_case: bool = False, id: int = 1):
        if prompt is None:
            text = "Please carefully observe the image and come up with a caption for the image."
            if lower_case:
                text = text.lower()
        else:
            text = prompt

        if not data_point.get("image_url"):
            image = data_point["image"].convert("RGB")
            # You need to save the images to a directory since QwenVL cannot process PIL images directly.
            if data_point.get("caption"):
                image_path = "qwen_caption_images/" + str(id) + ".png"
            else:
                image_path = "qwen_multichoice_images/" + str(id) + ".png"
            image.save(image_path)
        else:
            image_path = data_point["image_url"]
            
        # Specify hyperparameters for generation (No need to do this if you are using transformers>=4.32.0)
        # model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

        query = self.tokenizer.from_list_format([
            {'image': image_path},
            {'text': text},
        ])
        response, history = self.model.chat(self.tokenizer, query=query, history=None)

        return response, 0

