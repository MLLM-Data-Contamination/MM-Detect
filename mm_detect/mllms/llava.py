import argparse
import torch
import os
import json
from tqdm import tqdm

import sys
# The absolute path to the mm_detect directory.
sys.path.append(os.path.abspath("/remote_shome/songdj/workspace/MM-Detect/mm_detect"))

from LLaVA.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from LLaVA.llava.conversation import conv_templates, SeparatorStyle
from LLaVA.llava.model.builder import load_pretrained_model
from LLaVA.llava.utils import disable_torch_init
from LLaVA.llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import requests
from io import BytesIO

class LLaVA:
    def __init__(
        self,
        model_name: str = None,
        max_output_tokens: int = 30,
        temperature: float = 0.0,
    ):
        model = model_name
        self.model_name=get_model_name_from_path(model_name)

        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path=model,
            model_base=None,
            model_name=self.model_name,
        )

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

        if self.model.config.mm_use_im_start_end:
            text = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + text
        else:
            text = DEFAULT_IMAGE_TOKEN + '\n' + text

        if "llama-2" in self.model_name.lower():
            conv_mode = "llava_llama_2"
        elif "mistral" in self.model_name.lower():
            conv_mode = "mistral_instruct"
        elif "v1.6-34b" in self.model_name.lower():
            conv_mode = "chatml_direct"
        elif "v1" in self.model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in self.model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image_tensor = process_images(
            [image], self.image_processor, self.model.config)[0]

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=True if self.temperature > 0 else False,
                max_new_tokens = self.max_output_tokens,
                use_cache=True,
                return_dict_in_generate=True,
                output_scores=True
                )
        sequences = output_ids.sequences
        logits = output_ids.scores

        # Apply softmax to the logits to get probabilities
        probabilities = [torch.nn.functional.log_softmax(log, dim=-1) for log in logits]
        token_logprobs = []
        for i, token_id in enumerate(sequences[0][1:]):
            if token_id == self.tokenizer.eos_token_id:
                break
            probability = probabilities[i][0, token_id].item()
            token_logprobs.append(probability)
        
        output = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)[0].strip() 

        return output, token_logprobs
