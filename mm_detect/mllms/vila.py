import torch
import re
from PIL import Image
import requests
from io import BytesIO

import os
import sys
sys.path.append(os.path.abspath("/home/leo/workspace/MM-Detect/mm_detect/packages"))

from VILA.llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_PLACEHOLDER,
    IMAGE_TOKEN_INDEX,
)
from VILA.llava.conversation import SeparatorStyle, conv_templates
from VILA.llava.mm_utils import KeywordsStoppingCriteria, get_model_name_from_path, process_images, tokenizer_image_token
from VILA.llava.model.builder import load_pretrained_model
from VILA.llava.utils import disable_torch_init

class VILA:
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

        self.max_new_tokens = max_output_tokens
        self.temperature = temperature

    def eval_model(self, data_point, prompt: str = None, lower_case: bool = False):
        if prompt is None:
            qs = "Please carefully observe the image and come up with a caption for the image."
            if lower_case:
                qs = qs.lower()
        else:
            qs = prompt
            
        if data_point.get("image"):
            image = data_point["image"].convert("RGB")
        else:
            url = data_point["image_url"]
            response = requests.get(url)

            img_bytes = BytesIO(response.content)
            image = Image.open(img_bytes).convert("RGB")

        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in qs:
            if self.model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if DEFAULT_IMAGE_TOKEN not in qs:
                # print("no <image> tag found in input. Automatically append one at the beginning of text.")
                # do not repeatively append the prompt.
                if self.model.config.mm_use_im_start_end:
                    qs = (image_token_se + "\n") + qs
                else:
                    qs = (DEFAULT_IMAGE_TOKEN + "\n") + qs
        # print("input: ", qs)
        
        if "llama-2" in self.model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in self.model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in self.model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image_tensor = process_images([image], self.image_processor, self.model.config).to(self.model.device, dtype=torch.float16)
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=[
                    image_tensor,
                ],
                do_sample=True if self.temperature > 0 else False,
                max_new_tokens=self.max_new_tokens,
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