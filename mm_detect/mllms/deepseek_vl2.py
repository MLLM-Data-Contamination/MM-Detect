import torch
from transformers import AutoModelForCausalLM

import sys
sys.path.append("/home/leo/workspace/MM-Detect/mm_detect/packages/DeepSeek-VL2")
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images

from PIL import Image
import requests
from io import BytesIO

class DeepseekVL2:
    def __init__(
        self,
        model_name: str = None,
        max_output_tokens: int = 30,
        temperature: float = 0.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        # specify the path to the model
        model_path = model_name
        self.vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer

        self.vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        self.vl_gpt = self.vl_gpt.to(torch.bfloat16).cuda().eval()

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

        ## single image conversation example
        ## Please note that <|ref|> and <|/ref|> are designed specifically for the object localization feature. These special tokens are not required for normal conversations.
        ## If you would like to experience the grounded captioning functionality (responses that include both object localization and reasoning), you need to add the special token <|grounding|> at the beginning of the prompt. Examples could be found in Figure 9 of our paper.
        # conversation = [
        #     {
        #         "role": "<|User|>",
        #         "content": "<image>\n<|ref|>The giraffe at the back.<|/ref|>.",
        #         "images": ["./images/visual_grounding_1.jpeg"],
        #     },
        #     {"role": "<|Assistant|>", "content": ""},
        # ]

        conversation = [
            {
                "role": "<|User|>",
                "content": "<image>\n" + text,
                # "images": [""],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        # load images and prepare for inputs
        prepare_inputs = self.vl_chat_processor(
            conversations=conversation,
            images=[image],
            force_batchify=True,
            system_prompt=""
        ).to(self.vl_gpt.device)

        # run image encoder to get the image embeddings
        inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        # run the model to get the response
        outputs = self.vl_gpt.language.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=self.max_new_tokens,
            do_sample=True if self.temperature > 0 else False,
            use_cache=True
        )

        eos = "<｜end▁of▁sentence｜>"

        answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=False)
        # print(answer)
        output = answer.split(eos)[0]
        # print(output)

        return output, 0
