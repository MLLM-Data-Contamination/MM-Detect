import re
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

from mm_detect.configs.config import supported_methods, config
from mm_detect.utils.logger import get_child_logger

from PIL import Image
from io import BytesIO
import requests

logger = get_child_logger("base")

def image_to_base64(image):
    import base64
    from io import BytesIO
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

class BaseContaminationChecker:
    def __init__(self, args):
        for key, value in args.__dict__.items():
            setattr(self, key, value)
        self.supported_methods = supported_methods

        # download the datasets -> using HF's datasets
        self.download_data()

        # subsample the eval dataset
        self.subsample_eval_data()

        # standardize the text field
        if self.text_key:
            self.normalize_text_key()

        if self.caption_key:
            self.normalize_caption_key()

        if self.eval_data_name == "MMMU/MMMU_Pro":
            self.normalize_image_key()

    def download_data(self):
        if self.eval_data_name:
            self.eval_data = load_dataset(self.eval_data_name, self.eval_data_config_name)
            if self.eval_set_key:
                self.eval_data = self.eval_data[self.eval_set_key]
        else:
            self.eval_data = []
        
        logger.info(f"There are {len(self.eval_data)} eval data points")

    def subsample_eval_data(self):
        if len(self.eval_data) > 0 and self.n_eval_data_points > 0:
            if self.eval_data_name == "lmms-lab/COCO-Caption2017" or self.eval_data_name == "lmms-lab/NoCaps":
                p = np.random.permutation(len(self.eval_data))
                selected_indices = []
                for idx in p:
                    if len(selected_indices) == self.n_eval_data_points:
                        break
                    
                    idx = int(idx)
                    data_point = self.eval_data[idx]
                    if data_point["image"] != None:
                        selected_indices.append(idx)
                self.eval_data = self.eval_data.select(selected_indices)

            elif self.eval_data_name == "SilentAntagonist/vintage-artworks-60k-captioned":
                p = np.random.permutation(len(self.eval_data))
                selected_indices = []
                for idx in p:
                    if len(selected_indices) == self.n_eval_data_points:
                        break

                    idx = int(idx)
                    data_point = self.eval_data[idx]
                    # url = data_point["image_url"]
                    if len(data_point["short_caption"]) <= 200:
                        selected_indices.append(idx)
                self.eval_data = self.eval_data.select(selected_indices)

            elif self.eval_data_name == "derek-thomas/ScienceQA":
                p = np.random.permutation(len(self.eval_data))
                selected_indices = []
                for idx in p:
                    if len(selected_indices) == self.n_eval_data_points:
                        break
                    
                    idx = int(idx)
                    data_point = self.eval_data[idx]
                    if data_point["image"] != None and len(data_point["choices"]) > 2:
                        selected_indices.append(idx)
                self.eval_data = self.eval_data.select(selected_indices)

            elif self.eval_data_name == "Lin-Chen/MMStar":
                p = np.random.permutation(len(self.eval_data))
                selected_indices = []
                for idx in p:
                    if len(selected_indices) == self.n_eval_data_points:
                        break
                    
                    idx = int(idx)
                    data_point = self.eval_data[idx]
                    if data_point["image"] != None and "Options:" in data_point["question"]:
                        selected_indices.append(idx)
                self.eval_data = self.eval_data.select(selected_indices)
            elif self.eval_data_name == "MMMU/MMMU_Pro":
                p = np.random.permutation(len(self.eval_data))
                selected_indices = []
                for idx in p:
                    if len(selected_indices) == self.n_eval_data_points:
                        break
                    
                    idx = int(idx)
                    data_point = self.eval_data[idx]
                    if data_point["image_1"] != None and data_point["image_2"] == None:
                        selected_indices.append(idx)
                self.eval_data = self.eval_data.select(selected_indices)

            logger.info(f"After subsampling, there are now {len(self.eval_data)} eval data points")

    def normalize_text_key(self):
        if self.eval_data:
            self.eval_data = self.normalize_text_key_(self.eval_data)

    def normalize_text_key_(self, subset):
        if self.text_key != "text":
            assert self.text_key in subset.features, "Error - please provide a text key that is in this dataset"
            subset = subset.add_column("text", subset[self.text_key])

        return subset
    
    def normalize_caption_key(self):
        self.eval_data = self.normalize_caption_key_(self.eval_data)

    def normalize_caption_key_(self, subset):
        assert self.caption_key in subset.features, "Error - please provide a caption key that is in this dataset"
        subset = subset.add_column("caption", subset[self.caption_key])

        return subset
    
    def normalize_image_key(self):
        self.eval_data = self.normalize_image_key_(self.eval_data)

    def normalize_image_key_(self, subset):
        assert self.image_key in subset.features, "Error - please provide an image key that is in this dataset"
        subset = subset.rename_column(self.image_key, "image")

        return subset
