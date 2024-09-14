#%%
from datasets import load_dataset
import torchvision.tv_tensors
from config import TrainingConfig
from dataclasses import dataclass
import torchvision
from pprint import pprint
from PIL import Image
import requests

@dataclass
class PreLoad:
    config = TrainingConfig()

    dataset = load_dataset(
        config.dataset_name,
        cache_dir=str(config.cache_dir),
        split=config.split,
        streaming=config.streaming,
    )

    def __post_init__(self):
        self.preprocess()

    
    def check(self):
        dataset = self.dataset
        pprint(f"dataset.features: {dataset.features}")
        data = next(iter(dataset))
        pprint(f"data-> {data}")

    def preprocess(self):
        self.url_to_image()
        self.dataset = self.dataset.with_format("torch")
        # self.dataset = self.dataset.shuffle(buffer_size=self.config.buffer_size,seed=self.config.seed)
        self.remove_invalid_image()
        self.rename_columns()
        self.select_columns()
        self.remove_small_image()

    def rename_columns(self):
        self.dataset = self.dataset.rename_columns({"title":"caption"})
        
    def select_columns(self):
        self.dataset = self.dataset.select_columns(["image","caption"])

    def _url_to_image(self,example):
        key = self.config.url_collumn_name
        url = example[key]
        try:
            image = Image.open(requests.get(url, stream=True).raw)
            image = torchvision.tv_tensors.Image(image)
        except:
            image = None
        example["image"] = image
        return example

    def url_to_image(self):
        self.dataset = self.dataset.map(self._url_to_image)
        
    def remove_small_image(self):
        self.dataset = self.dataset.filter(
            lambda x: x["image"].shape[1] >= self.config.input_image_size*1.1
            and x["image"].shape[2] >= self.config.input_image_size*1.1
        )

    def remove_invalid_image(self):
        self.dataset = self.dataset.filter(
            lambda x: x["image"] is not None and x["image"].shape[0] == 3
        )

if __name__ == "__main__":
    dataset = PreLoad().dataset
    data = next(iter(dataset))
    pprint(f"data ->{data}")

# %%
