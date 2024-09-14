#%%
from datasets import load_dataset
import torchvision.tv_tensors
from dataclasses import dataclass
import torchvision
from pprint import pprint
from PIL import Image
import requests
from transformers import AutoTokenizer
from functools import cache
from torch.utils.data import Dataset,DataLoader
from torchvision.utils import save_image

from config import TrainingConfig

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
conf = TrainingConfig()

@cache
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
        # self.dataset = self.dataset.shuffle(buffer_size=self.config.buffer_size,seed=self.config.seed)
        # self.dataset = self.dataset.take(self.config.num_steps)
        self.dataset = self.dataset.with_format("torch")
        self.remove_invalid_image()
        self.rename_columns()
        self.dataset = self.dataset.map(lambda examples: tokenizer(examples["caption"], padding="max_length", max_length=self.config.max_length))
        self.select_columns()
        self.remove_small_image()

    def rename_columns(self):
        self.dataset = self.dataset.rename_columns({"title":"caption"})
        
    def select_columns(self):
        self.dataset = self.dataset.select_columns(["image","input_ids","token_type_ids","attention_mask"])

    def _url_to_image(self,example):
        key = self.config.url_collumn_name
        url = example[key]
        if url is None:
            example["image"] = None
            return example
        filename = url.split("/")[-1]
        filename = self.config.cache_dir / filename

        if not filename.exists():
            try:
                image = Image.open(requests.get(url, stream=True).raw)
                image.save(filename,quality=95)
                image = torchvision.tv_tensors.Image(image)
            except Exception as e:
                print(e)
                image = None
        else:
            image = Image.open(filename)
            image = torchvision.tv_tensors.Image(image)
        
        example["image"] = image
        return example

    def url_to_image(self):
        self.dataset = self.dataset.map(self._url_to_image)
        
    def remove_small_image(self):
        self.dataset = self.dataset.filter(
            lambda x: x["image"].shape[1] >= self.config.image_size*1.1
            and x["image"].shape[2] >= self.config.image_size*1.1
        )

    def remove_invalid_image(self):
        self.dataset = self.dataset.filter(
            lambda x: x["image"] is not None and x["image"].shape[0] == 3
        )

class StreamDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample


# %%
# %%
