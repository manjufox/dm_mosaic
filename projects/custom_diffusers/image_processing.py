#%%
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.v2 import functional as tF
import torchvision
import torchvision.transforms.v2 as v2
from config import TrainingConfig
import numpy as np
from pprint import pprint
from dataset import PreLoad
from torch.utils.data import DataLoader
from PIL import Image
import sys
sys.path.append("../..")
sys.path.append("..")
sys.path.append(".")

from util.image_util import display_images

@dataclass
class ImageProcessing:
    config = TrainingConfig()
    zoom_out_range = (1, 2)
    kernel_size_range = (1, 2)
    noise_scale = 0.1
    step = 0

    batch_size = config.batch_size
    input_image_size = config.image_size
    train_img_image_size = config.image_size

    value = {}

    def set_image(self, x):
        x = torchvision.tv_tensors.Image(x)
        x = self.to_tensor(x)
        train_img = self.normalize(x)
        eval_img = self.normalize(x)
        self.value.update({"image": x,"img_eval":eval_img,"img_train":train_img})

    def do_augument(self, time_step=1):
        self.generate_train_label()
        func_list = [
            self.random_blur,
            self.random_affine,
            self.random_rotate,
            self.random_crop,
            self.random_mosaic,
        ]
        for i in range(time_step):
            for func in func_list:
                func()
                
    def collate_fn(self,examples):
        if isinstance(examples,dict):
            examples = [examples]
        for example in examples:
            self.set_image(example["image"])
            self.value.update({"h":self.value["img_train"].shape[1], "w":self.value["img_train"].shape[2]})
            # self.do_augument(1)
            example.update(self.value)

        batch = {}
        for key in examples[0].keys():
            batch[key] = self.to_batch(examples,key)

        return batch

    def generate_train_label(self):
        d = self.value
        self.step += 1
        d["step"] = torch.tensor([self.step],dtype=torch.float32)
        d["h"] = torch.tensor([d["img_train"].shape[1]],dtype=torch.float32)
        d["w"] = torch.tensor([d["img_train"].shape[2]],dtype=torch.float32)
        d["x"] = (torch.rand(1) -0.5) * d["img_train"].shape[1] * self.noise_scale
        d["y"] = (torch.rand(1) -0.5) * d["img_train"].shape[2] * self.noise_scale
        d["angle"] = (torch.rand(1) -0.5) * 180 * self.noise_scale
        d["scale_factor"] = torch.randint(self.zoom_out_range[0], self.zoom_out_range[1], size=(1,)).float()
        d["shear_x"] = torch.randint(-int(d["w"].item()), int(d["w"].item()), size=(1,)).float() * self.noise_scale
        d["shear_y"] = torch.randint(-int(d["h"].item()), int(d["h"].item()), size=(1,)).float() * self.noise_scale
        d["kernel_size"] = torch.randint(self.kernel_size_range[0], self.kernel_size_range[1], size=(2,)).float()*2-1
        d["b_sigma"] = torch.rand(1).float()
        d["mosaic_ratio"] = torch.randint(1,16,size=(1,)).float()
        
        train_label = torch.cat([
            d["step"],
            d["h"],
            d["w"],
            d["x"],
            d["y"],
            d["angle"],
            d["scale_factor"],
            d["shear_x"],
            d["shear_y"],
            d["kernel_size"][0].view(-1),
            d["kernel_size"][1].view(-1),
            d["b_sigma"].view(-1),
            d["mosaic_ratio"],
        ])

        true_label =torch.cat([
            torch.tensor([0.0],dtype=torch.float32),
            d["h"],
            d["w"],
            torch.tensor([0.0],dtype=torch.float32),
            torch.tensor([0.0],dtype=torch.float32),
            torch.tensor([0.0],dtype=torch.float32),
            torch.tensor([1.0],dtype=torch.float32),
            torch.tensor([0.0],dtype=torch.float32),
            torch.tensor([0.0],dtype=torch.float32),
            torch.tensor([0.0],dtype=torch.float32),
            torch.tensor([0.0],dtype=torch.float32),
            torch.tensor([0.0],dtype=torch.float32),
        ])
        self.value.update({"train_label": train_label,"true_label":true_label,"label_size":len(train_label)})
        return train_label

    def to_tensor(self,x=None):
        x = v2.Compose([
            v2.Resize(size=(self.config.image_size, self.config.image_size)),
            v2.Lambda(lambda x:v2.PILToTensor()(x) if isinstance(x,Image.Image) else x),
            v2.ToDtype(torch.float32),
            ])(x)
        return x
    
    def to_train(self,x=None):
        x = self.normalize(x)
        return x
    
    def to_eval(self,x=None):
        x = v2.Compose(
            [
                v2.Lambda(lambda x:torchvision.tv_tensors.Image(x) if not isinstance(x,torch.Tensor) else x),
                v2.Resize(size=(self.config.image_size, self.config.image_size)),
                v2.ToDtype(torch.float32),
                # v2.Normalize(mean=[0.5],std=[0.5]),
            ]
        )(x)
        return x

    def normalize(self,x=None):
        return v2.Compose(
            [
                # v2.Lambda(lambda x:x/255.0),
                v2.Normalize(mean=[0.5],std=[0.5])
            ]
        )(x)

    def de_normalize(self,x=None):
        return v2.Compose(
            [
                # v2.Normalize(mean=-1 * self.mean / self.std, std=1 / self.std),
                v2.Lambda(lambda x:(x+1)*255/2),
            ]
        )(x)

    def random_affine(self,x=None):
        d = self.value
        x = tF.affine(x,angle=d["angle"].item(),translate=(d["x"].item(),d["y"].item()),scale=d["scale_factor"].item(),shear=(d["shear_x"].item(),d["shear_y"].item()))
        return x

    def random_crop(self,x=None):
        d = self.value
        x = tF.crop(x,d["x"].int(),d["y"].int(),self.config.image_size,self.config.image_size)
        return x

    def random_zoomout(self,x=None):
        d = self.value
        c,h,w = x.shape
        x = tF.resize(x,size=[h*d["scale_factor"].int(),w*d["scale_factor"].int()])
        x = tF.center_crop(x,self.config.image_size)
        return x

    def random_rotate(self,x=None):
        d = self.value
        c,h,w = x.shape
        angle = d["angle"]
        x = tF.rotate(x, angle)
        return x

    def random_blur(self,x=None):
        d = self.value
        kernel_size = d["kernel_size"].int().tolist()
        sigma = d["b_sigma"].item()
        x = tF.gaussian_blur(x, kernel_size=kernel_size,sigma=sigma)
        return x

    def random_noise(self,x=None):
        x = x + torch.randint_like(x,0,255)
        return x

    def random_mosaic(self,x=None):
        d = self.value
        c,h,w = x.shape
        x = tF.resize(x,size=((h//d["mosaic_ratio"]).int(),(w//d["mosaic_ratio"]).int()))
        x = tF.resize(x,size=(h,w),interpolation=tF.InterpolationMode.NEAREST)
        return x
    
    def to_batch(self,data,key):
        for example in data:
            if isinstance(example[key],torch.Tensor):
                return torch.stack([example[key] for example in data],dim=0)
            else:
                return [example[key] for example in data]


if __name__ == "__main__":
    conf = TrainingConfig()
    dataset = PreLoad().dataset
    image_processing = ImageProcessing()
    dataloader = DataLoader(dataset,batch_size=conf.batch_size,collate_fn=image_processing.collate_fn)
    data = next(iter(dataloader))
    pprint(f"data ->{data['img_train'].shape,data['image'].shape}")
    display_images([data["img_train"],data["image"]],nrow=conf.batch_size)





# %%
