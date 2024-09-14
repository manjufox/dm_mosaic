# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.tv_tensors import Image
from torchvision.transforms import v2
import timm
from timm.models.byobnet import create_block
from timm.models.byobnet import LayerFn
from timm.layers import activations
import torchvision
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from diffusers.utils import make_image_grid
import numpy as np
import math

from typing import List, Optional, Union
# from timm.layers.attention2d import MultiQueryAttentionV2
# from timm.layers.norm import LayerNorm2d
# from torchmetrics.image import RootMeanSquaredErrorUsingSlidingWindow
import segmentation_models_pytorch as smp
import sys
sys.path.append("../..")
sys.path.append("..")
sys.path.append(".")
from util.image_util import display_images

from config import TrainingConfig
from pprint import pprint
from image_processing import ImageProcessing
from dataset import PreLoad
from torch.utils.data import DataLoader

conf = TrainingConfig()
impro = ImageProcessing()


if __name__ == "__main__":
    dataset = PreLoad().dataset
    impro = ImageProcessing()
    dataloader = DataLoader(dataset,batch_size=conf.batch_size,collate_fn=impro.collate_fn)
    examples = next(iter(dataloader))
    pprint(examples["image"].shape)
#%%
class TimestepEmbedding(nn.Module):
    def __init__(self,num_emb,act="silu"):
        super().__init__()
        self.linear = nn.Linear(1,num_emb)
        self.act = nn.SiLU() if act == "silu" else nn.Identity()

    def forward(self,timesteps):
        emb = self.act(self.linear(timesteps))
        emb = emb.view(-1,3,conf.image_size,conf.image_size)
        return emb

    # def time_embedding(y,timesteps=0,scale=1000):
    #     b,p,h,w = y.shape
    #     timesteps = timesteps if isinstance(timesteps,torch.Tensor) else torch.ones(b,p,h,w,device=y.device).float() * timesteps.item()
    #     x = timesteps.float()
    #     x = torch.exp(1/(x+scale+torch.cos(x)+torch.sin(x))) - 1
    #     out = y * x
    #     return out

if __name__ == "__main__":
    x = torch.randn((conf.batch_size,3,conf.image_size,conf.image_size))
    timesteps = torch.randint(0,conf.num_train_timesteps,(conf.batch_size,1)).float()
    emb = TimestepEmbedding(num_emb=math.prod([x.shape[i] for i in range(1,4)]))(timesteps)
    pprint(emb.shape)
#%%
class ByobBlock(nn.Module):
    def __init__(self,name,in_chs,out_chs,layer_num:int=2,kernel_size:int=3,stride:int=1,dilation=(1,1),bottle_ratio=4,group_size:int=0,**kwargs):
        super().__init__()
        
        layers = LayerFn()
        layers.norm_act = nn.Identity
        layers.act = activations.Swish
        if layer_num == 1:
            self.block = nn.Sequential(*[create_block(name,in_chs=in_chs,out_chs=out_chs,kernel_size=kernel_size,stride=stride,dilation=dilation,bottle_ratio=bottle_ratio,group_size=group_size,layers=layers) for _ in range(layer_num-1)])
            self.last_conv = nn.Identity()
        else:
            self.block = nn.Sequential(*[create_block(name,in_chs=in_chs,out_chs=in_chs,kernel_size=kernel_size,stride=stride,dilation=dilation,bottle_ratio=bottle_ratio,group_size=group_size,layers=layers) for _ in range(layer_num)])
            self.last_conv = create_block(name,in_chs=in_chs,out_chs=out_chs,kernel_size=kernel_size,stride=stride,dilation=dilation,bottle_ratio=bottle_ratio,group_size=group_size,layers=layers)
    def forward(self,x):
        return self.last_conv(self.block(x))
    

#%%

class CustomDecoder(nn.Module):
    def __init__(self,in_channels:int,out_channels:int,block_num:int,layer_num:int):
        super().__init__()
        self.blocks = nn.ModuleList([
            ByobBlock(name="bottle",in_chs=in_channels,out_chs=out_channels,layer_num=layer_num,kernel_size=3,stride=1,dilation=(1,1),bottle_ratio=1,group_size=0) 
            for _ in range(block_num+1)])
        self.batchnorm = nn.BatchNorm2d(out_channels)
        
    def forward(self,features,timesteps):
        # features = [patch_embedding(features[i]) for i in range(len(features))]
        features = [time_embedding(features[i],timesteps=timesteps) for i in range(len(features))]
        # d1_features = [self.blocks[i](features[i]) for i in range(len(features))]
        # # d2_features = [self.blocks2[i](d1_features[i]) for i in range(len(features))]
        
        # out = [features[i]@d1_features[i] + features[i] for i in range(len(features))]
        # out = sum(out)
        out = features[-1]
        # out = out.view(out.shape[0],3,512,512)
        for i in reversed(range(len(features))):
            out = self.blocks[i](out)  + features[i] 
            # out = self.batchnorm(out)
        out = self.blocks[0](out) 
        # out = self.batchnorm(out) 
        # out = F.tanh(out)
        return out
#%%
from diffusers import UNet2DModel
from diffusers.models.unets.unet_2d import UNet2DOutput
class CustomUnet(UNet2DModel):
    def __init__(self):
        super().__init__()
        self.config.sample_size = conf.image_size
        self.encoder = timm.create_model(conf.encoder_name,pretrained=True,features_only=True)
        self.decoder = CustomDecoder(in_channels=768,out_channels=768,block_num=3,layer_num=3)

    def forward(self,x,timesteps,class_labels=None,return_dict=True):
        x = v2.Resize(conf.encoder_in_shape[2:])(x)
        features = self.encoder(x)
        out = self.decoder(features,timesteps)
        out = out.view(out.shape[0],3,512,512)
        out = v2.Resize(conf.image_size)(out)

        if return_dict:
            return UNet2DOutput(sample=out)
        else:
            return (out,)



if __name__ == "__main__":
    model = CustomUnet()
    # print(model)
    x = torch.randn(conf.encoder_in_shape)
    out = model(x,timesteps=1)[0]
    pprint(out.shape)

#%%
from diffusers import UNet2DModel
from diffusers.models.unets.unet_2d import UNet2DOutput
from diffusers.models.embedding import TimestepEmbedding
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from decoder import UnetDecoder
from transformers import AutoImageProcessor, Swin2SRModel

class SMPNet(UNet2DModel):
    def __init__(self):
        super().__init__()
        self.timestepembedding = TimestepEmbedding(num_emb=3*conf.image_size*conf.image_size)
        self.config.sample_size = conf.image_size
        aux_params=dict(
            pooling = "avg",
            activation = None,
        )
        self.model = smp.Unet(
            encoder_name = 'mit_b5', 
            encoder_weights = 'imagenet', #None, #'imagenet', 
            in_channels=6,
            classes=3,
            activation = nn.SiLU,
            aux_params = None)
        # self.model.decoder = UnetDecoder(
        #     encoder_channels=self.model.encoder.out_channels,
        #     decoder_channels= [2048,1024,512,256,16],
        #     n_blocks=5,
        #     use_batchnorm=False,
        #     center=True,
        #     attention_type="scse",
        # )
        self.norm = nn.BatchNorm2d(3)

    def forward(self,x,timesteps,class_labels=None,return_dict=True):
        timesteps = self.timestepembedding(timesteps)
        x = torch.cat([x,timesteps],dim=1)
        out = self.model(x)

        if return_dict:
            return UNet2DOutput(sample=out)
        else:
            return (out,)

if __name__ == "__main__":
    model = SMPNet()
    # print(model)
    x = torch.randn(conf.encoder_in_shape)
    out = model(x,timesteps=1)[0]
    pprint(out.shape)

#%%
# from transformers import AutoImageProcessor, Swin2SRModel,Swin2SRConfig
# class Swin2SR(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.timestepembedding = TimestepEmbedding(num_emb=3*conf.image_size*conf.image_size)
#         # self.processor = AutoImageProcessor.from_pretrained("caidas/swin2SR-classical-sr-x2-64")
#         config = Swin2SRConfig(upscale=1,image_size=conf.image_size)
#         self.model = Swin2SRModel(config)#.from_pretrained("caidas/swin2SR-classical-sr-x2-64")
#         # self.model = Swin2SRModel.from_pretrained("caidas/swin2SR-classical-sr-x2-64")
        
#     def forward(self,x,timesteps,class_labels=None,return_dict=True):
#         timesteps = self.timestepembedding(timesteps)
#         x = torch.cat([x,timesteps],dim=1)

#         out = self.model(x)

#         if return_dict:
#             return UNet2DOutput(sample=out)
#         else:
#             return (out,)
#%%
from diffusers import ConsistencyModelPipeline
from diffusers import EulerAncestralDiscreteScheduler,DDPMScheduler
from diffusers import CMStochasticIterativeScheduler
from diffusers import UNet2DModel
from diffusers import DDIMPipeline
from diffusers import DDIMScheduler

class DiffusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # self.model = UNet2DModel(
        #     sample_size=conf.image_size,  # the target image resolution
        #     in_channels=3,  # the number of input channels, 3 for RGB images
        #     out_channels=3,  # the number of output channels
        #     layers_per_block=2,  # how many ResNet layers to use per UNet block
        #     block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
        #     down_block_types=(
        #         "AttnDownBlock2D", # a regular ResNet downsampling block
        #         "AttnDownBlock2D",
        #         # "AttnDownBlock2D",
        #         "AttnDownBlock2D",
        #         "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        #         "AttnDownBlock2D",
        #         "DownBlock2D",
        #     ),
        #     up_block_types=(
        #         "UpBlock2D",  # a regular ResNet upsampling block
        #         # "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        #         "AttnUpBlock2D", 
        #         "AttnUpBlock2D", 
        #         "AttnUpBlock2D", 
        #         "AttnUpBlock2D", 
        #         "AttnUpBlock2D", 
        #     ),
        # )
        self.model = SMPNet()

        # self.model = Swin2SR()
        self.scheduler = DDIMScheduler(num_train_timesteps=40)
        # self.scheduler = DDPMScheduler()
        self.pipeline = DDIMPipeline(unet=self.model,scheduler=self.scheduler)
        # self.pipeline = StableDiffusionImg2ImgPipeline(unet=self.model, scheduler=self.scheduler)

    def forward(self,examples):
        if self.training:
            examples = self.train_step(examples)
            if examples['steps'] is not None and examples['steps'] % conf.evaluate_each_n_steps ==0:
                with torch.no_grad():
                    examples = self.validate(examples)
        else:
            with torch.no_grad():
                examples = self.validate(examples)
        return examples

    def train_step(self,examples):
        clean_images = examples["img_train"]
        # Sample noise to add to the images
        noise = torch.randn(clean_images.shape,device=clean_images.device)      
        bs = clean_images.shape[0]
        # noise = torch.randn(examples["img_train"].shape,device=examples["img_train"].device)
        # timestep = np.random.choice(self.scheduler.timesteps,1)
        # timesteps = torch.from_numpy(timestep).to(clean_images.device)
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (1,), device=clean_images.device,dtype=torch.int64)
        # timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (bs,), device=clean_images.device,dtype=torch.int64)
        noisy_images = self.scheduler.add_noise(clean_images, noise, timesteps)
        noisy_images = self.scheduler.scale_model_input(noisy_images, timesteps)
        noise_pred = self.model(noisy_images, timesteps, return_dict=False)[0]
        loss = F.mse_loss(noise_pred, noise)
        examples.update({"noise_pred":noise_pred,"noisy_images":noisy_images, "timesteps": timesteps,"loss":loss})
        return examples
        # noise = torch.randn(examples["img_train"].shape,device=examples["img_train"].device)
        # timesteps = np.random.choice(self.scheduler.timesteps,1)
        # timesteps = torch.tensor(timesteps,device=examples["img_train"].device)
        # noisy_images = self.scheduler.add_noise(examples["img_train"], noise, timesteps).to(examples["img_train"].device)
        # scaled_input = self.scheduler.scale_model_input(noisy_images, timesteps[0]).to(examples["img_train"].device)
        # noise_pred = self.model(scaled_input, timesteps[0], return_dict=False)[0]
        # loss = F.mse_loss(noise_pred, noise)
        # loss_eval = F.mse_loss(noisy_images-noise_pred,examples["img_eval"])
        # examples.update({"feature_diffusion": noisy_images-noise_pred,"noisy_images":noisy_images, "timesteps": timesteps,"loss":loss,"loss_diffusion": loss,"loss_eval":loss_eval})
        # return examples

    def validate(self,examples):
        # latents = torch.randn(1,3,conf.image_size,conf.image_size,device=examples["img_eval"].device) * torch.randn((1,),device=examples["img_eval"].device) * torch.randint(-10,10,(1,),device=examples["img_eval"].device) + examples["img_eval"][0]
        images = self.pipeline(batch_size=1,generator=torch.Generator().manual_seed(conf.seed),num_inference_steps=20).images
        # images  = images[0]
        # images = impro.to_eval(images).unsqueeze(0)
        # display_images([images],show_image=False,nrow=conf.batch_size,save_image=True,output_dir=conf.output_dir,output_name="img_pred")
        # images = images.squeeze(0).detach().cpu().numpy()
        # latents = latents.squeeze(0).detach().cpu().numpy()
        image_grid = make_image_grid(images, rows=len(images), cols=1)
        image_grid = torchvision.tv_tensors.Image(image_grid)
        examples["logger"].add_image(f"img_pred/{examples['steps']%conf.save_image_each_n_steps}", image_grid, examples['steps'])
        # examples["logger"].add_image(f"img_eval/{examples['steps']%conf.save_image_each_n_steps}", latents, examples['steps'])
        
        return examples


if __name__ == "__main__":
    model = DiffusionModel()
    examples["steps"] = 0
    examples = model(examples)
    pprint(examples["timesteps"])
    pprint(examples["loss_diffusion"])

    # %%
    import requests
    from PIL import Image
    from io import BytesIO
    from diffusers import StableDiffusionUpscalePipeline
    import torch

    # load model and scheduler
    model_id = "stabilityai/stable-diffusion-x4-upscaler"
    pipeline = StableDiffusionUpscalePipeline.from_pretrained(
        model_id, variant="fp16", torch_dtype=torch.float16
    )

    pipeline = pipeline.to("cuda")

    # let's download an  image
    url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd2-upscale/low_res_cat.png"
    response = requests.get(url)
    low_res_img = Image.open(BytesIO(response.content)).convert("RGB")
    low_res_img = low_res_img.resize((128, 128))
    prompt = "a white cat"

    upscaled_image = pipeline(prompt=prompt, image=low_res_img).images[0]
    upscaled_image.save(conf.output_dir/"upsampled_cat.png")
    low_res_img.save(conf.output_dir/"low_res_cat.png")
# %%
    low_res_img.save(str(conf.output_dir/"low_res_cat.png"))
# %%
