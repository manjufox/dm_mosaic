# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.byobnet import create_block
from timm.models.byobnet import LayerFn
from timm.layers import activations
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler

from typing import List, Optional, Union
from timm.layers.attention2d import MultiQueryAttentionV2
from timm.layers.norm import LayerNorm2d
from torchmetrics.image import RootMeanSquaredErrorUsingSlidingWindow
import segmentation_models_pytorch as smp
import random
from timm.optim.adafactor import Adafactor
from tqdm import tqdm,trange

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

# # %%
if __name__ == "__main__":
    dataset = PreLoad().dataset
    impro = ImageProcessing()
    dataloader = DataLoader(dataset,batch_size=conf.batch_size,collate_fn=impro.collate_fn)
    examples = next(iter(dataloader))
    pprint(examples["image"].shape)



# #%%
# class Encodelayer(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.encoder = timm.create_model(
#             conf.encoder_name, pretrained=conf.encoder_pretrained, features_only=True
#         )

#     def forward(self, examples):
#         features = self.encoder(examples["encoder_input"])
#         examples.update({"encode_feature":features[-1],"encode_features":features,"encode_feature_num":len(features)})
#         return examples

#     def loss(self, x, y):
#         acc = F.mse_loss(x, y)
#         return acc


# if __name__ == "__main__":
#     model = Encodelayer()
#     examples = model(examples)
#     for i in examples["encode_features"]:
#         pprint(i.shape)

# #%%
# class ChannelAttention(nn.Module):
#     def __init__(self,dim,emb_dim=None,bias=False):
#         super().__init__()
#         self.dim = dim
#         self.emb_dim = emb_dim
#         self.bias = bias
#         self.W = nn.Parameter(torch.zeros(dim,dim))
#         self.sigmoid = nn.Sigmoid()
#         self.decoder = nn.TransformerDecoderLayer(d_model=dim,nhead=8)
#         self.decoderlayer = nn.TransformerDecoder(self.decoder,num_layers=1)
#         self.memory = nn.Parameter(torch.randn(dim,dim))
#         if emb_dim is not None:
#             self.linear = nn.Linear(dim,emb_dim)
#             self.linear2 = nn.Linear(emb_dim,dim)
#         if bias:
#             self.B = nn.Parameter(torch.randn(dim,dim))
#         else:
#             self.B = nn.Identity()

#     def forward(self,x):
#         blist = []
#         for batch in range(x.shape[0]):
#             ilist = []
#             for channel in range(x.shape[1]):
#                 a = x[batch,channel,:,:]
#                 # a = a.view(a.shape[0],-1)
#                 b = self.decoder(tgt=a,memory=self.memory)
#                 # a = a.view(-1,self.dim,self.dim)
#                 # a = self.sigmoid(a)
#                 # if self.emb_dim is not None:
#                 #     a = self.linear(a)
#                 #     a = self.linear2(a)
#                 #     a = a.T
#                 #     a = self.linear(a)
#                 #     a = self.linear2(a)
#                 #     a = a.T
#                 # a = self.sigmoid(a)
#                 # a = a@x[b,i,:,:].T@self.W
#                 c = a + b
#                 ilist.append(c)
#             ilist = torch.stack(ilist,dim=0)
#             blist.append(ilist)
#         blist = torch.stack(blist,dim=0)
#         return blist
        

# if __name__ == "__main__":
#     n = torch.randn(conf.batch_size,conf.encoder_out_shape[0],conf.encoder_out_shape[1],conf.encoder_out_shape[2])
#     model = ChannelAttention(dim=conf.encoder_out_shape[1])
#     output = model(n)
#     pprint(output.shape)

# #%%
# layers = LayerFn()
# layers.act = activations.Swish
# class ByobBlock(nn.Module):
#     def __init__(self,in_chs,out_chs,layer_num=2,kernel_size=3,stride=1,dilation=(1,1),bottle_ratio=4,group_size=0,layers=layers,**kwargs):
#         super().__init__()
#         self.block = nn.Sequential(*[create_block("bottle",in_chs=in_chs,out_chs=in_chs,kernel_size=kernel_size,stride=stride,dilation=dilation,bottle_ratio=bottle_ratio,group_size=group_size,layers=layers) for _ in range(layer_num-1)])
#         self.last_conv = create_block("bottle",in_chs=in_chs,out_chs=out_chs,kernel_size=kernel_size,stride=stride,dilation=dilation,bottle_ratio=bottle_ratio,group_size=group_size,layers=layers)
#     def forward(self,x):
#         return self.last_conv(self.block(x))


# if __name__ == "__main__":

#     block = ByobBlock(in_chs=conf.encoder_out_shape[0],out_chs=conf.encoder_out_shape[0])
#     print(block)

#     output = block(torch.randn(1,conf.encoder_out_shape[0],conf.encoder_out_shape[1],conf.encoder_out_shape[2]))
#     pprint(output.shape)

# #%%

# class MultiQueryAttention(MultiQueryAttentionV2):
    
#     def forward(self, x, m: Optional[torch.Tensor] = None):
#         """Run layer computation."""
#         s = x.shape
#         if m is None:
#             m = x

#         reshaped_x = self._reshape_input(x)
#         reshaped_m = self._reshape_input(m)

#         q = torch.einsum('bnd,hkd->bnhk', reshaped_x, self.query_proj)
#         k = torch.einsum('bmd,dk->bmk', reshaped_m, self.key_proj)

#         attn = torch.einsum('bnhk,bmk->bnhm', q, k)
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)

#         v = torch.einsum('bmd,dv->bmv', reshaped_m, self.value_proj)
#         o = torch.einsum('bnhm,bmv->bnhv', attn, v)
#         result = torch.einsum('bnhv,dhv->bnd', o, self.out_proj)
#         result = self.proj_drop(result)
#         return result.reshape(s)

# class DecodeLayer(nn.Module):
#     def __init__(self,layer_num=3):
#         super().__init__()
#         channels = [768,768,768]
#         self.layer_num = layer_num
#         self.attnBlock = nn.ModuleList([
#             nn.ModuleList([
#                 MultiQueryAttention(dim=768,dim_out=768,num_heads=8,key_dim=64,value_dim=64,attn_drop=0.,proj_drop=0.)
#                 for _ in range(layer_num)
#             ])
#             for _ in range(len(channels))
#             ])
#         self.norm = LayerNorm2d(conf.encoder_out_shape[0])
#         self.act = activations.Swish()
#         self.rmse = RootMeanSquaredErrorUsingSlidingWindow()
#         self.ConvBlock = nn.ModuleList([
#             nn.ModuleList([ 
#                 ByobBlock(in_chs=channels[i],out_chs=channels[i],layer_num=2,kernel_size=3,stride=1,bottle_ratio=1,group_size=0) 
#                 for _ in range(layer_num)
#             ])
#             for i in range(len(channels))
#             ])
#         # self.lastact = activations.Tanh()
#     def forward(self, examples):
#         middle_features = examples["encode_features"]
#         out = middle_features[-1]
#         for feature,convlist,attnlist in zip(middle_features,self.ConvBlock,self.attnBlock):
#             feature = self.norm(feature)
#             for conv,attn in zip(convlist,attnlist):
#                 for i in range(self.layer_num):
#                     out = self.norm(out)
#                     out = attn(x=out,m=feature)
#                     out = self.norm(out)
#                     out = conv(out)
#                     out = out + feature
#                     out = self.norm(out) 
            
            
#         out = torch.reshape(out,(conf.batch_size,3,512,512))
#         out = F.interpolate(out,size=conf.encoder_in_shape[1:])
#         # loss = self.loss(out,examples["train_img"])
#         loss = torch.zeros(1,device=out.device)
#         examples.update({"decode_feature": out,"decode_loss":loss})
#         return examples

#     def loss(self, x, y):
#         if x.shape != y.shape:
#             x = F.interpolate(x,size=y.shape[2:])
#         loss = self.rmse(x, y)
#         return loss

# if __name__ == "__main__":
#     model = DecodeLayer()
#     output = model(examples)
#     pprint(output["decode_feature"].shape)

# #%%
# class Model(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.encoder = Encodelayer()
#         self.decoder = DecodeLayer()

#     def forward(self, examples):
#         examples= self.encoder(examples)
#         examples = self.decoder(examples)
#         self.loss(examples)
#         return examples

#     def loss(self,examples):
#         examples["diversity_loss"] = self.total_diversity_loss()*0.001
#         examples["loss"] = examples["decode_loss"] + examples["diversity_loss"]
#         return examples

#     def total_diversity_loss(self):
#         diversity_loss = 0.0
#         for param in self.decoder.parameters():
#             if (
#                 param.ndimension() == 4
#             ):  # Conv2dの重み（形状: out_channels, in_channels, kH, kW）
#                 weight = param.view(
#                     param.size(0), -1
#                 )  # (out_channels, in_channels * kernel_size * kernel_size)
#                 weight_mean = torch.mean(
#                     weight, dim=1, keepdim=True
#                 )  # (out_channels, 1)
#                 weight_centered = (
#                     weight - weight_mean
#                 )  # (out_channels, in_channels * kernel_size * kernel_size)
#                 covariance_matrix = torch.matmul(
#                     weight_centered, weight_centered.t()
#                 )  # (out_channels, out_channels)
#                 diag = torch.diag(covariance_matrix)
#                 covariance_matrix = covariance_matrix - torch.diag_embed(diag)
#                 diversity_loss += torch.sum(
#                     torch.abs(covariance_matrix)
#                 )  # Sum of absolute values of off-diagonal elements
#         return diversity_loss

# if __name__ == "__main__":
#     model = Model()
#     output = model(examples)
#     pprint(output["decode_feature"].shape)
#%%
class DiffusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = smp.create_model(
            arch="Unetplusplus",                     # name of the architecture, e.g. 'Unet'/ 'FPN' / etc. Case INsensitive!
            # encoder_name="timm-efficientnet-b8",
            encoder_name="timm-regnety_320",
            encoder_weights="imagenet",
            in_channels=conf.encoder_in_shape[0],
            classes=conf.encoder_in_shape[0],
            decoder_use_batchnorm =True,
            decoder_attention_type ="scse",
        )
        self.time_emb = nn.Embedding(conf.num_train_timesteps,conf.encoder_in_shape[1]*conf.encoder_in_shape[2])

    def forward(self,examples):
        if self.training:
            examples = self.train_step(examples)
            if examples['steps'] % 1000 ==0:
                with torch.no_grad():
                    examples = self.validate(examples)
        else:
            with torch.no_grad():
                examples = self.validate(examples)
        return examples

    def train_step(self,examples):
        scheduler = DPMSolverMultistepScheduler(num_train_timesteps=conf.num_train_timesteps)
        noise = torch.randn(examples["train_img"].shape, device=examples["train_img"].device)
        timesteps = torch.randint(1,conf.num_train_timesteps,(1,),device=examples["train_img"].device)
        noisy_images = scheduler.add_noise(examples["train_img"], noise, timesteps)
        scheduler.set_timesteps(int(timesteps.item()),device=examples["train_img"].device)

        t_emb = self.time_emb(timesteps).view(1,1,conf.encoder_in_shape[1],conf.encoder_in_shape[2])
        t_emb = t_emb.repeat(conf.batch_size,1,1,1)
        noisy_images = noisy_images+t_emb
        noise_pred = self.model(noisy_images)
        loss = F.mse_loss(noise_pred, noise)
        examples.update({"diffusion_feature": noisy_images-noise_pred,"prev_sample":noisy_images, "timesteps": timesteps, "diffusion_loss": loss,"loss":loss})
        return examples

    def validate(self,examples):
        scheduler = DPMSolverMultistepScheduler(num_train_timesteps=conf.num_train_timesteps)
        noise = torch.randn(examples["eval_img"].shape, device=examples["eval_img"].device)
        # timesteps = torch.tensor([conf.num_train_timesteps],device=examples["eval_img"].device).long()
        timesteps = torch.randint(1,conf.num_train_timesteps,(1,),device=examples["encoder_input"].device)
        noisy_images = scheduler.add_noise(examples["eval_img"], noise, timesteps)
        scheduler.set_timesteps(int(timesteps.item()),device=examples["eval_img"].device)
        
        # noisy_images = torch.cat(noisy_images,t_emb,dim=1)
        examples["noisy_images"] = noisy_images
        examples["diffusion_feature"] = torch.zeros_like(noisy_images)
        examples["timesteps"] = 0

        for t in scheduler.timesteps:
            t_emb = self.time_emb(timesteps).view(1,1,conf.encoder_in_shape[1],conf.encoder_in_shape[2])
            t_emb = t_emb.repeat(conf.batch_size,1,1,1)
            noisy_images = noisy_images+t_emb
            noise_pred = self.model(noisy_images)
            noisy_images = scheduler.step(noise_pred,t,noisy_images)[0]
            loss = F.mse_loss(noisy_images,examples["eval_img"])
            examples["diffusion_feature"] = noisy_images
            examples["timesteps"] = timesteps - t
            examples["eval_loss"] = loss
        pprint(f'mean:{noisy_images.mean()} t:{timesteps.item()-t}, eval_loss:{loss}')
        examples["eval_img"] = impro.de_normalize(examples["eval_img"])
        examples["noisy_images"] = impro.de_normalize(examples["noisy_images"])
        examples["diffusion_feature"] = impro.de_normalize(examples["diffusion_feature"])
        display_images(examples["eval_img"],show_image=False,nrow=conf.batch_size,save_image=True,output_dir=conf.output_dir,output_name=f"eval_img")
        display_images(examples["noisy_images"],show_image=False,nrow=conf.batch_size,save_image=True,output_dir=conf.output_dir,output_name=f"noisy_images")
        display_images(examples["diffusion_feature"],show_image=False,nrow=conf.batch_size,save_image=True,output_dir=conf.output_dir,output_name=f"pred_img")
        
        # examples.update({"timesteps": timesteps-t, "diffusion_loss": loss})
        return examples


if __name__ == "__main__":
    model = DiffusionModel()
    examples = model(examples)
    pprint(examples["timesteps"])
    pprint(examples["diffusion_loss"])

#%%

# %%




# %%
if __name__ == "__main__":
    model = Model()
    # for name,param in model.named_parameters():
    #     if param.shape == (768, 16, 3, 3):
    #         print(name)

    print(model.diffuser.model.segmentation_head[0].weight.dtype)

# %%

# %%