# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.byobnet import create_block
from timm.models.byobnet import LayerFn
from timm.layers import activations
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
import segmentation_models_pytorch as smp
import random
from timm.optim.adafactor import Adafactor

import sys
sys.path.append("../..")
sys.path.append("..")
sys.path.append(".")

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



#%%
class Encodelayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model(
            conf.encoder_name, pretrained=conf.encoder_pretrained, features_only=True
        )

    def forward(self, examples):
        features = self.encoder(examples["eval_img"])
        examples.update({"encode_feature":features[-1],"encode_features":features,"encode_feature_num":len(features)})
        return examples

    def loss(self, x, y):
        acc = F.mse_loss(x, y)
        return acc


if __name__ == "__main__":
    model = Encodelayer()
    examples = model(examples)
    pprint(examples["encode_feature"].shape)

#%%
class ChannelAttention(nn.Module):
    def __init__(self,dim,emb_dim=None,bias=False):
        super().__init__()
        self.dim = dim
        self.emb_dim = emb_dim
        self.bias = bias
        self.W = nn.Parameter(torch.zeros(dim,dim))
        self.sigmoid = nn.Sigmoid()
        self.decoder = nn.TransformerDecoderLayer(d_model=dim,nhead=8)
        self.decoderlayer = nn.TransformerDecoder(self.decoder,num_layers=1)
        self.memory = nn.Parameter(torch.randn(dim,dim))
        if emb_dim is not None:
            self.linear = nn.Linear(dim,emb_dim)
            self.linear2 = nn.Linear(emb_dim,dim)
        if bias:
            self.B = nn.Parameter(torch.randn(dim,dim))
        else:
            self.B = nn.Identity()

    def forward(self,x):
        blist = []
        for batch in range(x.shape[0]):
            ilist = []
            for channel in range(x.shape[1]):
                a = x[batch,channel,:,:]
                # a = a.view(a.shape[0],-1)
                b = self.decoder(tgt=a,memory=self.memory)
                # a = a.view(-1,self.dim,self.dim)
                # a = self.sigmoid(a)
                # if self.emb_dim is not None:
                #     a = self.linear(a)
                #     a = self.linear2(a)
                #     a = a.T
                #     a = self.linear(a)
                #     a = self.linear2(a)
                #     a = a.T
                # a = self.sigmoid(a)
                # a = a@x[b,i,:,:].T@self.W
                c = a + b
                ilist.append(c)
            ilist = torch.stack(ilist,dim=0)
            blist.append(ilist)
        blist = torch.stack(blist,dim=0)
        return blist
        

if __name__ == "__main__":
    n = torch.randn(conf.batch_size,conf.encoder_out_shape[0],conf.encoder_out_shape[1],conf.encoder_out_shape[2])
    model = ChannelAttention(dim=conf.encoder_out_shape[1])
    output = model(n)
    pprint(output.shape)

#%%
layers = LayerFn()
layers.act = activations.Swish
class ByobBlock(nn.Module):
    def __init__(self,in_chs,out_chs,layer_num=1,kernel_size=3,stride=1,dilation=(1,1),bottle_ratio=4,group_size=0,layers=layers,**kwargs):
        super().__init__()
        self.block = nn.Sequential(*[create_block("bottle",in_chs=in_chs,out_chs=in_chs,kernel_size=kernel_size,stride=stride,dilation=dilation,bottle_ratio=bottle_ratio,group_size=group_size,layers=layers) for _ in range(layer_num-1)])
        self.last_conv = create_block("bottle",in_chs=in_chs,out_chs=out_chs,kernel_size=kernel_size,stride=stride,dilation=dilation,bottle_ratio=bottle_ratio,group_size=group_size,layers=layers)
    def forward(self,x):
        return self.last_conv(self.block(x))


if __name__ == "__main__":

    block = ByobBlock(in_chs=conf.encoder_out_shape[0],out_chs=conf.encoder_out_shape[0])
    print(block)

    output = block(torch.randn(1,conf.encoder_out_shape[0],conf.encoder_out_shape[1],conf.encoder_out_shape[2]))
    pprint(output.shape)
#%%
class DiffusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = self.build_model(encoder_name="timm-efficientnet-b8",in_channels=conf.encoder_out_shape[0],num_classes=conf.encoder_out_shape[0])
        # self.scheduler = DPMSolverMultistepScheduler(num_train_timesteps=conf.num_train_timesteps)
        self.time_emb = nn.Parameter(torch.normal(0,1,size=(conf.encoder_out_shape)))
        self.optimizer = Adafactor(self.parameters(),lr=conf.learning_rate)

    def build_model(self,encoder_name,in_channels,num_classes):
        model = smp.create_model(
            arch="manet",                     # name of the architecture, e.g. 'Unet'/ 'FPN' / etc. Case INsensitive!
            encoder_name=encoder_name,
            # encoder_weights="advprop",
            encoder_weights=None,
            in_channels=in_channels,
            classes=num_classes,
            activation=None,
            )
        return model

    def time_embedding(self,timesteps):
        time_emb = self.time_emb * timesteps
        return time_emb

    def forward(self,examples):
        examples = self.train_step(examples)
        return examples

    def train_step(self,examples):
        scheduler = DPMSolverMultistepScheduler(num_train_timesteps=conf.num_train_timesteps)
        noise = torch.randn(examples["encode_feature"].shape, device=examples["encode_feature"].device)
        timesteps = torch.tensor([conf.num_train_timesteps],device=examples["encode_feature"].device).long()
        noisy_images = scheduler.add_noise(examples["encode_feature"], noise, timesteps)
        scheduler.set_timesteps(int(timesteps.item()),device=examples["encode_feature"].device)

        loss_list = []
        for t in scheduler.timesteps:
            t_emb = self.time_embedding(t)
            noise_pred = self.model(noisy_images + t_emb)
            noisy_images = scheduler.step(model_output=noise_pred, timestep=t, sample=noisy_images).prev_sample.detach()
            loss = F.mse_loss(noise_pred, noisy_images)
            if torch.isnan(loss):
                noisy_images = examples["encode_feature"]
                break
            loss_list.append(loss)

        loss_list = torch.stack(loss_list, dim=0).mean()
        if torch.isnan(loss_list):
            conf.num_train_timesteps = max(1,conf.num_train_timesteps-1)
        else:
            conf.num_train_timesteps = min(90,conf.num_train_timesteps+random.randint(0,1))
        examples.update({"diffusion_feature": noisy_images, "timesteps": conf.num_train_timesteps-t, "diffusion_loss": loss_list})
        return examples


if __name__ == "__main__":
    model = DiffusionModel()
    examples = model(examples)
    pprint(examples["timesteps"])
    pprint(examples["diffusion_loss"])

#%%
class MiddleLayer(nn.Module):
    def __init__(self,input_size=conf.encoder_out_shape,output_size=conf.decoder_in_shape,num=1):
        super().__init__()
        self.num = num
        # self.Block = ByobBlock(in_chs=input_size[0],out_chs=output_size[0],layer_num=1,kernel_size=3,stride=1,bottle_ratio=4,group_size=0)
        # self.Attn = ChannelAttention(dim=input_size[2],emb_dim=3048,bias=False)
        # self.conv = ConvNormAct(in_channels=input_size[0],out_channels=output_size[0],kernel_size=1)

        
    def forward(self, examples):
        mid_feature = examples["encode_feature"]
        # mid_feature = self.Block(mid_feature)
        # shortcut = mid_feature
        # for i in range(self.num):
        #     mid_feature = self.Attn(mid_feature)
        # mid_feature = mid_feature + shortcut
        # mid_feature = self.conv(mid_feature)
        examples.update({"middle_feature":mid_feature})
        return examples

if __name__ == "__main__":
    model = MiddleLayer()
    output = model(examples)
    pprint(output["middle_feature"].shape)


#%%
# class UpBlock(nn.Module):
#     def __init__(self, in_channel, out_channel, kernel_size=1):
#         super().__init__()

#         # self.convnormact = ConvNormAct(in_channels=in_channel//4,out_channels=out_channel,kernel_size=kernel_size)

#         self.convup = nn.Sequential(
#                 nn.PixelShuffle(2),
#                 # self.convnormact,
#             )

#     def forward(self, input):
#         outup = self.convup(input)
#         return outup

# if __name__ == "__main__":
#     a = torch.randn(1,3*4**5,8,8)
#     model = UpBlock(in_channel=3*4**5,out_channel=3*4**4)
#     b = model(a)
#     pprint(b.shape)
#%%
#%%

class DecodeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        channels = [3,3*4,3*4**2,3*4**3,3*4**4]
        self.Up = nn.PixelShuffle(2)
        self.ConvBlock = nn.ModuleList([ByobBlock(in_chs=channels[i-1],out_chs=channels[i-1],layer_num=2,kernel_size=3,stride=1,bottle_ratio=4,group_size=0) for i in reversed(range(1,len(channels)))])
        self.last_conv = ByobBlock(in_chs=channels[0],out_chs=channels[0],layer_num=4,kernel_size=3,stride=4,bottle_ratio=4,group_size=0)
        self.lastact = activations.Tanh()
        
    def forward(self, examples):
        middle_feature = examples["diffusion_feature"]
        out = middle_feature
        for ConvBlock in self.ConvBlock:
            out = self.Up(out)
            out = ConvBlock(out)
        out = self.lastact(out)

        loss = self.loss(out,examples["eval_img"])
        examples.update({"decode_feature": out,"decode_loss":loss})
        return examples

    def loss(self, x, y):
        if x.shape != y.shape:
            z = F.interpolate(x,size=y.shape[2:])
        loss = F.mse_loss(z, y)
        return loss

if __name__ == "__main__":
    model = DecodeLayer()
    output = model(examples)
    pprint(output["decode_feature"].shape)

# %%


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encodelayer()
        self.middle = MiddleLayer()
        self.diffuser = DiffusionModel()
        self.decoder = DecodeLayer()

    def forward(self, examples):
        self.check_grad()
        examples= self.encoder(examples)
        examples = self.middle(examples)
        examples = self.diffuser(examples)
        examples = self.decoder(examples)
        self.loss(examples)
        return examples

    def loss(self,examples):
        examples["diversity_loss"] = self.total_diversity_loss()*0.001
        examples["loss"] = examples["decode_loss"] + examples["diversity_loss"]
        return examples

    def total_diversity_loss(self):
        diversity_loss = 0.0
        for param in self.decoder.parameters():
            if (
                param.ndimension() == 4
            ):  # Conv2dの重み（形状: out_channels, in_channels, kH, kW）
                weight = param.view(
                    param.size(0), -1
                )  # (out_channels, in_channels * kernel_size * kernel_size)
                weight_mean = torch.mean(
                    weight, dim=1, keepdim=True
                )  # (out_channels, 1)
                weight_centered = (
                    weight - weight_mean
                )  # (out_channels, in_channels * kernel_size * kernel_size)
                covariance_matrix = torch.matmul(
                    weight_centered, weight_centered.t()
                )  # (out_channels, out_channels)
                diag = torch.diag(covariance_matrix)
                covariance_matrix = covariance_matrix - torch.diag_embed(diag)
                diversity_loss += torch.sum(
                    torch.abs(covariance_matrix)
                )  # Sum of absolute values of off-diagonal elements
        return diversity_loss

    def check_grad(self):
        for name,param in self.named_parameters():
            if param.dtype == torch.float16:
                print(name,param.dtype)
        

if __name__ == "__main__":
    model = Model()
    output = model(examples)
    pprint(output["decode_feature"].shape)

# %%
if __name__ == "__main__":
    model = Model()
    # for name,param in model.named_parameters():
    #     if param.shape == (768, 16, 3, 3):
    #         print(name)

    print(model.diffuser.model.segmentation_head[0].weight.dtype)

# %%

# %%