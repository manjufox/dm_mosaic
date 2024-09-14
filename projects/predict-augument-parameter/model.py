# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torchvision.transforms import v2
import timm
import timm.optim
import timm.data
import timm.data.transforms
import timm.layers
import timm.models
import torchmetrics
from torchmetrics.image import RootMeanSquaredErrorUsingSlidingWindow
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
# if conf.debug:
#     dataset = PreLoad().dataset
#     impro = ImageProcessing()
#     dataloader = DataLoader(dataset,batch_size=2,collate_fn=impro.collate_fn)
#     examples = next(iter(dataloader))

#%%

class Encodelayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model(
            conf.encoder_name, pretrained=conf.encoder_pretrained, num_classes=conf.num_classes
        )
        self.encoder.head = nn.Linear(self.encoder.head.in_features, conf.num_classes)

    def forward(self, examples):
        img = impro.to_batch(examples,"train_img")
        train_label = impro.to_batch(examples,"train_label")
        feature = self.encoder.forward_features(img)
        pred_label = self.encoder.forward_head(feature)
        loss = self.loss(pred_label, train_label)

        for i,example in enumerate(examples):
            example.update(
                {"out_feature": feature[i], "pred_label": pred_label[i], "encode_loss": loss}
        )
        return examples,feature

    def loss(self, x, y):
        # acc = torchmetrics.functional.classification.multiclass_accuracy(
        #     preds=x, target=y, num_classes=conf.num_classes
        # )
        acc = F.mse_loss(x, y)
        return acc


# if conf.debug:
#     model = Encodelayer()
#     outputs = model(examples)
#     pprint(outputs[0]["pred_label"].shape)
#     pprint(outputs[0]["out_feature"].shape)
#     pprint(outputs[0]["encode_loss"])
    # del encoder, example, outputs,conf
    # torch.cuda.empty_cache()


# %%
class MiddleLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(conf.num_classes, 512)
        self.norm = nn.LayerNorm(512)
        self.resize = v2.Resize(size=(512,512))

    def forward(self, examples,feature):
        # img = impro.to_batch(examples,"train_img")
        # feature = impro.to_batch(examples,"out_feature")
        feature = torch.reshape(feature[:,:-1,:],(conf.batch_size,3,512,512))
        # img = self.resize(img)
        # out = torch.cat([img,feature],dim=1)
        # pred_label = impro.to_batch(examples,"pred_label")
        # true_label = impro.to_batch(examples,"true_label")
        # out = torch.zeros_like(feature)
        # pred_label_embedding = self.embedding(pred_label)
        # true_label_embedding = self.embedding(true_label)
        # p = torch.einsum("ij,ik->ijk", pred_label_embedding, true_label_embedding)
        # p = p.unsqueeze(1)
        # out = img + p
        # for i,example in enumerate(examples):
        #     example.update({"middle_feature": feature[i]})
        return examples,feature


# if conf.debug:
#     dataset = PreLoad().dataset
#     impro = ImageProcessing()
#     dataloader = DataLoader(dataset,batch_size=conf.batch_size,collate_fn=impro.collate_fn)
#     examples = next(iter(dataloader))
#     for example in examples:
#         example["out_feature"] = torch.randn(1024,768)
#     model = MiddleLayer()
#     output = model(examples)
#     pprint(output[0]["middle_feature"].shape)
# %%
#%%

class DecodeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # self.pixelshuffle = nn.PixelShuffle(2)
        # self.unshuffle = nn.PixelUnshuffle(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = timm.layers.MixedConv2d
        self.norm = timm.layers.norm.LayerNorm2d
        self.activation = 
        self.lastconv = timm.layers.MixedConv2d(
            in_channels=3*2**5, out_channels=3, kernel_size=7, stride=2, padding=1
        )
        self.lastact = nn.Sigmoid()
        self.loss = RootMeanSquaredErrorUsingSlidingWindow()

        block_num = 5
        kernels = [i * 2 - 1 for i in range(3, block_num + 3)]
        # in_channels = [6,8,16,32,64,64,64,64,64,64]
        channels = [3*2**i for i in range(0,block_num+2)]
        stride = [2,1,1,1,1,1,1,1,1,1,1,1]
        self.blocks = nn.ModuleList()
        for i in range(block_num):
            self.blocks.append(nn.Sequential(
                    # self.norm(channels[i]),
                    self.conv(in_channels=channels[i],out_channels=channels[i+1],kernel_size=kernels[i],
                              stride=stride[i],padding=str(kernels[i] // 2)),
                    # self.activation,
                    self.conv(in_channels=channels[i+1],out_channels=channels[i+1],kernel_size=kernels[i],stride=1,padding=str(kernels[i] // 2),bias=True),
                    self.activation,
                )
            )
        self.init_weights()

    def init_weights(self):
        for param in self.parameters():
            if param.ndimension() == 4:
                nn.init.normal_(param,0,0.1)

                
    def forward(self, examples,feature):

        m = feature
        # m = torch.randint(0,255,(conf.batch_size,6,512,512)).float()
        m = self.upsample(m)
        for block in self.blocks:
            
            m = block(m)
            # print(m.shape)
            # m = self.pixelshuffle(m)
            # print(m.shape)
        m = self.lastconv(m)
        m = self.lastact(m)
        # m = self.unshuffle(m)
        # print(m.shape)
        m = F.interpolate(m,size=(conf.input_image_size,conf.input_image_size))
        # print(m.shape)
        eval_img = torch.stack([example["eval_img"] for example in examples])
        loss = self.loss(m,eval_img)
        for i,example in enumerate(examples):
            example.update({"decode_feature": m[i],"decode_loss":loss})
        return examples

    def loss(self, x, y):
        loss = F.mse_loss(x, y)
        return loss




# if conf.debug:
#     dataset = PreLoad().dataset
#     impro = ImageProcessing()
#     dataloader = DataLoader(dataset,batch_size=conf.batch_size,collate_fn=impro.collate_fn)
#     examples = next(iter(dataloader))
#     # encoder = Encodelayer()
#     # examples = encoder(examples)
#     # middle = MiddleLayer()
#     # examples = middle(examples)
#     model = DecodeLayer()
#     output = model(examples)
#     pprint(output[0]["decode_feature"].shape)

# %%


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encodelayer().eval()
        self.middle = MiddleLayer()
        self.decoder = DecodeLayer()

    def forward(self, examples):
        examples,feature = self.encoder(examples)
        examples,feature = self.middle(examples,feature)
        examples = self.decoder(examples,feature)
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

if __name__ == "__main__":
    dataset = PreLoad().dataset
    impro = ImageProcessing()
    dataloader = DataLoader(dataset,batch_size=conf.batch_size,collate_fn=impro.collate_fn)
    examples = next(iter(dataloader))
    model = Model()
    output = model(examples)
    pprint(output[0]["decode_feature"].shape)
# if conf.debug:
#     model = Model()
#     output = model(examples)
#     pprint(output[0]["decode_feature"].shape)
#     del model, example, output
#     torch.cuda.empty_cache()

# %%
