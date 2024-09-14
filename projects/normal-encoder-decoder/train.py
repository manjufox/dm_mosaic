# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange,Reduce
from einops import reduce
import timm
import timm.optim
import timm.data
import timm.data.transforms
import timm.layers
import timm.models
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
import timm.utils
from datasets import load_dataset
from datasets import get_dataset_config_names
from datasets import load_metric
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers import Trainer
from torchinfo import summary
import numpy as np
import sys
import functools
import os
from lomo_optim import Lomo
from lomo_optim import AdaLomo
from lightning.pytorch.callbacks import TQDMProgressBar
import lightning as L
from lightning.pytorch.strategies import DDPStrategy
from torchmetrics.image import ErrorRelativeGlobalDimensionlessSynthesis

sys.path.append("../..")
sys.path.append("..")
sys.path.append(".")

from util.debug_print import dprint
from util.image_util import display_images
from util.push import push_message
from config import TrainingConfig

config = TrainingConfig()

import warnings

if not config.debug:
    warnings.filterwarnings("ignore")

from torch.utils.tensorboard import SummaryWriter

print(torch.cuda.is_available())
torch.set_grad_enabled(True)
torch.set_default_dtype(torch.float32)
torch.set_default_device(config.device)
# %%
# 1. データの準備
# - データセットの収集と前処理
# - データの分割（訓練、検証、テスト）

mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])

train_transform = v2.Compose(
    [
        v2.PILToTensor(),
        v2.ToDtype(torch.float32),
        v2.RandomCrop(size=(config.image_size, config.image_size)),
        # RandomMask(),
        # v2.Lambda(lambda x: (x/255)),
        v2.Normalize(mean=mean.tolist(), std=std.tolist()),
    ]
)

random_transform = v2.Compose(
    [
        v2.RandomErasing(p=1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
    ]
)

eval_transform = v2.Compose(
    [
        v2.Resize(size=(config.image_size, config.image_size)),
        v2.ToDtype(torch.float32),
    ]
)
de_transform = v2.Compose(
    [
        v2.Normalize(mean=-1 * mean / std, std=1 / std),
        # v2.Lambda(lambda x: (x*255)),
        # v2.Lambda(lambda x: torch.clamp(x, 0, 255).byte()),
        # v2.Lambda(lambda x: v2.ToPILImage()(x)),
        # v2.Lambda(lambda x: x.convert("RGB")),
    ]
)


def data_collator(examples):
    dprint(f"examples: {examples[0]}")
    images = torch.stack([train_transform(item["image"]) for item in examples])
    labels = torch.stack([torch.tensor(item["label"]) for item in examples])
    return {"image": images, "label": labels}


def filter_function(x):
    return (
        x["image"].size[0] >= config.image_size
        and x["image"].size[1] >= config.image_size
    )


# @functools.cache
def create_dataset():
    configs = get_dataset_config_names(config.dataset_name)
    dprint(f"configs: {configs}")
    dataset = load_dataset(
        config.dataset_name,
        data_dir=config.data_dir,
        cache_dir=config.dataset_cache_dir,
        split=config.split,
        streaming=config.streaming,
        num_proc=config.num_proc,
    )
    # dataset = dataset.take(config.take_num)
    dprint(f"dataset_dir: {dir(dataset)}")
    dprint(f"dataset_vars: {vars(dataset)}")
    dprint(f"dataset.features: {dataset.features}")
    # dprint(f"dataset_info: {dataset.info}")
    # dprint(f"dataset_size: {dataset.dataset_size}")
    # dprint(f"dataset: {next(iter(dataset)).keys()}")
    dataset = dataset.rename_columns({"pixel_values": "image"})
    dataset = dataset.filter(
        lambda x: x["image"].size[0] >= config.image_size
        and x["image"].size[1] >= config.image_size
    )
    # dataset = dataset.map(data_collator, batched=True)
    # dataset = dataset.shuffle(seed=config.seed)
    # dataset = dataset.with_format("torch")
    return dataset


from timm.data.loader import create_loader


dataset = create_dataset()
class DataLoader(torch.utils.data.Dataset):   
    def __init__(self):        
        """do not open lmdb here!!"""    
    def open_lmdb(self):         
        self.env = lmdb.open(self.lmdb_dir, readonly=True, create=False)         
        self.txn = self.env.begin(buffers=True)    def __getitem__(self, item: int):        
            if not hasattr(self, 'txn'):            
                self.open_lmdb()       
            """        Then do anything you want with env/txn here.        """
# dataloader = create_loader(dataset,(3,config.image_size,config.image_size),batch_size=config.batch_size,use_prefetcher=True)
dataloader = DataLoader(dataset, batch_size=config.batch_size, collate_fn=data_collator, num_workers=config.num_workers)

for batch in dataloader:
    dprint(batch["image"].shape)
    dprint(batch["label"].shape)
    break

if config.debug:
    test_images = next(iter(dataloader))["image"]
    dprint(f"test_images[0].mean(): {test_images[0].mean()}")
    dprint(f"test_images[0].std(): {test_images[0].std()}")
    dprint(f"test_images[0].max(): {test_images[0].max()}")
    dprint(f"test_images[0].min(): {test_images[0].min()}")
    display_images(test_images)

    # push_message(title="dataload section", body="done")

# %%
# 2. モデルの定義
from transformers import AutoImageProcessor, Swinv2Model


if config.debug:
    # test_images = test_images.to(config.device)
    test_images = next(iter(dataloader))["image"]
    image_processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256",use_fast=True)
    model = Swinv2Model.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")

    num_patches = (model.config.image_size // model.config.patch_size) ** 2
    inputs = image_processor(images=test_images, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    last_hidden_states = outputs.last_hidden_state
    dprint(last_hidden_states.shape)
    features = last_hidden_states
    del model, image_processor, inputs, outputs
    torch.cuda.empty_cache()
#%%

# %%
class Upscaler(nn.Module):
    def __init__(self,input_dim=(64, 768)):
        super().__init__()
        c, h = input_dim
        self.rearrange = Rearrange("b (d1 d2) (c h w) -> b c (d1 h) (d2 w)",d1=8,d2=8,c=3,h=16,w=16)
        self.blocks = 6
        self.channels = [3,
                        *[max(c // (i-1), 128) for i in range(2,self.blocks)],
                        3
                        ]
        self.kernels = [3+i for i in range(self.blocks)]
        self.strides = [
                        *[2 for i in range(self.blocks-2)],
                        *[1 for i in range(2)],
                        ]
        self.padding = [max(0,i - 1) for i in self.strides]
        self.reach_dim = config.output_image_size
        self.conv = timm.layers.MixedConv2d
        self.conv_transpose = nn.ConvTranspose2d
        self.activation = nn.SiLU()
        self.norm = timm.layers.norm.LayerNorm2d
    
        Blocks = []
        for i in reversed(range(len(self.channels) - 1)):
            layer = []
            # layer.append(self.norm(self.channels[i+1]+3))
            layer.append(nn.Upsample(scale_factor=2))
            # layer.append(
            #     self.conv_transpose(
            #         in_channels=self.channels[i + 1]+3,
            #         out_channels=self.channels[i + 1]+3,
            #         kernel_size=self.kernels[i],
            #         stride=self.strides[i],
            #         padding=self.padding[i],
            #         output_padding=self.padding[i],
            #         bias=True,
            #     )
            # )
            # layer.append(self.norm(num_channels=self.channels[i + 1]))
            # layer.append(self.activation)
            layer.append(
                self.conv(
                    in_channels=self.channels[i + 1]+3,
                    out_channels=self.channels[i],
                    kernel_size=self.kernels[i],
                    stride=self.strides[i],
                    padding=self.padding[i],
                    bias=True,
                )
            )
            # layer.append(self.norm(self.channels[i]))
            layer.append(self.activation)
            Blocks.append(nn.Sequential(*layer))

        self.Blocks = nn.ModuleList(Blocks)

        self.weight_init()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x,img):
        x = self.rearrange(x)
        for block in self.Blocks:
            y = F.interpolate(img, size=x.shape[2:], mode="bilinear")
            x = torch.cat([x,y],dim=1)
            x = block(x)
            dprint(f"x.shape: {x.shape}")
        return x


if config.debug:
    model = Upscaler()
    dprint(all(p.is_cuda for p in model.parameters()))
    output = model(features,test_images)
    display_images(output)
    dprint(summary(model))
    del model, output
    torch.cuda.empty_cache()


# %%


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256",use_fast=True)
        self.model = Swinv2Model.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")

    def forward(self, x):
        inputs = self.image_processor(images=x, return_tensors="pt")

        # with torch.no_grad():
        outputs = self.model(**inputs)

        return outputs.last_hidden_state


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = Upscaler()

    def forward(self, x,img):
        return self.decoder(x,img)


class EncoderDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder().train()
        self.decoder = Decoder().train()

    def forward(self, x):
        return self.decoder(self.encoder(x),x)

    def total_diversity_loss(self):
        diversity_loss = 0.0
        for param in self.decoder.parameters():
            if param.ndimension() == 4:  # Conv2dの重み（形状: out_channels, in_channels, kH, kW）
                weight = param.view(param.size(0), -1)  # (out_channels, in_channels * kernel_size * kernel_size)
                weight_mean = torch.mean(weight, dim=1, keepdim=True)  # (out_channels, 1)
                weight_centered = weight - weight_mean  # (out_channels, in_channels * kernel_size * kernel_size)
                covariance_matrix = torch.matmul(weight_centered, weight_centered.t())  # (out_channels, out_channels)
                diag = torch.diag(covariance_matrix)
                covariance_matrix = covariance_matrix - torch.diag_embed(diag)
                diversity_loss += torch.sum(torch.abs(covariance_matrix))  # Sum of absolute values of off-diagonal elements
        return diversity_loss


# %%
# define the LightningModule
class LitAutoEncoder(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = EncoderDecoder()
        self.x = None
        self.x_hat = None
        self.loss = None
        self.metrics = nn.MSELoss()
        self.num_steps = 0

    def loss_fn(self, pred, target):
        return self.metrics(pred, target)

    def resize_output(self, output):
        return F.interpolate(
            output,
            size=(config.output_image_size, config.output_image_size),
            mode="bilinear",
        )

    def on_step_end(self,steps, **kwargs):
        text = f"num_steps: {steps},loss: {self.loss.item():.3f},x_hat.mean: {self.x_hat.mean():.3f},x_hat.std: {self.x_hat.std():.3f},x.mean: {self.x.mean():.3f},x.std: {self.x.std():.3f}"
        print(text)
        push_message(title=f"loss: {self.loss.item():.3f}", body=text)
        
        if steps % 5 == 0:
            self.x, self.x_hat = de_transform(self.x), de_transform(self.x_hat)
            grid = torch.cat([self.x_hat.detach().cpu(), self.x.detach().cpu()])
            display_images(grid, show_image=False, save_image=True, output_dir=config.output_dir)

    def training_step(self, batch, batch_idx):
        self.loss = self._get_reconstruction_loss(batch)
        self.log("loss", self.loss, prog_bar=True)

        return self.loss

    def _get_reconstruction_loss(self, batch):
        self.x = batch["image"]
        self.x = random_transform(self.x)
        self.x_hat = self.model(self.x)
        self.x, self.x_hat = eval_transform(batch["image"]), eval_transform(self.x_hat)
        self.loss = self.loss_fn(self.x_hat, self.x)
        self.loss += self.model.total_diversity_loss()*0.01
        self.num_steps += 1

        self.on_step_end(self.num_steps)
        del self.x, self.x_hat
        torch.cuda.empty_cache()

        return self.loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        optimizer = create_optimizer_v2(
            self.model,
            opt='adafactor',
            lr=config.learning_rate,
            weight_decay=0.,
            momentum=0.9,
            foreach=None,
            filter_bias_and_bn=True,
            layer_decay=None,
            param_group_fn=None,
        )

        scheduler, num_epochs = create_scheduler_v2(
            optimizer=optimizer,
            sched="cosine",
            num_epochs=config.num_epochs,
            decay_epochs=10,
            cooldown_epochs=0,
            patience_epochs=10,
            decay_rate=0.1,
            min_lr=0,
            warmup_lr=1e-5,
            warmup_epochs=0,
            warmup_prefix=False,
            noise_pct=0.67,
            noise_std=1.0,
            noise_seed=42,
            cycle_mul=1.0,
            cycle_decay=0.1,
            cycle_limit=1,
            k_decay=1.0,
            plateau_mode="max",
            step_on_epochs=True,
            updates_per_epoch=0,
        )

        return [optimizer], [
            {"scheduler": scheduler, "interval": "epoch", "monitor": "loss"}
        ]

    def lr_scheduler_step(self, scheduler, metric):
        # scheduler.step_update(start_epoch * updates_per_epoch)
        scheduler.step(epoch=self.current_epoch, metric=metric)


# %%


# init the autoencoder
autoencoder = LitAutoEncoder()
train_loader = dataloader
trainer = L.Trainer(
    enable_checkpointing=True,
    default_root_dir=config.default_root_dir,
    # limit_train_batches=config.batch_size,
    # max_epochs=config.num_epochs,
    profiler=config.profiler,
    # callbacks=[TQDMProgressBar()],
    precision=16,
    accelerator="gpu",
    devices=1,
)


push_message(title="train section", body="start")
trainer.fit(model=autoencoder, train_dataloaders=train_loader)
push_message(title="train section", body="done")


# %%
# # %%
# if config.debug:
#     metrics = ErrorRelativeGlobalDimensionlessSynthesis()
#     preds = torch.rand([16, 1, 16, 16], generator=torch.manual_seed(42))
#     target = preds * 1
#     dprint(metrics(preds, target))

# # %%


# class CustomTrainer(Trainer):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def _remove_unused_columns(
#         self, dataset: "datasets.Dataset", description: Optional[str] = None
#     ):
#         return dataset

# from datasets import load_metric

# accuracy = load_metric('accuracy',trust_remote_code=True)

# def compute_metrics(eval_pred):
#     """Computes accuracy on a batch of predictions"""
#     predictions = np.argmax(eval_pred.predictions, axis=1)
#     return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)

# # %%

# model = EncoderDecoder()
# print(summary(model))

# trainer = CustomTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset,
#     eval_dataset=dataset,
#     # data_collator=data_collator,
#     compute_metrics=compute_metrics,
#     # callbacks=[Callback()],
# )

# trainer.train()

# %%
# # %%
# from accelerate import Accelerator
# from huggingface_hub import create_repo, upload_folder
# from tqdm.auto import tqdm
# from pathlib import Path
# import os

# from diffusers.utils import make_image_grid


# def train_loop(config, model, optimizer, train_dataloader, lr_scheduler):
#     # Initialize accelerator and tensorboard logging
#     accelerator = Accelerator(
#         mixed_precision=config.mixed_precision,
#         gradient_accumulation_steps=config.gradient_accumulation_steps,
#         log_with="tensorboard",
#         project_dir=os.path.join(config.output_dir, "logs"),
#     )
#     if accelerator.is_main_process:
#         if config.output_dir is not None:
#             os.makedirs(config.output_dir, exist_ok=True)
#         if config.push_to_hub:
#             repo_id = create_repo(
#                 repo_id=config.hub_model_id or Path(config.output_dir).name,
#                 exist_ok=True,
#             ).repo_id
#         accelerator.init_trackers("train_example")

#     # Prepare everything
#     # There is no specific order to remember, you just need to unpack the
#     # objects in the same order you gave them to the prepare method.
#     model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
#         model, optimizer, train_dataloader, lr_scheduler
#     )

#     global_step = 0

#     # Now you train the model
#     for epoch in range(config.num_epochs):
#         progress_bar = tqdm(
#             total=len(train_dataloader), disable=not accelerator.is_local_main_process
#         )
#         progress_bar.set_description(f"Epoch {epoch}")

#         for step, batch in enumerate(train_dataloader):
#             clean_images = batch["train_images"]
#             clean_images = clean_images.to(accelerator.device)
#             eval_images = v2.Resize(
#                 size=(config.output_image_size, config.output_image_size)
#             )(clean_images)
#             eval_images = eval_images.to(accelerator.device)

#             with accelerator.accumulate(model):
#                 # Predict the noise residual
#                 pred = model(clean_images)
#                 pred = (pred + 1) / 2 * 255
#                 dprint(f"pred.shape: {pred}")
#                 dprint(f"pred.mean(): {pred.mean()}")
#                 dprint(f"pred.std(): {pred.std()}")
#                 dprint(f"pred.max(): {pred.max()}")
#                 dprint(f"pred.min(): {pred.min()}")

#                 loss = ergas(pred, eval_images)
#                 # loss = F.mse_loss(pred, eval_images)
#                 accelerator.backward(loss)

#                 accelerator.clip_grad_norm_(model.parameters(), 1)
#                 optimizer.step()
#                 lr_scheduler.step()
#                 optimizer.zero_grad()

#             progress_bar.update(1)
#             logs = {
#                 "loss": loss.detach().item(),
#                 "lr": lr_scheduler.get_last_lr()[0],
#                 "step": global_step,
#             }
#             progress_bar.set_postfix(**logs)
#             accelerator.log(logs, step=global_step)
#             global_step += 1

#             test_dir = os.path.join(config.output_dir, "samples")
#             os.makedirs(test_dir, exist_ok=True)
#             grid = torch.cat([pred, eval_images], dim=0)
#             dprint(f"grid.shape: {grid.shape}")
#             # テンソルをPIL Imageに変換
#             grid_tensor = make_grid(grid, nrow=config.train_batch_size, normalize=True)
#             grid_image = ToPILImage()(grid_tensor)
#             grid_image.save(f"{test_dir}/{step:04d}_{epoch:04d}.png")
#             # display_images(grid_image)
#             # image_grid = make_image_grid(grid, rows=2, cols=2)
#             # image_grid.save(f"{test_dir}/{epoch:04d}.png")

#         # After each epoch you optionally sample some demo images with evaluate() and save the model
#         if accelerator.is_main_process:
#             if (
#                 epoch + 1
#             ) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
#                 # Save the images
#                 test_dir = os.path.join(config.output_dir, "samples")
#                 os.makedirs(test_dir, exist_ok=True)
#                 image_grid = make_image_grid([pred, eval_images], rows=4, cols=4)
#                 image_grid.save(f"{test_dir}/{epoch:04d}.png")
#                 display_images(image_grid)

#             if (
#                 epoch + 1
#             ) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
#                 if config.push_to_hub:
#                     upload_folder(
#                         repo_id=repo_id,
#                         folder_path=config.output_dir,
#                         commit_message=f"Epoch {epoch}",
#                         ignore_patterns=["step_*", "epoch_*"],
#                     )
#                 else:
#                     model.save_pretrained(config.output_dir)


# # %%
# model = EncoderDecoder()
# # print(summary(model,input_size=(config.train_batch_size,config.channel_size,config.image_size,config.image_size)))

# optimizer = timm.optim.create_optimizer_v2(
#     model_or_params=model.parameters(),
#     opt="adamw",
#     lr=config.learning_rate,
#     weight_decay=0.01,
#     momentum=0.9,
# )

# from diffusers.optimization import get_cosine_schedule_with_warmup

# lr_scheduler = get_cosine_schedule_with_warmup(
#     optimizer=optimizer,
#     num_warmup_steps=config.lr_warmup_steps,
#     num_training_steps=(len(dataloader) * config.num_epochs),
# )
# # with torch.autograd.set_detect_anomaly(True):
# train_loop(config, model, optimizer, dataloader, lr_scheduler)

# %%
