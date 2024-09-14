# %%
from math import e
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
from torch.utils.data import DataLoader
import sys
import lightning as L
from lightning.pytorch.callbacks import ModelSummary
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from tensorboardX import SummaryWriter
from pprint import pprint
from itertools import cycle

import multiprocessing
multiprocessing.freeze_support()

sys.path.append("../..")
sys.path.append("..")
sys.path.append(".")

from util.image_util import display_images
from util.push import push_message

from config import TrainingConfig
from image_processing import ImageProcessing
from dataset import PreLoad
from model import Model

conf = TrainingConfig()
impro = ImageProcessing()

import warnings
if not conf.debug:
    warnings.filterwarnings("ignore")

print(f"project: {conf.project_name}")
print(f"cuda: {torch.cuda.is_available()}")
# torch.set_grad_enabled(True)
# torch.set_default_dtype(torch.float16)
# torch.set_default_device(conf.device)
#%%

# %%
# define the LightningModule
class LitModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = Model()
        self.num_steps = 0
        # self.save_hyperparameters()
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        opt, diffusion_opt = self.optimizers()
        
        # モデルの計算
        output = self.forward(batch)
        loss = output["loss"]
        diffusion_loss = output["diffusion_loss"] 

        # diffusion_lossの勾配計算
        diffusion_opt.zero_grad()
        self.manual_backward(diffusion_loss, retain_graph=True)
        self.clip_gradients(diffusion_opt, gradient_clip_val=1, gradient_clip_algorithm="norm")
        diffusion_opt.step()
        self.lr_scheduler_step(self.lr_schedulers()[1], diffusion_loss)

        # lossの勾配計算
        opt.zero_grad()
        self.manual_backward(loss)
        self.clip_gradients(opt, gradient_clip_val=1, gradient_clip_algorithm="norm")
        opt.step()
        self.lr_scheduler_step(self.lr_schedulers()[0], loss)

        del output
        torch.cuda.empty_cache()
        return loss
        # opt,diffusion_opt = self.optimizers()
        # # モデルの計算
        # output = self.forward(batch)
        # self.loss = output["loss"]
        # diffusion_loss = output["diffusion_loss"] 

        # # 勾配計算
        # diffusion_opt.zero_grad()
        # self.manual_backward(diffusion_loss,retain_graph=True)
        # self.clip_gradients(diffusion_opt, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
        # diffusion_opt.step()
        # self.lr_scheduler_step(self.lr_schedulers()[1],self.loss)
        # # 勾配計算
        # opt.zero_grad()
        # self.manual_backward(self.loss)
        # self.clip_gradients(opt, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
        # opt.step()
        # self.lr_scheduler_step(self.lr_schedulers()[0],self.loss)
        # del output
        # torch.cuda.empty_cache()
        # return self.loss

    def forward(self, batch):
        
        output = self.model(batch)
        self.loss = output["loss"]
        
        
        # for key,value in output.items():
        #     if isinstance(value,torch.Tensor):
        #         self.log(key,value,prog_bar=True)
        self.log("diffusion_loss",output["diffusion_loss"],prog_bar=True)
        self.log("decode_loss",output["decode_loss"],prog_bar=True)
        self.log("diversity_loss",output["diversity_loss"],prog_bar=True)
        self.log("timesteps",output["timesteps"].float().mean(),prog_bar=True)
        self.log("loss",self.loss,prog_bar=True)
        # self.log("train_label",output["train_label"].mean(),prog_bar=True)
        # self.log("pred_label",output["pred_label"].mean(),prog_bar=True)
        # self.log("m.mean",output["middle_feature"].mean(),prog_bar=True)
        # self.log("m.std",output["middle_feature"].std(),prog_bar=True)
        self.log(f"d.mean",output["decode_feature"].mean(),prog_bar=True)
        self.log(f"t.mean",output["train_img"].mean(),prog_bar=True)
        self.log(f"d.std",output["decode_feature"].std(),prog_bar=True)
        self.log(f"t.std",output["train_img"].std(),prog_bar=True)

        
        self.num_steps += 1
        self.on_step_end(self.num_steps,output)

        return output

    def on_step_end(self,steps,output,**kwargs):
        if steps % 10 == 0:
            x = output["eval_img"].clone().to("cuda:1")
            # x = impro.de_normalize(x)
            z = output["decode_feature"].clone().to("cuda:1")
            if x.shape != z.shape:
                z = F.interpolate(z,size=x.shape[2:])
            # z = impro.de_normalize(z)
            grid = [x,z]
            display_images(grid, show_image=False,nrow=conf.batch_size,save_image=True, output_dir=conf.output_dir,output_name=f"compare_{steps%20}")
            display_images([z], show_image=False,nrow=conf.batch_size,save_image=True, output_dir=conf.output_dir,output_name=f"output_{steps%20}")

    def validation_step(self, batch, batch_idx):
        loss = self.forward(batch)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        loss = self.forward(batch)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        optimizer = create_optimizer_v2(self.model,opt='adafactor',lr=conf.learning_rate,weight_decay=0.,momentum=0.9,foreach=None,filter_bias_and_bn=True,layer_decay=None,param_group_fn=None,)
        diffusion_optimizer = create_optimizer_v2(self.model.diffuser,opt='adafactor',lr=conf.learning_rate,weight_decay=0.,momentum=0.9,foreach=None,filter_bias_and_bn=True,layer_decay=None,param_group_fn=None,)
        scheduler, num_epochs = create_scheduler_v2(optimizer=optimizer,sched="cosine",num_epochs=conf.num_epochs,decay_epochs=10,cooldown_epochs=0,patience_epochs=10,decay_rate=0.1,min_lr=conf.min_lr,warmup_lr=conf.learning_rate,warmup_epochs=0,warmup_prefix=False,noise_pct=0.67,noise_std=1.0,noise_seed=42,cycle_mul=1.0,cycle_decay=0.1,cycle_limit=1,step_on_epochs=True,updates_per_epoch=0,)
        diffusion_scheduler, num_epochs = create_scheduler_v2(optimizer=diffusion_optimizer,sched="cosine",num_epochs=conf.num_epochs,decay_epochs=10,cooldown_epochs=0,patience_epochs=10,decay_rate=0.1,min_lr=conf.min_lr,warmup_lr=conf.learning_rate,warmup_epochs=0,warmup_prefix=False,noise_pct=0.67,noise_std=1.0,noise_seed=42,cycle_mul=1.0,cycle_decay=0.1,cycle_limit=1,step_on_epochs=True,updates_per_epoch=0,)
        return [optimizer,diffusion_optimizer], [
            {"scheduler": scheduler, "interval": "step", "monitor": "loss"},
            {"scheduler": diffusion_scheduler, "interval": "step", "monitor": "diffusion_loss"}
        ]

    def lr_scheduler_step(self, scheduler, metric):
        self.log("lr",*scheduler._get_lr(self.global_step),prog_bar=True)
        scheduler.step(epoch=self.current_epoch, metric=metric)



# 1. データの準備
dataset = PreLoad().dataset
dataloader = DataLoader(dataset,batch_size=conf.batch_size,num_workers=conf.num_workers,collate_fn=impro.collate_fn)
logger = TensorBoardLogger(conf.default_root_dir, name="lightning_logs")


from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import ModelSummary
lr_monitor = LearningRateMonitor(logging_interval='step')

loss_checkpoint = ModelCheckpoint(
    dirpath=conf.ckpt_dir,
    filename=f"{conf.project_name}_loss",
    monitor="loss",
    save_last=True,
    save_top_k=1,
    mode="min",
    every_n_train_steps=100
)
trainer = L.Trainer(
    enable_checkpointing=True,
    default_root_dir=conf.default_root_dir,
    # profiler=profiler,
    # profiler="simple",
    callbacks=[loss_checkpoint,lr_monitor,ModelSummary(max_depth=1)],
    precision="16-mixed",
    max_epochs=conf.num_epochs,
    # precision="transformer-engine-float16",
    accelerator="auto",
    # accelerator="gpu",
    devices=1,
    strategy="auto",
    # gradient_clip_val=1,
    logger=logger,
    log_every_n_steps=1,
    # enable_progress_bar=True,
)
model = LitModel()
# trainer.fit(model=model, train_dataloaders=dataloader,ckpt_path=conf.ckpt_dir/"last-v3.ckpt")
trainer.fit(model=model, train_dataloaders=dataloader)
# try:
#     trainer.fit(model=model, train_dataloaders=dataloader,ckpt_path=conf.ckpt_path)
# except Exception as e:
#     print(e)
#     trainer.fit(model=model, train_dataloaders=dataloader)
# push_message(title="train section", body="done")

# Create a tuner for the trainer


# %%

# %reload_ext tensorboard
# %tensorboard --logdir=conf.default_root_dir/"lightning_logs"