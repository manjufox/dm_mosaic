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
from pprint import pprint

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
torch.set_grad_enabled(True)
torch.set_default_dtype(torch.float32)
torch.set_default_device(conf.device)


# %%

# %%
# define the LightningModule
class LitModel(L.LightningModule):
    def __init__(self,example):
        super().__init__()
        self.model = Model()
        self.num_steps = 0
        self.save_hyperparameters()
        # self.example_input_array = example

    def training_step(self, batch, batch_idx):
        self.loss = self.forward(batch)
        self.log("loss", self.loss, prog_bar=True)

        return self.loss

    def forward(self, batch):
        output = self.model(batch)
        self.loss = output[0]["encode_loss"] + output[0]["decode_loss"] #+ self.model.total_diversity_loss()*0.001
        self.log("encode_loss",output[0]["encode_loss"],prog_bar=True)
        self.log("decode_loss",output[0]["decode_loss"],prog_bar=True)
        self.log("diversity_loss",self.model.total_diversity_loss(),prog_bar=True)
        self.log("train_label",output[0]["train_label"].mean(),prog_bar=True)
        self.log("pred_label",output[0]["pred_label"].mean(),prog_bar=True)
        self.log(f"y.mean",batch[0]["train_img"].mean(),prog_bar=True)
        self.log(f"y.std",batch[0]["train_img"].std(),prog_bar=True)
        self.log(f"z.mean",batch[0]["decode_feature"].mean(),prog_bar=True)
        self.log(f"z.std",batch[0]["decode_feature"].std(),prog_bar=True)

        
        self.num_steps += 1

        self.on_step_end(self.num_steps,batch)
        del batch
        torch.cuda.empty_cache()
        self.batch = None
        return self.loss

    def on_step_end(self,steps,batch=None,**kwargs):
        pass
        # text = f"num_steps: {steps},loss: {self.loss.item():.3f},x_hat.mean: {self.x_hat.mean():.3f},x_hat.std: {self.x_hat.std():.3f},x.mean: {self.x.mean():.3f},x.std: {self.x.std():.3f}"
        # print(text)
        # push_message(title=f"loss: {self.loss.item():.3f}", body=text)
        
        if steps % 10 == 0:
            x = batch[0]["image"]
            y = batch[0]["train_img"]
            z = batch[0]["decode_feature"]
            grid = [y.detach().cpu(),z.detach().cpu()]
            # display_images(x, show_image=False, save_image=True, output_dir=conf.output_dir)
            # pprint(f"y.mean: {y.mean().item():.3f},z.mean: {z.mean().item():.3f},y.std: {y.std().item():.3f},z.std: {z.std().item():.3f}")
            display_images(grid, show_image=False, save_image=True, output_dir=conf.output_dir)

    def validation_step(self, batch, batch_idx):
        loss = self.forward(batch)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        loss = self.forward(batch)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        optimizer = create_optimizer_v2(
            self.model,
            opt='adafactor',
            lr=conf.learning_rate,
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
            num_epochs=conf.num_epochs,
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
        self.log("lr",scheduler.get_last_lr(),prog_bar=True)
        scheduler.step(epoch=self.current_epoch, metric=metric)



# 1. データの準備
dataset = PreLoad().dataset
dataloader = DataLoader(dataset,batch_size=conf.batch_size,num_workers=conf.num_workers,collate_fn=impro.collate_fn)
example = next(iter(dataloader))

from lightning.pytorch.callbacks import ModelSummary
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import StochasticWeightAveraging
from lightning.pytorch.loggers import TensorBoardLogger

logger = TensorBoardLogger(conf.default_root_dir, name="lightning_logs")

# checkpoint_callback = ModelCheckpoint(dirpath=conf.default_root_dir/"lightning_logs", save_top_k=2, monitor="loss")
loss_checkpoint = ModelCheckpoint(
    dirpath=conf.ckpt_dir,
    filename=f"best_loss",
    monitor="loss",
    save_last=True,
    save_top_k=1,
    save_weights_only=True,
    mode="min",
)
trainer = L.Trainer(
    enable_checkpointing=True,
    default_root_dir=conf.default_root_dir,
    profiler=conf.profiler,
    callbacks=[ModelSummary(max_depth=-1),loss_checkpoint,StochasticWeightAveraging(swa_lrs=1e-2)],
    precision="16-mixed",
    max_epochs=conf.num_epochs,
    # precision="transformer-engine-float16",
    accelerator="gpu",
    devices=1,
    strategy="auto",
    gradient_clip_val=0.5,
    logger=logger,
    log_every_n_steps=10,
    enable_model_summary=True,
    enable_progress_bar=True,
)
model = LitModel(example = example)
# model = torch.compile(model)
# push_message(title="train section", body="start")
# tuner = Tuner(trainer)
# tuner.scale_batch_size(model, mode="power")
trainer.fit(model=model, train_dataloaders=dataloader)
# push_message(title="train section", body="done")

# Create a tuner for the trainer


# %%

# %reload_ext tensorboard
# %tensorboard --logdir=conf.default_root_dir/"lightning_logs"