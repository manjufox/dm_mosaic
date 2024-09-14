# %%
import torch
import torch.nn.functional as F
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
from torch.utils.data import DataLoader
import sys
import lightning as L
from lightning.pytorch.callbacks import ModelSummary
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

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
from model import DiffusionModel

conf = TrainingConfig()
impro = ImageProcessing()

import warnings
if not conf.debug:
    warnings.filterwarnings("ignore")

print(f"project: {conf.project_name}")
print(f"cuda: {torch.cuda.is_available()}")
# torch.set_grad_enabled(True)
# torch.set_default_dtype(torch.float16)
# define the LightningModule

class LitModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = DiffusionModel()
        self.num_steps = 0
        self.save_hyperparameters()
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        batch["steps"] = self.global_step
        diffusion_opt = self.optimizers()
        # モデルの計算
        output = self.forward(batch)
        loss = output["loss"]
        diffusion_loss = output["diffusion_loss"] #+ output["diversity_loss"]

        # diffusion_lossの勾配計算
        diffusion_opt.zero_grad()
        self.manual_backward(diffusion_loss)
        self.clip_gradients(diffusion_opt, gradient_clip_val=1, gradient_clip_algorithm="norm")
        diffusion_opt.step()
        self.lr_scheduler_step(self.lr_schedulers(), diffusion_loss)


        self.log("diffusion_loss",output["diffusion_loss"],prog_bar=True)
        # self.log("decode_loss",output["decode_loss"],prog_bar=True)
        # self.log("diversity_loss",output["diversity_loss"],prog_bar=True)
        self.log("timesteps",float(output["timesteps"]),prog_bar=True)
        # self.log("loss",self.loss,prog_bar=True)
        # self.log("train_label",output["train_label"].mean(),prog_bar=True)
        # self.log("pred_label",output["pred_label"].mean(),prog_bar=True)
        # self.log("m.mean",output["middle_feature"].mean(),prog_bar=True)
        # self.log("m.std",output["middle_feature"].std(),prog_bar=True)
        # self.log(f"d.mean",output["decode_feature"].mean(),prog_bar=True)
        # self.log(f"t.mean",output["train_img"].mean(),prog_bar=True)
        # self.log(f"d.std",output["decode_feature"].std(),prog_bar=True)
        # self.log(f"t.std",output["train_img"].std(),prog_bar=True)
        
        # del output
        # torch.cuda.empty_cache()
        return loss

    def forward(self, batch):
        output = self.model(batch)
        self.num_steps += 1
        self.on_step_end(self.num_steps,output)
        return output

    def on_step_end(self,steps,output,**kwargs):
        if steps % 100 == 0:
            x = output["prev_sample"].clone().to("cuda:0")
            # x = impro.de_normalize(x)
            z = output["diffusion_feature"].clone().to("cuda:0")
            if x.shape != z.shape:
                z = F.interpolate(z,size=x.shape[2:])
            # z = impro.de_normalize(z)
            grid = [x,z]
            display_images(grid, show_image=False,nrow=conf.batch_size,save_image=True, output_dir=conf.output_dir,output_name=f"compare_{steps%20}")
            # display_images([z], show_image=False,nrow=conf.batch_size,save_image=True, output_dir=conf.output_dir,output_name=f"output_{steps%20}")

        # if steps % 50 == 0:
        #     self.validation_step

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        loss = self.forward(batch)
        self.log("test_loss", loss)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        self.model.eval()
        output = self.forward(batch)
        output["eval_img"] = impro.de_normalize(output["eval_img"])
        output["noisy_images"] = impro.de_normalize(output["noisy_images"])
        output["diffusion_feature"] = impro.de_normalize(output["diffusion_feature"])
        display_images(output["eval_img"],show_image=False,nrow=conf.batch_size,save_image=True,output_dir=conf.output_dir,output_name=f"eval_img_{batch_idx}")
        display_images(output["noisy_images"],show_image=False,nrow=conf.batch_size,save_image=True,output_dir=conf.output_dir,output_name=f"noisy_images_{batch_idx}")
        display_images(output["diffusion_feature"],show_image=False,nrow=conf.batch_size,save_image=True,output_dir=conf.output_dir,output_name=f"pred_img_{batch_idx}")
        self.model.train()
        return output

    def configure_optimizers(self):
        diffusion_optimizer = create_optimizer_v2(self.model,opt='adafactor',lr=conf.learning_rate,weight_decay=0.,momentum=0.9,foreach=None,filter_bias_and_bn=True,layer_decay=None,param_group_fn=None,)
        diffusion_scheduler, num_epochs = create_scheduler_v2(optimizer=diffusion_optimizer,sched="cosine",num_epochs=conf.num_epochs,decay_epochs=10,cooldown_epochs=0,patience_epochs=10,decay_rate=0.1,min_lr=conf.min_lr,warmup_lr=conf.learning_rate,warmup_epochs=0,warmup_prefix=False,noise_pct=0.67,noise_std=1.0,noise_seed=42,cycle_mul=1.0,cycle_decay=0.1,cycle_limit=1,step_on_epochs=True,updates_per_epoch=0,)
        return [diffusion_optimizer], [
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
    monitor="diffusion_loss",
    # save_last=True,
    save_top_k=1,
    mode="min",
    every_n_train_steps=1000
)
trainer = L.Trainer(
    # enable_checkpointing=True,
    default_root_dir=conf.default_root_dir,
    # profiler=profiler,
    profiler=conf.profiler,
    callbacks=[loss_checkpoint,lr_monitor,ModelSummary(max_depth=1)],
    precision="16-mixed",
    max_epochs=conf.num_epochs,
    # precision="transformer-engine-float16",
    accelerator="auto",
    devices=1,
    strategy="auto",
    # gradient_clip_val=1,
    logger=logger,
    log_every_n_steps=5,
    # check_val_every_n_epoch=1,
    # val_check_interval=50,
)
#%%
model = LitModel()
# model = LitModel.load_from_checkpoint(conf.ckpt_dir/"image-segmentation_loss.ckpt")
trainer.fit(model=model, train_dataloaders=dataloader)
# %%
# model = LitModel.load_from_checkpoint(conf.ckpt_dir/"image-segmentation_loss-v11.ckpt")
# trainer.predict(model=model, dataloaders=dataloader)
# %%
