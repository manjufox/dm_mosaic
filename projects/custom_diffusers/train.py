# %%
import sys
import time

import lightning as L
import torch
import torch.nn.functional as F
import torch.optim as optim
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

sys.path.append("../..")
sys.path.append("..")
sys.path.append(".")

from config import TrainingConfig
from image_processing import ImageProcessing

from dataset import PreLoad
from model import DiffusionModel
from util.image_util import display_images
from util.run_tensorboard import run_tensorboard

conf = TrainingConfig()
impro = ImageProcessing()

import warnings

if not conf.debug:
    warnings.filterwarnings("ignore")

print(f"project: {conf.project_name}")
print(f"cuda: {torch.cuda.is_available()}")
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True

try:
    run_tensorboard(conf)
except:
    pass

# torch.set_grad_enabled(True)
# torch.set_default_dtype(torch.float16)

class LitModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = DiffusionModel()
        self.num_steps = 0
        self.save_hyperparameters()
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        batch["steps"] = self.global_step
        batch["logger"] = logger.experiment
        diffusion_opt = self.optimizers()
        # モデルの計算
        output = self.forward(batch)
        loss = output["loss"]

        # diffusion_lossの勾配計算
        diffusion_opt.zero_grad()
        self.manual_backward(loss)
        self.clip_gradients(diffusion_opt, gradient_clip_val=1, gradient_clip_algorithm="norm")
        diffusion_opt.step()
        self.lr_scheduler_step(self.lr_schedulers(), loss)


        for key,value in output.items():
            if any([k in key for k in ["loss","time"]]):
                prog_bar = True
            else:
                prog_bar = False
            if isinstance(value,torch.Tensor):
                if len(value.shape) == 0:
                    self.log(key,value,prog_bar=prog_bar)
                elif value.dtype == torch.float32:
                    self.log(f"{key}_mean",value.mean(),prog_bar=prog_bar)
        return loss

    def forward(self, batch):
        start = time.time()
        output = self.model(batch)
        end = time.time() - start
        self.log("time",end,prog_bar=True)
        self.num_steps += 1
        self.on_step_end(self.num_steps,output)
        return output

    def on_step_end(self,steps,output,**kwargs):
        if steps % conf.save_image_each_n_steps == 0:
            x = output["noisy_images"].detach().cpu()
            x = impro.de_normalize(x)
            z = output["noise_pred"].detach().cpu()
            if x.shape != z.shape:
                z = F.interpolate(z,size=x.shape[2:])
            z = impro.de_normalize(z)
            grid = [x,z]
            output["logger"].add_image(f"noise_pred/{steps%conf.save_image_each_n_steps}", z[0], steps)
            display_images(grid, show_image=False,nrow=conf.batch_size,save_image=True, output_dir=conf.output_dir,output_name=f"compare_{steps%conf.save_image_each_n_steps}")

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        pass

    def configure_optimizers(self):
        # diffusion_optimizer = create_optimizer_v2(self.model,opt='adafactor',lr=conf.learning_rate,weight_decay=0.,momentum=0.9,foreach=None,filter_bias_and_bn=True,layer_decay=None,param_group_fn=None,)
        # diffusion_scheduler, num_epochs = create_scheduler_v2(optimizer=diffusion_optimizer,sched="poly",num_epochs=conf.num_steps,decay_epochs=100,cooldown_epochs=0,patience_epochs=10,decay_rate=0.1,
        #                                                       min_lr=conf.min_lr,warmup_lr=conf.learning_rate,warmup_epochs=0,warmup_prefix=False,noise_pct=0.67,noise_std=1.0,noise_seed=42,cycle_mul=1.0,cycle_decay=0.1,cycle_limit=1,step_on_epochs=True,updates_per_epoch=0,)
        optimizer = optim.AdamW(self.model.parameters(),lr=conf.max_lr,weight_decay=0.01)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=5)
        return [optimizer], [
            {"scheduler": scheduler, "interval": "step", "monitor": "loss"}
        ]

    def lr_scheduler_step(self, scheduler, metric):
        self.log("lr",scheduler.get_last_lr()[-1],prog_bar=True)
        scheduler.step(metrics=metric)


# 1. データの準備
dataset = PreLoad().dataset
dataloader = DataLoader(dataset,batch_size=conf.batch_size,num_workers=conf.num_workers,collate_fn=impro.collate_fn)
logger = TensorBoardLogger(conf.default_root_dir, name=conf.log_dir.name)




lr_monitor = LearningRateMonitor(logging_interval='step')

loss_checkpoint = ModelCheckpoint(
    dirpath=conf.ckpt_dir,
    filename=f"{conf.project_name}_loss",
    monitor="loss",
    # save_last=True,
    save_top_k=1,
    mode="min",
    every_n_train_steps=conf.check_point_each_n_steps
)
trainer = L.Trainer(
    # enable_checkpointing=True,
    default_root_dir=conf.default_root_dir,
    profiler=conf.profiler,
    callbacks=[loss_checkpoint,lr_monitor,ModelSummary(max_depth=1)],
    precision="16-mixed",
    max_epochs=conf.num_epochs,
    accelerator="auto",
    devices=1,
    strategy="auto",
    # strategy=L.pytorch.strategies.DDPStrategy(find_unused_parameters=True),
    logger=logger,
    log_every_n_steps=5,
)
#%%
model = LitModel()
# model = LitModel.load_from_checkpoint(conf.ckpt_dir/f"{conf.project_name}_{conf.num_train_timesteps}_loss.ckpt")
trainer.fit(model=model, train_dataloaders=dataloader)
# %%
# model = LitModel.load_from_checkpoint(conf.ckpt_dir/"image-segmentation_loss-v11.ckpt")
# trainer.predict(model=model, dataloaders=dataloader)
# %%# %%
# %%
