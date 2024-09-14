import torch
import numpy as np
import matplotlib.pyplot as plt

import torchvision.transforms.functional as F
import torchvision.transforms as v2
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
from datetime import datetime
from PIL import Image

import sys
sys.path.append("..")
sys.path.append(".")

plt.rcParams["savefig.bbox"] = 'tight'


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

def make_grid_tensor(batch_tensor,nrow:int=2,normalize:bool=True,grid=True):
    if not isinstance(batch_tensor,list):
        batch_tensor = [batch_tensor]

    for i in range(len(batch_tensor)):
        match batch_tensor[i]:
            case Image.Image():
                batch_tensor[i] = v2.ToTensor()(batch_tensor[i])
                batch_tensor[i] = batch_tensor[i].unsqueeze(0)
            case torch.Tensor():
                batch_tensor[i] = batch_tensor[i].detach().cpu()
                batch_tensor[i] = batch_tensor[i].unsqueeze(0) if batch_tensor[i].dim() == 3 else batch_tensor[i]
            case list():
                batch_tensor[i] = make_grid_tensor(batch_tensor[i],nrow=1,normalize=normalize,grid=False)

    batch_tensor = torch.stack(batch_tensor,dim=0) if batch_tensor[0].dim() == 3 else torch.cat(batch_tensor,dim=0)
    # batch_tensor = v2.Normalize(mean=[0.5], std=[0.5])(batch_tensor)
    if grid:
        grid_tensor = make_grid(batch_tensor,nrow=nrow,normalize=normalize)
        return grid_tensor
    return batch_tensor


def display_images(batch_tensor,nrow:int=2,normalize:bool=True,show_image:bool=True,save_image:bool=False,output_dir=None,output_name:str=None):
    grid_tensor = make_grid_tensor(batch_tensor,nrow=nrow,normalize=normalize)
    grid_image = ToPILImage()(grid_tensor)
    if show_image:
        show(grid_image)
    if save_image:
        if output_name is None:
            output_name = datetime.now().strftime('%Y%m%d_%H%M%S')
        grid_image.save(f"{output_dir}/{output_name}.png")
    return grid_tensor
