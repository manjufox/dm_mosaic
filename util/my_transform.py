import torch
import torch.nn as nn
import torchvision.transforms.v2 as v2
import numpy as np 
import random
import matplotlib.pyplot as plt
from torchvision.io import read_image
import math

import sys
sys.path.append("..")
sys.path.append(".")
from util.debug_print import dprint
from setting.setting import *

class TransformBase(nn.Module):
    def __init__(self):
        super().__init__()

class Resize(TransformBase):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x):
        return v2.functional.resize(x, self.size)

class Crop(TransformBase):
    def __init__(self, size=None, point=None):
        super().__init__()

        self.point = point
        if size is None:
            self.size = (H,W)
        else:
            self.size = size

    def forward(self, x):
        return x[...,
            self.point[0]:self.point[0]+self.size[0], 
            self.point[1]:self.point[1]+self.size[1]
        ]

class Affine(TransformBase):
    def __init__(self, size, degrees=0, translate1=0, translate2=0, scale=1, shear=0):
        super().__init__()
        x, y = size
        self.degrees = degrees * 45
        self.translate = (x*translate1, y*translate2)
        self.scale = scale
        self.shear = shear

    def forward(self, x):
        return v2.functional.affine(x, self.degrees, self.translate, self.scale, self.shear)

class GaussianBlur(TransformBase):
    def __init__(self, kernel_size, sigma):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        
    def forward(self, x):
        return v2.functional.gaussian_blur(x, self.kernel_size, self.sigma)

class GaussianNoise(TransformBase):
    def __init__(self, mean, sigma):
        super().__init__()
        self.mean = mean
        self.sigma = sigma
        
    def forward(self, x):
        return v2.functional.gaussian_noise(x, self.mean, self.sigma)

class Mask(TransformBase):
    def __init__(self, size, point):
        super().__init__()
        self.h,self.w= size
        self.x0 = point[1]
        self.y0 = point[0]
        self.xw = self.x0+self.w
        self.yh = self.y0+self.h

    def forward(self, x):
        # mask = torch.zeros_like(x)
        if len(x.shape) == 4:
            x[:, :, self.y0:self.yh, self.x0:self.xw] = torch.ones(size=(x.shape[0], x.shape[1], self.h, self.w))
        else:
            x[:, self.y0:self.yh, self.x0:self.xw] = torch.ones(size=(x.shape[0], self.h, self.w))
        return x

def generate_noise(num=100,noise_scale=0.01):
    # np.random.seed(0)
    # random.seed(0)
    # dn = np.random.uniform(-1,1, size=num) * NOISE_SCALE
    # dm = np.random.uniform(0,1, size=num) * NOISE_SCALE
    # params = ["gmean", 'gnoise']
    params = ["scale","gmean","gnoise"]
    # samples =  np.linspace(0,1,num) + np.random.uniform(-1,1, size=num) * NOISE_SCALE
    # samples = samples / samples[-1] 
    noise = {param: np.full(shape=(num,),fill_value=np.random.rand()/num) for param in params}
    noise['scale'] = np.linspace(0,np.random.uniform(0,1),num)
    # noise["gblur"] = np.linspace(0.000001,np.random.uniform(0,1),num)*NOISE_SCALE
    noise["gnoise"] =np.linspace(0,np.random.uniform(0,1),num)
    noise["gmean"] =np.linspace(0,np.random.uniform(0,1),num)
    # noise["scale"] = np.linspace(0.5,np.random.uniform(0,1),num)*NOISE_SCALE
    # noise["translate1"] = np.zeros(num)
    # noise["translate2"] = np.zeros(num)

    return noise

def noisy_image_generator(x, noise, i):
    # params = {k: noise[k][i] for k in noise}
    # params = noise["scale"][i]
    # x = Mask(size=(256,256), x0=params['x0'], y0=params['y0'], xw=params['xw'], yh=params['yh'])(x)
    # x = GaussianBlur(kernel_size=5, sigma=params['gblur'])(x)
    scale = noise['scale'][i]
    mean = noise['gmean'][i]
    sigma = noise['gnoise'][i]
    x = Affine(size=(H,W), scale=math.pow(SCALE_FACTOR,scale))(x)
    x = GaussianNoise(mean=mean, sigma=sigma)(x)
    # x = v2.Resize(size=(int(H*params['scale']),int(W*params['scale'])))(x)
    # affine = Affine(
    #     size=(H,W),
    #     degrees=params['degree'],
    #     translate1=0,#params['translate1'],
    #     translate2=0,#params['translate2'],
    #     scale=params['scale']+1,f
    #     shear=0
    # )
    # x = affine(x)params['degree'], translate1=0, translate2=0
    # print(math.exp(params))
    # x = v2.functional.resize(x, (H^params['scale'],W^params['scale']))
    # label = torch.tensor([params['gmean'], params['gnoise']],dtype=torch.float32)
    label = torch.tensor([noise['scale'][i],noise['gmean'][i], noise['gnoise'][i]],dtype=torch.float32)
    return x, label

def show_image(img):
    if len(img.shape) == 4:
        img = img[0]
    plt.imshow(img.permute(1, 2, 0))
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    image = read_image(TEST_IMG)
    img = Crop(size=(H,W), point=(100,100))(image)
    img = img.unsqueeze(0)
    img = img/255
    noise = generate_noise(DENOISE_TIME)
    print(noise['scale'])
    for i in range(DENOISE_TIME):

        noisy_img,label = noisy_image_generator(img, noise, i)
        if i % 100 == 0:
            show_image(noisy_img)
    
