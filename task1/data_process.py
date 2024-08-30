# -*- coding: utf-8 -*-
"""
Created on Tue May 14 14:50:52 2024

@author: TeresaFinis
"""

from torchvision import transforms
import numpy as np
from PIL import Image
import torchvision
from torchvision.transforms import InterpolationMode


def processImage(img):
    
    if np.max(img)>255:
        img = img/2**16
    else:
        img = img/2**8
    img = img*255

    img = Image.fromarray(np.uint8(img)).convert('RGB')
    
    return img


def data_transforms(input_size, normalize=True,interpolation_order=2,volume_mode=False,
                        nb_bscans=None,resize=False):
       
    if interpolation_order not in [0,2]:
        raise Warning('interpolation_order should be either 0 or 2. Considering interpolation order 2')
    interpolation = InterpolationMode.BILINEAR
    if interpolation_order == 0:
        interpolation = InterpolationMode.NEAREST
              
    if type(input_size) is int:
        input_size = [input_size,input_size]
 
    test_transforms = [
        transforms.Resize((input_size[0],input_size[1]),interpolation=interpolation),
        transforms.ToTensor(),
    ]
    if normalize:
        test_transforms.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))


    data_transform = transforms.Compose(test_transforms)
    
    return data_transform



