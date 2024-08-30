import cv2
import numpy as np
import torch
from torchvision import transforms
import monai
from monai.config import KeysCollection
from typing import Union

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class Resize_cv2(transforms.Resize):
    def __init__(self, size, interpolation=cv2.INTER_AREA):
        super().__init__(size)
        self.interpolation = interpolation
        self.size = size
    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be scaled.

        Returns:
            PIL Image or Tensor: Rescaled image.
        """
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
    
        img_resized = cv2.resize(np.asarray(img), self.size, interpolation=self.interpolation)
        # Convert back to a tensor if necessary
        if isinstance(img_resized, np.ndarray):
            if img_resized.ndim == 2: #stupid cv removes gray channel
                img_resized = torch.tensor(np.expand_dims(img_resized, axis=0))
            else:
                img_resized = torch.tensor(img_resized).permute(2, 0, 1)  # Convert from (H, W, C) to (C, H, W)
        
        return img_resized
    
# B-scan supervised training Transformations
def bscan_sup_transforms(rotation=10):
    train_transform = transforms.Compose([Resize_cv2((224,224),interpolation=cv2.INTER_AREA),
                                      transforms.RandomAffine(0,(0.05,0.05),fill=0, interpolation=transforms.InterpolationMode.BILINEAR),
                                      transforms.RandomRotation(degrees=rotation, interpolation=transforms.InterpolationMode.BILINEAR),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ConvertImageDtype(torch.float32)])
    return train_transform

# With Norm, based on v1 not v2!!!!
def bscan_sup_transformsv3(NORM,rotation=10):
    train_transform = transforms.Compose([Resize_cv2((224,224),interpolation=cv2.INTER_AREA),
                                      transforms.RandomAffine(0,(0.05,0.05),fill=0, interpolation=transforms.InterpolationMode.BILINEAR),
                                      transforms.RandomRotation(degrees=rotation, interpolation=transforms.InterpolationMode.BILINEAR),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ConvertImageDtype(torch.float32),
                                      transforms.Normalize(*NORM)])
    return train_transform

def Monai2DTransforms(transforms, image_keys=["image"]):
    transf = monai.transforms.Compose([monai.transforms.RandLambdad(keys = image_keys, func = lambda x: transforms(x),prob=1.0), # Apply random torchvision transforms, works with caching as well
                                       monai.transforms.ToTensord(keys = image_keys + ["label"], track_meta=False)])
    return transf


# SimCLR Transformations
def simclr_transforms(image_size: Union[int, tuple], jitter: tuple = (0.4, 0.4, 0.2, 0.1),
                      p_blur: float = 1.0, p_solarize: float = 0.0,
                      normalize: list = [[0.485],[0.229]], translation=True, scale=(0.4, 0.8), gray_scale=True,
                      sol_threshold=0.42, p_jitter=0.8, vertical_flip=False, p_horizontal_flip=0.5, rotate=False):
    """
    Returns a composition of transformations for SimCLR training.

    Args:
        image_size (Union[int, tuple]): The size of the output image. If int, the output image will be square with sides of length `image_size`. If tuple, the output image will have dimensions `image_size[0]` x `image_size[1]`.
        jitter (tuple, optional): Tuple of four floats representing the range of random color jitter. Defaults to (0.4, 0.4, 0.2, 0.1).
        p_blur (float, optional): Probability of applying Gaussian blur. Defaults to 1.0.
        p_solarize (float, optional): Probability of applying random solarization. Defaults to 0.0.
        normalize (list, optional): List of two lists representing the mean and standard deviation for image normalization. Defaults to [[0.485],[0.229]].
        translation (bool, optional): Whether to apply small random translation. Defaults to True.
        scale (tuple, optional): Tuple of two floats representing the range of random scale for random resized crop. Defaults to (0.4, 0.8).
        gray_scale (bool, optional): Whether the input image is in gray scalr or not. Defaults to True.
        sol_threshold (float, optional): Threshold for random solarization. Defaults to 0.42.
        p_jitter (float, optional): Probability of applying color jitter. Defaults to 0.8.
        vertical_flip (bool, optional): Whether to apply random vertical flip. Defaults to False.
        p_horizontal_flip (float, optional): Probability of applying random horizontal flip. Defaults to 0.5.
        rotate (bool, optional): Whether to apply random rotation. Defaults to False.

    Returns:
        torchvision.transforms.Compose: A composition of transformations.
    """
    trans_list = []
    image_size = pair(image_size)

    # Add small translation. This was added after TINC paper
    if translation:
        trans_list.append(transforms.RandomAffine(0,(0.05,0.05), fill=0, interpolation=transforms.InterpolationMode.BILINEAR))
    if rotate:
        trans_list.append(transforms.RandomRotation(rotate, interpolation=transforms.InterpolationMode.BILINEAR))

    trans_list += [Resize_cv2((224,224),interpolation=cv2.INTER_AREA),
                   transforms.RandomResizedCrop(image_size, scale=scale, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                  transforms.RandomHorizontalFlip(p=p_horizontal_flip),
                  transforms.ConvertImageDtype(torch.float32)]

    trans_list.append(transforms.RandomApply([transforms.ColorJitter(*jitter)], p=p_jitter))

    if vertical_flip:
        trans_list.append(transforms.RandomVerticalFlip(p=0.5))
    #If image is not grayscale add RandomGrayscale
    if not gray_scale:
        trans_list.append(transforms.RandomGrayscale(p=0.2))
    # Turn off blur for small images
    if image_size[0]<=32:
        p_blur = 0.0
    # Add Gaussian blur
    if p_blur==1.0:
        trans_list.append(transforms.GaussianBlur(image_size[0]//20*2+1))
    elif p_blur>0.0:
        trans_list.append(transforms.RandomApply([transforms.GaussianBlur(image_size[0]//20*2+1)], p=p_blur))
    # Add RandomSolarize
    if p_solarize>0.0:
        trans_list.append(transforms.RandomSolarize(sol_threshold, p=p_solarize))

    # Feb 9, moved norm to the end
    if normalize:
        trans_list.extend([transforms.Normalize(*normalize)])
    
    return transforms.Compose(trans_list)


def Barlow_augmentaions(image_size, normalize=[[0.485],[0.229]], gray_scale=True, translation=False, p_blur=0.8, scale=(0.08, 1.0), 
                        temporal=False, sol_threshold=0.42, jitter: tuple = (0.4, 0.4, 0.2, 0.1), p_solarize=0.2, p_jitter=0.8, 
                        vertical_flip=False, p_horizontal_flip=0.5, rotate=False):
    if temporal:        
        trans1 = simclr_transforms(image_size,
                                   p_blur = p_blur,
                                   p_solarize = p_solarize,
                                   normalize = normalize,
                                   translation = translation,
                                   scale = scale,
                                   gray_scale = gray_scale,
                                   sol_threshold = sol_threshold,
                                   jitter = jitter,
                                   p_jitter = p_jitter,
                                   vertical_flip=vertical_flip,
                                   p_horizontal_flip=p_horizontal_flip,
                                   rotate=rotate)
        return trans1
    
    # This is for vanilla Barlow twins for OCT
    trans1 = simclr_transforms(image_size,
                                   p_blur = p_blur,
                                   p_solarize = 0.0,
                                   normalize = normalize,
                                   translation = translation,
                                   scale = scale,
                                   gray_scale = gray_scale,
                                   sol_threshold = sol_threshold,
                                   jitter = jitter,
                                   p_jitter = p_jitter,
                                   vertical_flip=vertical_flip,
                                   p_horizontal_flip=p_horizontal_flip,
                                   rotate=rotate)
        
    trans2 = simclr_transforms(image_size,
                                   p_blur = max(0.0, 1.0-p_blur),
                                   p_solarize = p_solarize,
                                   normalize = normalize,
                                   translation = translation,
                                   scale = scale,
                                   gray_scale = gray_scale,
                                   sol_threshold = sol_threshold,
                                   jitter = jitter,
                                   p_jitter=p_jitter,
                                   vertical_flip=vertical_flip,
                                   p_horizontal_flip=p_horizontal_flip,
                                   rotate=rotate)
        
    return [trans1, trans2]

def Monai2DContrastTransforms(transforms):
    transf = monai.transforms.Compose([monai.transforms.CopyItemsd(keys=["image"], times=1, names=["image_1"], allow_missing_keys=False),
                     monai.transforms.RandLambdad(keys = ["image"], func = lambda x: transforms[0](x),prob=1.0),
                     monai.transforms.RandLambdad(keys = ["image_1"], func = lambda x: transforms[1](x),prob=1.0),
                     monai.transforms.ToTensord(keys = ["image","image_1","label"], track_meta=False)])

    return transf
