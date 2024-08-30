from .resnet import GrayResNet
from .projector import MLP
from .mae import mae_vit_small_patch16_d,mae_vit_small_patch16,mae_vit_base_patch16, mae_vit_large_patch16
from .vit import vit_base_patch16, vit_large_patch16, vit_huge_patch14, vit_small_patch16

Models = {
    "GrayResNet": GrayResNet,
    "mae_vit_small_patch16_d": mae_vit_small_patch16_d, # small decoder
    "mae_vit_small_patch16": mae_vit_small_patch16,
    "mae_vit_base_patch16": mae_vit_base_patch16,
    "mae_vit_large_patch16": mae_vit_large_patch16,
    "vit_small_patch16": vit_small_patch16,
    "vit_base_patch16": vit_base_patch16,
    "vit_large_patch16": vit_large_patch16,
    "vit_huge_patch14": vit_huge_patch14
    }