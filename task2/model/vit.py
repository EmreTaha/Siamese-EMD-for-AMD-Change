import timm.models.vision_transformer
from functools import partial
import torch
from torch import nn
from timm.models.layers import trunc_normal_

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.pos_embed = nn.Parameter(torch.randn(1, self.patch_embed.num_patches + 1, kwargs['embed_dim']) * .02)
        # In a simple setup, I found retfound style slightly better
        #nn.init.normal_(self.head.weight, std=2e-5) # Retfound style
        trunc_normal_(self.head.weight, std=2e-5) # Retfound style
        nn.init.normal_(self.cls_token, std=.02)
        #trunc_normal_(self.backbone.head.weight, std=0.01) # MOCOv3 style

        embed_dim = kwargs['embed_dim']

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            
            # Push the fc_norm inside the head and make the original norm an identity
            self.fc_norm = nn.Identity()
            self.head =  torch.nn.Sequential(norm_layer(embed_dim), 
                                                 self.head)

            #del self.norm  # remove the original norm
            self.norm = nn.Identity()  # replace with identity
        else:
            self.norm = kwargs['norm_layer'](kwargs['embed_dim']) #TODO add this as optional. this is beforre returning CLS

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x_glob = x[:, self.num_prefix_tokens:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x_glob) # I changed it to identity, real fc_norm is in the head
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

    @torch.no_grad()
    def get_last_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1) 
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        for blk in self.blocks:
            x = blk(x)
        
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        # Clumped down a lot, original mae uses old timm, latest timm is too new
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

def vit_small_patch16( **kwargs):
    """ ViT-Small (ViT-S/16)
    """
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base_patch16(patch_size=(16,16), **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model