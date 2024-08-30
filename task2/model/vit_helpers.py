from timm.models.layers import to_3tuple
import torch
import torch.nn as nn

import numpy as np

class PatchEmbed3D(nn.Module):
    """ 3D Image to Patch Embedding
    """
    def __init__(
            self,
            img_size=224,
            patch_size=(16,16,8),
            in_chans=1,
            embed_dim=768,
            norm_layer=None,
            flatten=True,
            bias=True,
    ):
        super().__init__()
        patch_size = to_3tuple(patch_size)
        self.img_size = to_3tuple(img_size)
        self.patch_size = patch_size
        self.grid_size = []
        for im_size, pa_size in zip(img_size, patch_size):
            self.grid_size.append(im_size // pa_size)        
        
        self.num_patches = np.prod(self.grid_size)
        self.flatten = flatten

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        #B, C, H, W = x.shape #TODO fix this to D
        _, _,  H, W, D = x.shape            
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)   # BCHWD -> BNC
        x = self.norm(x)
        return x
    
def get_3d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width and depth
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    #TODO check where d comes
    grid_size = to_3tuple(grid_size)
    h, w, d = grid_size
    grid_h = np.arange(h, dtype=float)
    grid_w = np.arange(w, dtype=float)
    grid_d = np.arange(d, dtype=float)
    grid = np.meshgrid(grid_w, grid_h, grid_d)  # here w goes first #TODO xy in np, if swithc to pytorch, need to switch to yx #TODO check ij of pytorch
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([3, 1, h, w, d])
    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 3 == 0

    # use 1/3rd to encode grid_h, grid_w, grid_d
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[0])  # (H*W*D, D/3)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[1])  # (H*W*D, D/3)
    emb_d = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[2])  # (H*W*D, D/3)

    emb = np.concatenate([emb_h, emb_w, emb_d], axis=1) # (H*W*D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb