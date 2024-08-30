import numpy as np
import torch
from matplotlib import pyplot as plt
import os
from timm.models.layers import trunc_normal_


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model, orig_n_patch):
    #TODO 3D, didnt check 2D
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        #orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        #new_size = int(num_patches ** 0.5)
        new_size = np.array(model.patch_embed.grid_size)
        # class_token and dist_token are kept unchanged
        if (orig_n_patch != new_size).any():
            print(f'Position interpolate from {orig_n_patch} to {new_size}')
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            #pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = pos_tokens.reshape(-1, *orig_n_patch, embedding_size).permute(0, 4, 1, 2, 3)
            pos_tokens = torch.nn.functional.interpolate(
                #pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                pos_tokens, size=tuple(new_size.astype(int)), mode='area')
            #pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            pos_tokens = pos_tokens.permute(0, 2, 3, 4, 1).flatten(1, 3)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed

def show_image(image, title='', NORM=None):
    if NORM:
        plt.imshow(torch.clip((np.squeeze(image) * NORM[1][0] + NORM[0][0]) * 255, 0, 255).int(),'gray')
    else:
        plt.imshow(np.squeeze((torch.clip((image) * 255, 0, 255)/255)),'gray')    
    plt.title(title, fontsize=16)
    plt.axis('off')
    return

def run_one_image(img, model, save_dir, NORM, mask_ratio=0.75):
    x = img.cuda()

    # make it a batch-like
    if len(x.shape)<4:
        x = x.unsqueeze(dim=0)

    # run MAE
    _, y, mask = model(x.float(), mask_ratio=mask_ratio)
    y = model.unpatchify(y)
    y = y.detach().cpu()
    x = x.detach().cpu()

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *x.shape[1]) 
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = mask.detach().cpu()
    
    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask

    # make the plt figure larger
    plt.rcParams['figure.figsize'] = [24, 24]

    plt.subplot(1, 4, 1)
    show_image(x[0].permute(1,2,0), "original", NORM)

    plt.subplot(1, 4, 2)
    show_image(im_masked[0].permute(1,2,0), "masked", NORM)

    plt.subplot(1, 4, 3)
    show_image(y[0].permute(1,2,0), "reconstruction", NORM)

    plt.subplot(1, 4, 4)
    show_image(im_paste[0].permute(1,2,0), "reconstruction + visible", NORM)

    # Save the figure to a file    
    plt.savefig(save_dir+'.png', dpi=300)  # Modify the filename and DPI as needed

    # Clear the current figure after saving to avoid overlap if this function is called multiple times
    plt.clf()

def run_two_image(img, img_future, model, save_dir, NORM, mask_ratio=0.75):
    x = img.cuda()
    x_future = img_future.cuda()

    # make it a batch-like
    if len(x.shape)<4:
        x = x.unsqueeze(dim=0)
        x_future = x_future.unsqueeze(dim=0)

    # run MAE
    _, y, mask = model(x.float(), x_future, mask_ratio=mask_ratio)
    y = model.unpatchify(y)
    y = y.detach().cpu()
    x = x.detach().cpu()
    x_future = x_future.detach().cpu()

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2*x.shape[1]) 
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = mask.detach().cpu()
    
    # masked future image
    im_masked = x_future * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x_future * (1 - mask) + y * mask

    # make the plt figure larger
    plt.rcParams['figure.figsize'] = [24, 24]

    plt.subplot(1, 5, 1)
    show_image(x[0], "original", NORM)

    plt.subplot(1, 5, 2)
    show_image(x_future[0], "future", NORM)

    plt.subplot(1, 5, 3)
    show_image(im_masked[0], "masked", NORM)

    plt.subplot(1, 5, 4)
    show_image(y[0], "reconstruction", NORM)

    plt.subplot(1, 5, 5)
    show_image(im_paste[0], "reconstruction + visible", NORM)

    # Save the figure to a file    
    plt.savefig(save_dir+'.png', dpi=300)  # Modify the filename and DPI as needed

    # Clear the current figure after saving to avoid overlap if this function is called multiple times
    plt.clf()

def run_one_volume(img, model, save_dir, NORM, mask_ratio=0.75):
    x = img.cuda()

    # make it a batch-like
    if len(x.shape)<5:
        x = x.unsqueeze(dim=0)

    # run MAE
    *_, y, mask = model(x, mask_ratio=mask_ratio)
    y = model.unpatchify3D(y)
    y = y.detach().cpu()
    x = x.detach().cpu()

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]*model.patch_embed.patch_size[1]*model.patch_embed.patch_size[2] ) 
    mask = model.unpatchify3D(mask)  # 1 is removing, 0 is keeping
    mask = mask.detach().cpu()
    
    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask

    # Save the volume to a file    
    np.savez_compressed(save_dir, original=x, masked=im_masked, reconstruction=y, reconstruction_visible=im_paste)

def run_two_volume(img, img_future, model, save_dir, NORM, mask_ratio=0.75):
    # I hate my life, this is a mental illness
    x = img.cuda()
    x_future = img_future.cuda()

    # make it a batch-like assumes volume
    if len(x.shape)<5: 
        x = x.unsqueeze(dim=0)
        x_future = x_future.unsqueeze(dim=0)

    # run MAE
    *_, y, mask = model(x.float(), x_future, mask_ratio=mask_ratio)
    y = model.unpatchify3D(y)
    y = y.detach().cpu()
    x = x.detach().cpu()
    x_future = x_future.detach().cpu()

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]*model.patch_embed.patch_size[1]*model.patch_embed.patch_size[2]) 
    mask = model.unpatchify3D(mask)  # 1 is removing, 0 is keeping
    mask = mask.detach().cpu()
    
    # masked future image
    im_masked = x_future * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x_future * (1 - mask) + y * mask

    # Save the volume to a file    
    np.savez_compressed(save_dir, original=x, future=x_future, masked=im_masked, reconstruction=y, reconstruction_visible=im_paste)

def val_saver(valloader, model, save_dir, epoch, NORM=None, mask_ratio=0.75, single=True, volume=False):
    model.eval()
    os.makedirs(save_dir+'/'+str(epoch), exist_ok=True)
    with torch.no_grad():
        for i in valloader:
            save_path = save_dir+'/'+str(epoch)+'/'+str(i["patID"][0])+'_'+str(i['visitID'][0])
            if volume:
                #t = str(i["path"][0].split('/')[-1])
                #save_path = save_dir+'/'+str(epoch)+'/'+t.split('.')[0] #TODO this is for vibes, it doesnt work with HARBOR!
                if single: 
                    run_one_volume(i['image'], model, save_path, NORM, mask_ratio)
                else:
                    run_two_volume(i['image'], i['image_1'], model, save_path, NORM, mask_ratio)
            else:
                if single:
                    run_one_image(i['image'], model, save_path, NORM, mask_ratio)
                else:
                    run_two_image(i['image'], i['image_1'], model, save_path, NORM, mask_ratio)
    model.train()
    return


# layer wise decayer
# Layer decay doesnt work with native pytorch, thus large_lr_list is introduced
def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75, large_lr_list=[], lr=1e-5):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}

    num_layers = len(model.blocks) + 1

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay

        if n in large_lr_list:
            this_lr = lr*10
        else:
            this_lr = lr
            
        layer_id = get_layer_id_for_vit(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "lr": this_lr,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "lr": this_lr,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    # print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))

    return list(param_groups.values())


def get_layer_id_for_vit(name, num_layers):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    if name in ['cls_token', 'pos_embed']:
        return 0
    elif name.startswith('patch_embed'):
        return 0
    elif name.startswith('blocks'):
        return int(name.split('.')[1]) + 1
    else:
        return num_layers