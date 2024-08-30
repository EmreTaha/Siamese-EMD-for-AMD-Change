import glob
import os
import re
import torch

import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join

def save_last_k(model, epoch, path, k=5):
    """
    Saves pytorch model, and keeps only top k models. If k is -1 it keeps all
    Args:
        model: Pytorch model to save
        epoch: Current epoch
        path: Path to save model
        k: Number of previous models to keep
    """
    save_files = glob.glob(path+'/*pth')
    save_files.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)]) #sort them numerically

    if k==-1:
        pass
    elif len(save_files) > k-1:
        os.remove(save_files[0])

    try:
        model_state_dict = model.state_dict()
    except AttributeError:
        model_state_dict = model.module.state_dict()

    torch.save(model_state_dict, path+'/model_'+str(epoch)+'.pth')

def load_last_k(path,device,k=-1):
    files = [f for f in listdir(path) if isfile(join(path, f)) and 'pth' in join(path, f)]

    if len(files)>0:
        if k==-1:
            last_epoch = max([int(file[6:-4]) for file in files])
            return torch.load(join(path, 'model_'+str(last_epoch)+'.pth'), map_location=device)
        else:
            return torch.load(join(path, 'model_'+str(k)+'.pth'), map_location=device)

def plot_lr(lr_list, path):
    '''
    Plots learning rate change
    '''
    plt.plot(lr_list, label='learning rate change')
    plt.xlabel('Steps')
    plt.ylabel('LRate')
    plt.legend()
    plt.savefig(path+'/lrate.png')



def check_existing_model(save_root, device):
    """
    Epochs must be saved in format: f'epoch_{epoch:03}.tar'.
    E.g. for epoch 20: epoch_020.tar.
    """
    # Get all files
    files = [f for f in listdir(save_root) if isfile(join(save_root, f))]
    files = [file for file in files if 'tar' in file] # Other files gets in, filter them out
    
    # init
    epoch_start = 0
    saved_data = None
    if len(files)>0 and (files[-1].split('.')[-1] == 'tar'):
        user_answer = "Users_answer"
        while user_answer not in ["y","n"]:
            user_answer = input("Pretrained model available, use it?[y/n]: ").lower()[0]
        if user_answer=="y":
            #epoch_start = max([int(file[-7:-4]) for file in files])
            epoch_start = max([int(file.split('_')[1].split('.')[0]) for file in files])
            # Load data
            saved_data = torch.load(join(save_root, f'epoch_{epoch_start:03}.tar'), map_location=device)
    
    return epoch_start, saved_data