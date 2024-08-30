# -*- coding: utf-8 -*-
"""
Created on Fri May 17 11:22:24 2024

@author: TeresaFinis
"""
import os 
import torch
import json
import numpy as np
import imageio
import argparse
from pathlib import Path
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from data_process import * 
from models import SiameseNetwork
from models_vit import interpolate_pos_embed


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='MARIO challenge - Siamese Network approach')


    parser.add_argument('-test_img_1','--test_img_1',
                        help='Path to the test image 1',
                        default=r'./sample_images/B3944840.png',
                        type=str)
      
    parser.add_argument('-test_img_2','--test_img_2',
                        help='Path to the test image 2',
                        default=r'./sample_images/AA8E1A00.png',
                        type=str)
    
    parser.add_argument('-weights','--weights',
                        help='Path for the trained weights',
                        type=str,
                        default=r'./model_weights/weights_best')
    
    args_test = parser.parse_args()
    
    # =========================================================================
    # Define variables 

    test_img_1 = args_test.test_img_1
    test_img_2 = args_test.test_img_2    
    weights_path = args_test.weights + '.pth'
                
    im_size = 224
          
    label_dict = {0: 'Reduced', 1: 'Stable', 2: 'Worsened', 3: 'Other'}
        
    nb_classes = len(label_dict)
    
    # =========================================================================
    # Read data 
    
    img1 = imageio.imread(test_img_1)   
    img2 = imageio.imread(test_img_2)   
    
    img1 = processImage(img1)
    img2 = processImage(img2)  
    
    transf = data_transforms(input_size=(im_size,im_size))

    img1 = transf(img1)
    img2 = transf(img2)


    # =========================================================================
    # Load trained model 
  
    encoder = SiameseNetwork(nb_classes = nb_classes, **{'device': device})
    
    #print('\n encoder', encoder)

    checkpoint = torch.load(weights_path, map_location='cpu')

    if 'model' in checkpoint.keys(): 
        checkpoint = checkpoint['model']
        
    if 'model' in checkpoint: 
        #checkpoint_model = checkpoint['model']
        checkpoint_model = {'model.'+k:checkpoint_model[k] for k in checkpoint_model.keys() if (k.startswith('block') or k.startswith('cls') or 
                                                                                                k.startswith('patch') or k.startswith('pos'))}
    else:
        checkpoint_model = checkpoint 

    state_dict = encoder.state_dict()        

    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]
            
    # interpolate position embedding
    interpolate_pos_embed(encoder, checkpoint_model)
    
    # load pre-trained model
    msg = encoder.load_state_dict(checkpoint_model, strict=False)
    print(msg)
    
    print("\n Loaded pre-trained checkpoint for SiamRETFound")

    encoder = encoder.to(device)

    # =========================================================================
    # Make prediction 
    
    encoder.eval()
   
    with torch.no_grad():
        x1,x2, n1, n2 = img1, img2, test_img_1, test_img_2
        x1 = x1[None,...].to(device)
        x2 = x2[None,...].to(device)

        print('\n Predicting change for image pair')
        out = encoder(x1,x2)
        
        pred  = np.argmax(out.detach().cpu().numpy(),axis=1)
        
        logits = out.detach().cpu().numpy()[0]

        #print('\n output logits.:', logits)

        probs_out = np.exp(logits)/np.sum(np.exp(logits))

        #print('\n output probs.:', probs_out)
        
        print('Image 1: ', Path(test_img_1).name)
        print('Image 2: ', Path(test_img_2).name)
        
        print('\n Predicted class:', label_dict[int(pred)])
            
