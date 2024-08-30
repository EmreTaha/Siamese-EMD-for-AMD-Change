# -*- coding: utf-8 -*-
"""
Created on Tue May 14 15:54:09 2024

@author: TeresaFinis
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

import models_vit

class SiameseNetwork(nn.Module):

    def __init__(self, nb_classes, device = 'cpu'):
        super(SiameseNetwork, self).__init__()
          
        self.model = models_vit.vit_large_patch16(
            num_classes=nb_classes,
            drop_path_rate=0.1,
            global_pool=True,
        )
        
        self.model.to(device)
                     
        self.fc_in_features = self.model.head.in_features
  
        self.model.forward = self.model.forward_features
  
        f = self.model.forward_features
        ff = self.model.patch_embed
        fff = self.model.pos_embed

        del self.model.head
     
        self.fc = nn.Sequential(
               nn.Linear(self.fc_in_features*2, 256),
               nn.ReLU(inplace=True),
               nn.Dropout(p=0.25),
               nn.Linear(256, nb_classes),
          )

        self.fc.apply(self.init_weights)
 
            
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x):
        output = self.model(x)
        
        output = output.view(output.size()[0], -1)

        return output

    def forward(self, input1, input2):
        
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        output = torch.cat((output1, output2), 1)
        self.features = output
    
        output = self.fc(output)
        
        return output
    
    
