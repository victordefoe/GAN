import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.autograd import Variable, Function

from torch.nn import init
import math


import os
import logging

class generator(nn.Module):
    """
        this is the generator model for GAN
    """
    def __init__(self, noise_dim):
        super(generator, self).__init__()
        self.init_layer = nn.Linear(noise_dim, 4*4*512)
        self.layers = nn.ModuleList()
        self.layers.extend([
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                        in_channels=512,
                        out_channels=256,
                        kernel_size=4,
                        stride=1,
                        padding=0,
                        output_padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                        in_channels=256,
                        out_channels=128,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.8),
            nn.ConvTranspose2d(
                        in_channels=128,
                        out_channels=1,
                        kernel_size=3,
                        stride=2,
                        padding=0,
                        output_padding=1),
            nn.Tanh()
            ]),
            
        
    
    def forward(self, x):
        x = self.init_layer(x)
        x = x.view(-1, 512, 4, 4)
        for ele in self.layers:
            x = ele(x)
            # print('WEIDO: ', x.shape)
        
        return x

class discriminator(nn.Module):
    """
        this is the discriminator model for GAN
    """
    def __init__(self):
        super(discriminator, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.extend([
            nn.Conv2d(1, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Dropout(p=0.8),
            nn.Conv2d(128, 256, kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        ])

        self.fc = nn.Linear(4*4*512, 1)
        self.active = nn.Sigmoid()


    def forward(self, x):
        for ele in self.layers:
            x = ele(x)
            print('DIM_OF_D',x.shape)
        x = x.view(-1, 4*4*512)
        x = self.fc(x)
        x = self.active(x)

        return x
        

if __name__ == '__main__':
    G = generator(100)
    a = torch.randn((64,100))
    b = G(a)
    print(a.size(),b.size())

