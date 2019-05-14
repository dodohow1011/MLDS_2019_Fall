import torch
import torch.nn as nn
import numpy as np
import sys

def initialize_weights(model):
    for m in model._modules:
        if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
            model._modules[m].weight.data.normal_(0.0, 0.02)
            model._modules[m].bias.data.zero_()

class Generator(nn.Module):
    def __init__(self, input_channel, out_channel, dropout):
        super(Generator, self).__init__()

        self.loss = nn.BCELoss()

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(input_channel, out_channel, 4, 1, 0, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout),
            
            nn.ConvTranspose2d(out_channel, out_channel, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout),
        
            nn.ConvTranspose2d(out_channel, out_channel, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout),

            nn.ConvTranspose2d(out_channel, out_channel, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout),
        
            nn.ConvTranspose2d(out_channel, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

        initialize_weights(self)

    def forward(self, input_noise):
        output_image = self.layers(input_noise)
        return output_image

class Discriminator(nn.Module):
    def __init__(self,  out_channel, dropout):
        super(Discriminator, self).__init__()

        self.loss = nn.BCELoss()

        self.layers = nn.Sequential(
            nn.Conv2d(3, out_channel, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(out_channel, out_channel, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2, inplace=True),
        
            nn.Conv2d(out_channel, out_channel, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(out_channel, out_channel, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2, inplace=True),
        
            nn.Conv2d(out_channel, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

        initialize_weights(self)

    def forward(self, input_image):
        output_score = self.layers(input_image)
        return output_score
