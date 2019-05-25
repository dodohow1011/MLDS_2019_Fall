import numpy as np
import sys
import os
import pickle
import imageio
import torch
import torch.optim as optim 
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torchvision import transforms
from argparse import ArgumentParser
from torchsummary import summary

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

NOISE_DIM = 100
LABEL_DIM = 23
LEAKY_RELU_ALPHA = 0.2

deconv1_channels_out = int(320)
deconv2_channels_in  = int(deconv1_channels_out)
deconv2_channels_out = int(deconv2_channels_in / 2)
deconv3_channels_in  = int(deconv2_channels_out)
deconv3_channels_out = int(deconv3_channels_in / 2)
deconv4_channels_in  = int(deconv3_channels_out)
deconv4_channels_out = int(deconv4_channels_in / 2)
deconv5_channels_in  = int(deconv4_channels_out)
deconv5_channels_out = int(3)

conv1_channels_in    = int(3)
conv1_channels_out   = int(32)
conv2_channels_in    = int(conv1_channels_out)
conv2_channels_out   = int(conv2_channels_in * 2)
conv3_channels_in    = int(conv2_channels_out)
conv3_channels_out   = int(conv3_channels_in * 2)
conv4_channels_in    = int(conv3_channels_out)
conv4_channels_out   = int(conv4_channels_in * 2)
conv5_channels_in    = int(conv4_channels_out)


class Generator(nn.Module):
    def __init__(self, label_dim=100):
        super(Generator, self).__init__()
        self.deconv1   = nn.ConvTranspose2d(in_channels=NOISE_DIM+label_dim, out_channels=deconv1_channels_out, kernel_size=4, stride=1, padding=0, bias=False)
        self.bn1       = nn.BatchNorm2d(deconv1_channels_out)
        self.deconv2   = nn.ConvTranspose2d(in_channels=deconv2_channels_in, out_channels=deconv2_channels_out, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2       = nn.BatchNorm2d(deconv2_channels_out)
        self.deconv3   = nn.ConvTranspose2d(in_channels=deconv3_channels_in, out_channels=deconv3_channels_out, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3       = nn.BatchNorm2d(deconv3_channels_out)
        self.deconv4   = nn.ConvTranspose2d(in_channels=deconv4_channels_in, out_channels=deconv4_channels_out, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4       = nn.BatchNorm2d(deconv4_channels_out)
        self.deconv5   = nn.ConvTranspose2d(in_channels=deconv5_channels_in, out_channels=deconv5_channels_out, kernel_size=4, stride=2, padding=1, bias=False)

    def initWeight(self, mean=0.0, std=0.02):
        for m in self._modules:
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(mean, std)
                m.bias.data.zero_()

    def forward(self, noise, labelcode):
        noise = torch.cat([noise, labelcode], 1)
        x = F.leaky_relu(self.bn1(self.deconv1(noise)), LEAKY_RELU_ALPHA)
        x = F.leaky_relu(self.bn2(self.deconv2(x)), LEAKY_RELU_ALPHA)
        x = F.leaky_relu(self.bn3(self.deconv3(x)), LEAKY_RELU_ALPHA)
        x = F.leaky_relu(self.bn4(self.deconv4(x)), LEAKY_RELU_ALPHA)
        x = torch.tanh(self.deconv5(x))
        return x

class Discriminator(nn.Module):
    def __init__(self, label_dim=100):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=conv1_channels_in, out_channels=conv1_channels_out, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(conv1_channels_out)
        self.conv2   = nn.Conv2d(in_channels=conv2_channels_in, out_channels=conv2_channels_out, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2     = nn.BatchNorm2d(conv2_channels_out)
        self.conv3   = nn.Conv2d(in_channels=conv3_channels_in, out_channels=conv3_channels_out, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3     = nn.BatchNorm2d(conv3_channels_out)
        self.conv4   = nn.Conv2d(in_channels=conv4_channels_in, out_channels=conv4_channels_out, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4     = nn.BatchNorm2d(conv4_channels_out)
        self.output  = nn.Linear(conv4_channels_out*4*4+100, 1)


    def initWeight(self, mean=0.0, std=0.02):
        for m in self._modules:
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(mean, std)
                m.bias.data.zero_()

    def forward(self, image, labelcode, drop_rate=0.2):
        x = F.leaky_relu(self.bn1(self.conv1(image)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        x = x.view(x.size(0), -1)
        labelcode = labelcode.squeeze()
        x = torch.cat([x, labelcode], 1)
        x = torch.sigmoid(self.output(x))
        return x

class Embed(nn.Module):
    def __init__(self, label_dim):
        super(Embed, self).__init__()
        self.dense = nn.Linear(label_dim, 100)

    def forward(self, labelcode):
        x = F.relu(self.dense(labelcode.view(labelcode.size(0), -1)))
        x = x.unsqueeze(2)
        x = x.unsqueeze(3)
        return x

if __name__ == '__main__':
    E = Embed(LABEL_DIM).cuda()
    G = Generator(100).cuda()
    D = Discriminator(100).cuda()
    summary(E, (LABEL_DIM, 1, 1))
    summary(G, [(NOISE_DIM, 1, 1), (100, 1, 1)])
    summary(D, [(3, 64, 64), (100, 1, 1)])
