"""

    Created on 03/06/21 10:07 AM
    @author: Kartik Prabhu

"""
from torch import nn
import os
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from PIL import Image

import Baseconfig
import transform_utils


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        #Encoder

        self.enc1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 512, kernel_size=3),
            torch.nn.BatchNorm2d(512),
            torch.nn.ELU(),
        )
        self.enc2 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3),
            torch.nn.BatchNorm2d(512),
            torch.nn.ELU(),
            torch.nn.MaxPool2d(kernel_size=3)
        )
        self.enc3 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 256, kernel_size=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ELU()
        )

        self.enc4 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 128, kernel_size=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ELU()
        )

        self.enc5 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 64, kernel_size=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ELU()
        )


    def forward(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))
        x = F.relu(self.enc5(x))

        x = torch.flatten(x)
        print(x.shape)
        return x



class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        #Decoder
        self.dec1 = nn.Linear(in_features=16, out_features=32)
        self.dec2 = nn.Linear(in_features=32, out_features=64)
        self.dec3 = nn.Linear(in_features=64, out_features=128)
        self.dec4 = nn.Linear(in_features=128, out_features=256)
        self.dec5 = nn.Linear(in_features=256, out_features=224*224*3) # Output image (28*28 = 784)

    def forward(self, x):
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        x = F.relu(self.dec4(x))
        x = F.relu(self.dec5(x))
        return x

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        encoded_output = self.encoder(x)
        decoded_output =self.decoder(encoded_output)

        return decoded_output,encoded_output


if __name__ == '__main__':
    config = Baseconfig.config
