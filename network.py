import numpy as py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import cv2 as cv

class MaskNet(nn.Module):
    def __init__(self):
        super(MaskNet,self).__init__()
        self.encoder1 = nn.Sequential(
            
            nn.Conv2d(3,8,3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            #nn.MaxPool2d(2,2)
        )
        self.encoder2 = nn.Sequential(
            #nn.MaxPool2d(2,2)
            nn.Conv2d(8,16,3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.encoder3 = nn.Sequential(
            #nn.MaxPool2d(2,2)
            nn.Conv2d(16,32,3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.encoder4 = nn.Sequential(
            nn.Conv2d(32,64,3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        #self.unpool = nn.MaxUnpool2d(2,2,padding=0)
        self.decoder1 = nn.Sequential(
            #nn.MaxUnpool2d(2,2)           
            nn.ConvTranspose2d(64,32,4,stride=2,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(32,16,4,stride=2,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(16,8,4,stride=2,padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(8,3,4,stride=2,padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            
        )
        #self.pool = nn.MaxPool2d(2,2)
        self.upsample = nn.Upsample((320,320))
        
    def forward(self,x):
        #print(x.size())
        a = self.encoder1(x)
        b = self.encoder2(a)
        c = self.encoder3(b)
        d = self.encoder4(c)
        #print(x.size())
        dt = self.decoder1(d)+c
        ct = self.decoder2(dt)+b
        bt = self.decoder3(ct)+a
        x = self.decoder4(bt)
        x = self.upsample(x)
        #print(x.size())
        return x

