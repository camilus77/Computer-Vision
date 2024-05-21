import torch
import torch.nn as nn
import numpy as np

# Generator network
class Generator(nn.Module):
    def __init__(self,batch_size,input_dim):
        super().__init__()
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim,128)
        self.LRelu = nn.LeakyReLU()
        self.fc2 = nn.Linear(128,1*28*28)
        self.tanH = nn.Tanh()

    # function for forward propogation
    def forward(self,x):
        layer1 = self.LRelu(self.fc1(x))
        layer2 = self.tanH(self.fc2(layer1))
        out =layer2.view(self.batch_size,1,28,28)
        return out

#Discriminator network
class Discriminator(nn.Module):
    def __init__(self,batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.fc1 = nn.Linear(1*28*28,128)
        self.LReLu = nn.LeakyReLU()
        self.fc2 = nn.Linear(128,1)
        self.SigmoidL = nn.Sigmoid()

    # function for forward propogation
    def forward(self,x):
        flat = x.view(self.batch_size,-1)
        layer1 = self.LReLu(self.fc1(flat))
        out = self.SigmoidL(self.fc2(layer1))
        return out.view(-1,1).squeeze(1)