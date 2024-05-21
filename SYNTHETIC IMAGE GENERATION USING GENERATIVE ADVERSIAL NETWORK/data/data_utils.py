#importing required libraries
import torch

# MNIST data from Torch
from torchvision.datasets import MNIST 

# to load the data
from torch.utils.data import DataLoader

# to transform the data
from torchvision import transforms


# function to transform and load the data with Torch
def get_dl(batchsize):

    # to transform the data as Tensors
    train_transforms = transforms.Compose([transforms.ToTensor()])
    
    # downloading train and test MNIST Data and transforming it to tensors
    train_data = MNIST(root="./train.",train=True,download=True,transform=train_transforms)
    test_data = MNIST(root="./test.",train=True,download=True,transform=train_transforms)
    
    # loading the data
    train_loader = DataLoader(train_data,batch_size=batchsize,shuffle=False,drop_last=True)
    test_loader = DataLoader(test_data,batch_size=batchsize,shuffle=False,drop_last=True)
    
    # returning the train and test data 
    return train_loader,test_loader