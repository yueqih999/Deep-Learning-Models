import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
import matplotlib.pyplot as plt

path = 'VAE/data'
train_data=MNIST(path, train=True,download=False,transform=transforms.ToTensor())
test_data=MNIST(path, train=False,download=False,transform=transforms.ToTensor())

train_data_loader=DataLoader(train_data,batch_size=64,shuffle=True,num_workers=0)
test_data_loader=DataLoader(test_data,batch_size=64,shuffle=True, num_workers=0)

