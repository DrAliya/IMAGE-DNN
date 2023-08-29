#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import torch
import torchvision
from torchvision import datasets, models, transforms
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch.utils.data as data_utils
import pandas as pd
import os
from torch.optim import lr_scheduler
import glob
import PIL
from PIL import Image
from torch.utils import data as D
from torch.utils.data.sampler import SubsetRandomSampler
import random
import time
import sys

import lib
import copy

import matplotlib.pyplot as plt
print(torch.__version__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)



# In[2]:


batch_size = 64
validation_ratio = 0.1
random_seed = 10
initial_lr = 0.1
num_epoch = 300


transform = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(112),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
transform_train=transform
transform_validation=transform
transform_test=transform





train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)

#######################################
#######################################
#######################################
# note that we are reducing the data hwew

percentage=5
print("Percentage of data to be used is ",percentage)
num_train=percentage*len(train_dataset)//100
indices=random.sample(list(np.arange(len(train_dataset))),num_train)
# indices = torch.arange(num_train)
print(min(indices),max(indices),len(indices))
tr_1k = data_utils.Subset(train_dataset, indices)
train_dataset=tr_1k


train_loader = data_utils.DataLoader(dataset = train_dataset,
                                           batch_size = batch_size,
                                           shuffle = True)

                                     
                                                                
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)


                   
                                     
test_loader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=0
)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



                                                     
dataset_sizes={}
dataset_sizes["train"]=len(train_dataset)
dataset_sizes["val"]=len(testset)


dataloaders={}
dataloaders["train"]=train_loader
dataloaders["val"]=test_loader

print(dataset_sizes)


# In[3]:


model_ft = models.alexnet(pretrained=True)
# Here the size of each output sample is set to 2.
model_ft.classifier[6] = nn.Linear(4096,10)
model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


# In[4]:


acc=lib.evaluate_model(model_ft,test_loader,dataset_sizes["val"],device)
print("Acc before training",acc)


# In[8]:


num_epochs=5
model_state_path="simple_alexnet_cifar10"+str(num_epochs)+".pt"
model_ft=lib.simple_train(model_ft, criterion, optimizer_ft, exp_lr_scheduler, 
                dataloaders, dataset_sizes, device,num_epochs=num_epochs, model_state_path=model_state_path)



# In[9]:


acc=lib.evaluate_model(model_ft,test_loader,dataset_sizes["val"],device)
print("Acc after training",acc)


# In[ ]:




