# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 12:04:37 2022

@author: srpv
"""

import time
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from torch import nn, Tensor
from typing import Tuple
from utils import *
from Network import *
from Loss import *
from sklearn.model_selection import train_test_split# implementing train-test-split


torch.manual_seed(2020)
np.random.seed(2020)
random.seed(2020)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":
    torch.cuda.get_device_name()
      
embedding_dims = 32
batch_size = 512
epochs = 20

#%%

windowsize= 5000
path = r'C:\Users\srpv\Desktop\RMS Polymer Wear\Data' #path to the data folder.....

classfile = path+'/'+ 'Classlabel'+'.npy'
rawfile = path+'/'+ 'Raw'+'.npy'

classspace= np.load(classfile).astype(np.int64)
classspace = pd.DataFrame(classspace) 
classspace.columns = ['Categorical']

rawspace = np.load(rawfile).astype(np.float64)
rawspace = pd.DataFrame(rawspace)



data=pd.concat([rawspace,classspace], axis=1)
minval = min(data.Categorical.value_counts())
minval=np.round(minval,decimals=-3)    
print("windows of the class: ",minval)
minval=1500
data = pd.concat([data[data.Categorical == cat].head(minval) for cat in data.Categorical.unique() ])  

rawspace=data.iloc[:,:-1]
classspace=data.iloc[:,-1]


Featurespace =rawspace.to_numpy()
classspace =classspace.to_numpy()

#%%

sequences =[]
for i in range(len(classspace)):
    # print(i)
    sequence_features = Featurespace[i]
    label = classspace[i]
    sequences.append((sequence_features,label))
    
#%%

class Mechanism(Dataset):
    
    def __init__(self,sequences):
        self.sequences = sequences
    
    def __len__(self):
        
        return len(self.sequences)
    
    def __getitem__(self,idx):
        sequence,label =  self.sequences [idx]
        sequence=torch.Tensor(sequence)
        sequence = sequence.view(1, -1)
        label=torch.tensor(label).long()
        sequence,label
        return sequence,label

#%%

sequences = Mechanism(sequences) 
train, test = train_test_split(sequences, test_size=0.3,random_state=123)


trainset = torch.utils.data.DataLoader(train, batch_size=100, num_workers=0,
                                                shuffle=True)

testset = torch.utils.data.DataLoader(test, batch_size=100, num_workers=0,
                                                shuffle=True)

print(device)


#%%


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        # print('Learning rate =')
        # print(param_group['lr'])
        return param_group['lr']

  
#%%
         
model = Network(droupout=0.05,emb_dim=embedding_dims)
model.apply(init_weights)
model = torch.jit.script(model).to(device)

optimizer =  torch.optim.SGD(model.parameters(),lr=0.001,momentum=0.9)
scheduler = StepLR(optimizer, step_size = 50, gamma= 0.25 )


# criterion = torch.jit.script(TripletLoss())
criterion = CircleLoss(m=0.25, gamma=25)

model.train()


#%%

Loss_value =[]
Learning_rate=[]


for epoch in range(epochs):
    epoch_smoothing=[]
    learingrate_value = get_lr(optimizer)
    Learning_rate.append(learingrate_value)
    closs = 0
    scheduler.step()
    
    for i,batch in enumerate(trainset,0):
        
        data,output = batch
        data,output = data.to(device,dtype=torch.float),output.to(device,dtype=torch.long)
        output=output.squeeze()
        prediction = model(data) 
        loss = criterion(*convert_label_to_similarity(prediction, output))
        
        closs += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_smoothing.append(closs)
        
        if i%10 == 0:
            print('[%d  %d] loss: %.4f'% (epoch+1,i+1,loss))
            
    loss_train = closs / len(trainset)
    Loss_value.append(loss_train.cpu().detach().numpy())
    
print('Finished Training')

#%%
    
torch.save({"model_state_dict": model.state_dict(),
            "optimzier_state_dict": optimizer.state_dict()
            }, "trained_model.pth")


Loss_value=np.asarray(Loss_value)
Loss_embeddings = 'Loss_value'+'_Triplet'+ '.npy'
np.save(Loss_embeddings,Loss_value, allow_pickle=True)


#%%

plt.rcParams.update({'font.size': 15})
plt.figure(1)
plt.plot(Loss_value,'b',linewidth =2.0)
plt.title('Epochs vs Training value')
plt.xlabel('Epochs')
plt.ylabel('Loss value')
plt.legend(['Training loss'])
plt.savefig('Training loss.png', dpi=600,bbox_inches='tight')
plt.show()


#%%
train_results = []
labels = []

model.eval()
with torch.no_grad():
    for img, label in tqdm(trainset):
        print(img.shape)
        label=label.squeeze()
        print(label.shape)
        train_results.append(model(img.to(device,dtype=torch.float)).cpu().numpy())
        labels.append(label)
        
train_results = np.concatenate(train_results)
train_labels = np.concatenate(labels)
train_results.shape
train_labels.shape

train_embeddings = 'train_embeddings'+'_'+ '.npy'
train_labelsname = 'train_labels'+'_'+'.npy'
np.save(train_embeddings,train_results, allow_pickle=True)
np.save(train_labelsname,train_labels, allow_pickle=True)

test_results = []
labels = []

model.eval()
with torch.no_grad():
    for img,label in tqdm(testset):
        print(img.shape)
        label=label.squeeze()
        print(label.shape)
        test_results.append(model(img.to(device,dtype=torch.float)).cpu().numpy())
        labels.append(label)
        
test_results = np.concatenate(test_results)
test_labels = np.concatenate(labels)
test_results.shape
test_labels.shape


test_embeddings = 'test_embeddings'+'_'+ '.npy'
test_labelsname = 'test_labels'+'_'+'.npy'
np.save(test_embeddings,test_results, allow_pickle=True)
np.save(test_labelsname,test_labels, allow_pickle=True)


graph_name_2D='Training_Feature_2D' +'_'+'.png'
plot_embeddings(train_results, train_labels,graph_name_2D)
graph_name_2D='Testing_Feature_2D' +'_'+'.png'
plot_embeddings(test_results, test_labels,graph_name_2D)



#%%
count_parameters(model)


