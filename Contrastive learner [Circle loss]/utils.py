# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 21:57:22 2024

@author: srpv
contact: vigneashwara.solairajapandiyan@empa.ch


The codes in this following script will be used for the publication of the following work

"Classification of Progressive Wear on a Multi-Directional Pin-on-Disc 
Tribometer Simulating Conditions in Human Joints-UHMWPE against CoCrMo 
Using Acoustic Emission and Machine Learning"

@any reuse of this code should be authorized by the first owner, code author
"""
# %%
# Libraries to import

import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable
import torch.nn as nn
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        torch.nn.init.kaiming_normal_(m.weight)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        # print('Learning rate =')
        # print(param_group['lr'])
        return param_group['lr']


class Mechanism(Dataset):

    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):

        return len(self.sequences)

    def __getitem__(self, idx):
        sequence, label = self.sequences[idx]
        sequence = torch.Tensor(sequence)
        sequence = sequence.view(1, -1)
        label = torch.tensor(label).long()
        sequence, label
        return sequence, label


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


colors = ['green', 'red', 'blue', 'cyan', 'orange', 'purple', 'black', 'brown']
marker = ["*", ">", "X", "o", "s", "P", "+", "_"]
color = ['g', 'r', 'orange', 'b', 'purple']
mnist_classes = ['Class 1', 'Class 2', 'Class 3',
                 'Class 4', 'Class 5', 'Class 6', 'Class 7', 'Class 8']
graph_title = "Feature space distribution"


def plot_embeddings(embeddings, targets, graph_name_2D, xlim=None, ylim=None):
    plt.figure(figsize=(7, 5))
    count = 0
    for i in np.unique(targets):
        inds = np.where(targets == i)[0]
        plt.scatter(embeddings[inds, 0], embeddings[inds, 1], alpha=0.7,
                    color=colors[count], marker=marker[count], s=100)
        count = count+1
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(mnist_classes, bbox_to_anchor=(1.32, 1.05))
    plt.xlabel('Dimension 1', labelpad=10)
    plt.ylabel('Dimension 2', labelpad=10)
    plt.title(str(graph_title), fontsize=15)
    plt.savefig(graph_name_2D, bbox_inches='tight', dpi=600)
    plt.show()


def TSNEplot(output, target, perplexity):

    # array of latent space, features fed rowise

    output = np.array(output)
    target = np.array(target)

    print('target shape: ', target.shape)
    print('output shape: ', output.shape)
    print('perplexity: ', perplexity)

    group = target
    group = np.ravel(group)

    RS = np.random.seed(1974)
    tsne = TSNE(n_components=3, random_state=RS, perplexity=perplexity)
    tsne_fit = tsne.fit_transform(output)

    return tsne_fit, target, tsne


def dataprocessing(df):
    database = df
    print(database.shape)
    database = database.apply(lambda x: (x - np.mean(x))/np.std(x), axis=1)
    # anomaly_database=anomaly_database.to_numpy().astype(np.float64)
    return database
