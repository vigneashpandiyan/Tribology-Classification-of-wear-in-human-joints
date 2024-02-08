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


class PrintLayer(torch.nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        # Do your print / debug stuff here
        print(x.shape)
        return x


class Network(nn.Module):
    def __init__(self, droupout, emb_dim):
        super(Network, self).__init__()
        #torch.Size([100, 1, 5000])
        self.dropout = droupout

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=16),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.MaxPool1d(3),
            # PrintLayer(),

            nn.Conv1d(in_channels=4, out_channels=8, kernel_size=16),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.MaxPool1d(3),
            # PrintLayer(),

            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.MaxPool1d(3),
            # PrintLayer(),

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=16),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.MaxPool1d(3),
            # PrintLayer(),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=16),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.MaxPool1d(3),
            # PrintLayer(),

        )

        self.fc = nn.Sequential(
            nn.Linear(64*13, 64),
            nn.ReLU(),
            nn.Linear(64, emb_dim),

            # PrintLayer(),

        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 832)
        x = self.fc(x)

        return nn.functional.normalize(x)
