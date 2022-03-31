# _*_ coding: utf-8 _*_
"""
Time:     2022-03-31 8:36
Author:   Haolin Yan(XiDian University)
File:     mlp.py.py
"""
import torch
import torch.nn as nn
import torchsort


class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 regularization="l2",
                 regularization_strength=1.0,
                 ):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                nn.ReLU(),
                                nn.Dropout(),
                                nn.Linear(hidden_dim, hidden_dim),
                                nn.ReLU(),
                                nn.Dropout(),
                                nn.Linear(hidden_dim, output_dim))
        self.sort_rank = lambda x:torchsort.soft_rank(x,
                                                 regularization=regularization,
                                                 regularization_strength=regularization_strength)

    def forward(self, x):
        scores = self.fc(x)  # (bsize, output_dim)
        rank = self.sort_rank(scores.T)
        return rank, scores



