#!/usr/bin/env python
#coding:utf-8
#author:zhujianqi

from __future__ import print_function
import sys

import torch
from torch import nn


class LinearModel(nn.Module):
    def __init__(self, 
                 vocab_size,
                 embed_dim,
                 hidden_dim,
                 embeddings=None,
                 dropout=0.5,
                 num_classes=2,
                 padding_idx=1,
                 device='cpu'):
        super(LinearModel, self).__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout = dropout
        self.device = device

        self.embedding = nn.Embedding(self.vocab_size,
                                      self.embed_dim,
                                      padding_idx=padding_idx,
                                      _weight=embeddings)

        self.projection = nn.Sequential(nn.Dropout(self.dropout),
                                        nn.Linear(4*self.embed_dim, self.hidden_dim),
                                        nn.ReLU(),
                                        nn.Dropout(self.dropout),
                                        nn.Linear(self.hidden_dim, self.num_classes))

        self.output_active = nn.Softmax(dim=1)
        self.apply(self._init_model_weights)

    def forward(self, text1, len1, text2, len2):
        batch_size = text1.shape[0]
        embedded1 = torch.sum(self.embedding(text1), dim=1) / len1.view(batch_size, 1)
        embedded2 = torch.sum(self.embedding(text2), dim=1) / len2.view(batch_size, 1)
        input = torch.cat([embedded1, embedded2, torch.abs(embedded1-embedded2), embedded1*embedded2], 1)

        logit = self.projection(input)
        prob = self.output_active(logit)
        return logit, prob

    def _init_model_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight.data)
            nn.init.constant_(module.bias.data, 0.0)


