import sys
import time

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    """
    def __init__(self, features, feature_dim, 
            embed_dim, adj_lists, aggregator,
            num_sample=5,
            base_model=None, device_type=torch.device("cpu"), gcn=False):
        super(Encoder, self).__init__()
        self.features = features
        self.num_sample = num_sample
        self.feat_dim = feature_dim
        self.adj_lists = adj_lists
        self.aggregator = aggregator
        if base_model != None:
            self.base_model = base_model

        self.gcn = gcn
        self.embed_dim = embed_dim
        self.device = device_type
        self.weight = nn.Parameter(
                torch.FloatTensor(embed_dim,2 * self.feat_dim if self.gcn else self.feat_dim))
        init.xavier_uniform_(self.weight)

    def forward(self, nodes):
        neigh_feats = self.aggregator.forward(nodes, [self.adj_lists[int(node)] for node in nodes] ,num_sample=self.num_sample)
        if self.gcn:
            self_feats = self.features(nodes.to(self.device))
            combined = torch.cat([self_feats, neigh_feats], dim=1)
        else:
            combined = neigh_feats
        # combined = F.relu(self.weight.mm(combined.t()))
        combined = F.relu(torch.matmul(self.weight,combined.t()))
        return combined
