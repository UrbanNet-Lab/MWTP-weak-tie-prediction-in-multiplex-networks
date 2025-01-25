import sys
import time

import torch
import torch.nn as nn
from torch.autograd import Variable

import random


"""
Set of modules for aggregating embeddings of neighbors.
"""

class MeanAggregator(nn.Module):
    def __init__(self, features, device, gcn=True):

        super(MeanAggregator, self).__init__()

        self.features = features
        self.device = device
        self.gcn = gcn
        
    def forward(self, nodes, to_neighs,num_sample=5,nodeLayWeight = None):
        # _set = set
        # if not num_sample is None:
        #     _sample = random.sample
        #     samp_neighs = [_set(_sample(list(to_neigh),
        #                     num_sample,
        #                     )) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        #     # samp_neighs = [nodeLayWeight(nodes[kk],to_neigh,num_sample) if len(to_neigh) >= num_sample else to_neigh for kk,to_neigh in enumerate(to_neighs)]
        # else:
        #     samp_neighs = to_neighs
        #
        # if self.gcn:
        #     samp_neighs = [samp_neigh.union( set([int(nodes[i].item())])) for i, samp_neigh in enumerate(samp_neighs)]
        # unique_nodes_list = list(set.union(*samp_neighs))
        # unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)}
        # mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes), device=self.device))
        # column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        # row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        # mask[row_indices, column_indices] = 1
        # num_neigh = mask.sum(1, keepdim=True)
        # mask = mask.div(num_neigh)
        # embed_matrix = self.features(torch.tensor(unique_nodes_list, dtype=torch.long, device=self.device))
        # to_feats = mask.mm(embed_matrix)
        # return to_feats

        # _set = set
        # # 高效采样，避免每次转换为set
        # if num_sample is not None:
        #     samp_neighs = []
        #     for to_neigh in to_neighs:
        #         if len(to_neigh) >= num_sample:
        #             # 使用集合的快速采样，避免多次list转换
        #             sampled_neighs = random.sample(list(to_neigh), num_sample)
        #             samp_neighs.append(set(sampled_neighs))
        #         else:
        #             samp_neighs.append(to_neigh)
        # else:
        #     samp_neighs = to_neighs
        #
        # # 如果启用了GCN，将每个节点的邻居加入集合
        # if self.gcn:
        #     for i, samp_neigh in enumerate(samp_neighs):
        #         samp_neigh.add(int(nodes[i].item()))  # 使用集合的add方法
        #
        # # 使用集合的合并操作生成unique_nodes
        # unique_nodes_set = set.union(*samp_neighs)
        # unique_nodes_list = list(unique_nodes_set)
        # unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
        #
        # # 直接生成稀疏矩阵，减少不必要的内存开销
        # row_indices = []
        # column_indices = []
        # for i, samp_neigh in enumerate(samp_neighs):
        #     for n in samp_neigh:
        #         row_indices.append(i)
        #         column_indices.append(unique_nodes[n])
        #
        # # 创建稀疏mask矩阵
        # mask = torch.zeros(len(samp_neighs), len(unique_nodes), device=self.device)
        # mask[row_indices, column_indices] = 1
        #
        # # 归一化操作
        # num_neigh = mask.sum(1, keepdim=True)
        # mask = mask.div(num_neigh)
        #
        # # 获取嵌入矩阵并计算特征
        # embed_matrix = self.features(torch.tensor(unique_nodes_list, dtype=torch.long, device=self.device))
        # to_feats = mask.mm(embed_matrix)  # 使用矩阵乘法获取最终特征
        #
        # return to_feats

        _set = set
        # 高效采样，避免每次转换为set
        if num_sample is not None:
            samp_neighs = []
            for to_neigh in to_neighs:
                if len(to_neigh) >= num_sample:
                    # 使用集合的快速采样，避免多次list转换
                    sampled_neighs = random.sample(list(to_neigh), num_sample)
                    samp_neighs.append(set(sampled_neighs))
                else:
                    samp_neighs.append(to_neigh)
        else:
            samp_neighs = to_neighs

        # 如果启用了GCN，将每个节点的邻居加入集合
        if self.gcn:
            for i, samp_neigh in enumerate(samp_neighs):
                samp_neigh.add(int(nodes[i].item()))  # 使用集合的add方法
        # 使用集合的合并操作生成unique_nodes
        unique_nodes_set = set.union(*samp_neighs)
        unique_nodes_list = list(unique_nodes_set)
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}

        # 直接生成稀疏矩阵，减少不必要的内存开销
        row_indices = []
        column_indices = []
        for i, samp_neigh in enumerate(samp_neighs):
            for n in samp_neigh:
                row_indices.append(i)
                column_indices.append(unique_nodes[n])
        mask = torch.sparse_coo_tensor(
            torch.tensor([row_indices, column_indices], device=self.device),
            torch.ones(len(row_indices), device=self.device),
            torch.Size([len(samp_neighs), len(unique_nodes_list)]),
            device=self.device
        ).to_dense()

        # 归一化操作
        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh)
        # 获取嵌入矩阵并计算特征
        embed_matrix = self.features(torch.tensor(unique_nodes_list, dtype=torch.long, device=self.device))
        # to_feats = mask.mm(embed_matrix)  # 使用矩阵乘法获取最终特征
        to_feats = torch.matmul(mask,embed_matrix)  # 使用矩阵乘法获取最终特征
        return to_feats

class AggregatorMixin(object):
    @property
    def output_dim(self):
        tmp = torch.zeros((1, self.output_dim_))
        return self.combine_fn([tmp, tmp]).size(1)

        
class PoolAggregator(nn.Module, AggregatorMixin):
    def __init__(self, input_dim, output_dim, pool_fn, activation, hidden_dim=512, combine_fn=lambda x: torch.cat(x, dim=1)):
        super(PoolAggregator, self).__init__()
        
        self.mlp = nn.Sequential(*[
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.ReLU()
        ])
        self.fc_x = nn.Linear(input_dim, output_dim, bias=False)
        self.fc_neib = nn.Linear(hidden_dim, output_dim, bias=False)
        
        self.output_dim_ = output_dim
        self.activation = activation
        self.pool_fn = pool_fn
        self.combine_fn = combine_fn
    
    def forward(self, x, neibs):
        h_neibs = self.mlp(neibs)
        agg_neib = h_neibs.view(x.size(0), -1, h_neibs.size(1))
        agg_neib = self.pool_fn(agg_neib)
        
        out = self.combine_fn([self.fc_x(x), self.fc_neib(agg_neib)])
        if self.activation:
            out = self.activation(out)
        
        return out


class MaxPoolAggregator(PoolAggregator):
    def __init__(self, input_dim, output_dim, activation, hidden_dim=512, combine_fn=lambda x: torch.cat(x, dim=1)):
        super(MaxPoolAggregator, self).__init__(**{
            "input_dim" : input_dim,
            "output_dim" : output_dim,
            "pool_fn" : lambda x: x.max(dim=1)[0],
            "activation" : activation,
            "hidden_dim" : hidden_dim,
            "combine_fn" : combine_fn,
        })


class MeanPoolAggregator(PoolAggregator):
    def __init__(self, input_dim, output_dim, activation, hidden_dim=512, combine_fn=lambda x: torch.cat(x, dim=1)):
        super(MeanPoolAggregator, self).__init__(**{
            "input_dim" : input_dim,
            "output_dim" : output_dim,
            "pool_fn" : lambda x: x.mean(dim=1),
            "activation" : activation,
            "hidden_dim" : hidden_dim,
            "combine_fn" : combine_fn,
        })


class LSTMAggregator(nn.Module, AggregatorMixin):
    def __init__(self, input_dim, output_dim, activation, 
        hidden_dim=512, bidirectional=False, combine_fn=lambda x: torch.cat(x, dim=1)):
        
        super(LSTMAggregator, self).__init__()
        assert not hidden_dim % 2, "LSTMAggregator: hiddem_dim % 2 != 0"
        
        self.lstm = nn.LSTM(input_dim, hidden_dim // (1 + bidirectional), bidirectional=bidirectional, batch_first=True)
        self.fc_x = nn.Linear(input_dim, output_dim, bias=False)
        self.fc_neib = nn.Linear(hidden_dim, output_dim, bias=False)
        
        self.output_dim_ = output_dim
        self.activation = activation
        self.combine_fn = combine_fn
    
    def forward(self, x, neibs):
        print (x)
        x_emb = self.fc_x(x)
        
        agg_neib = neibs.view(x.size(0), -1, neibs.size(1))
        agg_neib, _ = self.lstm(agg_neib)
        agg_neib = agg_neib[:,-1,:] # !! Taking final state, but could do something better (eg attention)
        neib_emb = self.fc_neib(agg_neib)
        
        out = self.combine_fn([x_emb, neib_emb])
        if self.activation:
            out = self.activation(out)
        
        return out
