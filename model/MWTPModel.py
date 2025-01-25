import sys
import time

import torch
import torch.nn as nn
import numpy as np

from .MultiLayer import SemanticAttention1, BitwiseMultipyLogis1
from .aggregators import MeanAggregator
from .encoders import Encoder
from sklearn.metrics import roc_auc_score, f1_score,average_precision_score

from utils import accuracy, LogisticRegression


class MWTPModel(nn.Module):
    def __init__(self,layer_num, num_nodes, feat_data, adj_lists, emb_dim,device,inter_aggregation):
        super(MWTPModel, self).__init__()
        self.layer_num = layer_num
        self.num_nodes = num_nodes
        self.feat_data = feat_data
        self.adj_lists = adj_lists
        self.emb_dim = emb_dim
        self.device = device
        self.num_samples1 = 10
        self.num_samples2 = 10
        self.inter_aggregation = inter_aggregation
        self.generate_aggregate()

        self.MWTP = SupervisedGraphSage(self.enc_two, emb_dim, layer_num,device,inter_aggregation)
        self.MWTP.to(device)
        self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.MWTP.parameters()), lr=0.7)

    def forward(self, nodes, targets,layer_predict,run_type='train'):
        # t1 = time.time()
        predict = self.MWTP(nodes,layer_predict)
        # print('time predict:',time.time()-t1)
        # t1 = time.time()
        loss = self.MWTP.loss(predict,targets)
        # print('time loss:', time.time() - t1)
        # t1 = time.time()
        acc = self.MWTP.acc(predict,targets)
        # print('time acc:', time.time() - t1)
        if run_type == 'valid':
            return loss, acc
        # t1 = time.time()
        auc = self.MWTP.Auc(predict,targets)
        # print('time auc:', time.time() - t1)
        # t1 = time.time()
        ap = self.MWTP.ap(predict,targets)
        # print('time ap:', time.time() - t1)
        # t1 = time.time()
        f1 = self.MWTP.f1(predict,targets)
        # print('time f1:', time.time() - t1)
        return loss, acc, auc, ap, f1

    def generate_aggregate(self):
        self.agg_one = nn.ModuleList([]).to(self.device) # 放在哪个设备
        self.agg_two = nn.ModuleList([]).to(self.device)
        self.enc_one = nn.ModuleList([]).to(self.device)
        self.enc_two = nn.ModuleList([]).to(self.device)
        self.features = nn.ModuleList([]).to(self.device)
        for l in range(self.layer_num):
            self.features.append(nn.Embedding(self.num_nodes, self.feat_data[l].shape[1]))
            self.features[l].weight = nn.Parameter(torch.tensor(self.feat_data[l], dtype=torch.float, device=self.device), requires_grad=False)
            self.agg_one.append(MeanAggregator(self.features[l], self.device))
            self.enc_one.append(
                Encoder(self.features[l], self.feat_data[l].shape[1], 128, self.adj_lists[l], self.agg_one[l],
                        device_type=self.device,gcn=True))

            self.agg_two.append(MeanAggregator(lambda nodes: self.enc_one[l](nodes).t(), self.device))
            self.enc_two.append(
                Encoder(lambda nodes2: self.enc_one[l](nodes2).t(), self.enc_one[l].embed_dim, self.emb_dim,
                        self.adj_lists[l], self.agg_two[l],
                        base_model=self.enc_one[l],device_type=self.device,gcn=True))
            self.enc_one[l].num_sample = self.num_samples1
            self.enc_two[l].num_sample = self.num_samples2
    # def loss(self,):

class SupervisedGraphSage(nn.Module):

    def __init__(self, enc, embed_dim, layer_num, device, inter_aggregation):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.embed_dim = embed_dim
        self.layer_num = layer_num
        self.device = device
        self.criterion = nn.BCELoss()
        self.accuracy = accuracy
        self.logis = nn.ModuleList(LogisticRegression(self.embed_dim, 1,self.device) for _ in range(layer_num))
        if inter_aggregation == 'semantic':
            self.layerNodeAttention_weight = SemanticAttention1(self.embed_dim,device, dropout=0.5, alpha=0.5).to(device)
        elif inter_aggregation == 'logit':
            self.layerNodeAttention_weight = BitwiseMultipyLogis1(self.embed_dim,device).to(device)

    def forward(self, nodes,layer_predict):
        embeds = []
        for l in range(self.layer_num):
            embeds.append(self.enc[l](nodes).t())
        result = self.layerNodeAttention_weight(torch.stack(embeds),layer_predict)
        predict = self.logis[layer_predict](result)
        return predict

    def loss(self, predict, targets):
        return self.criterion(predict, targets.to(self.device))
    def acc(self, predict, targets):
        return self.accuracy(predict, targets.to(self.device))
    def Auc(self, predict, targets):
        return roc_auc_score(targets, predict.cpu().detach().numpy())
    def ap(self, predict, targets):
        return average_precision_score(targets, predict.cpu().detach().numpy())
    def f1(self, predict, targets, threshold=0.5):
        predict_np = predict.cpu().detach().numpy()
        threshold = np.median(predict_np)
        binary_predictions = (predict_np >= threshold).astype(int)
        return f1_score(targets, binary_predictions)