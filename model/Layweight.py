import torch
import torch.nn as nn
from torch.nn import init

class NeighborSelection(nn.Module):
    def __init__(self, input_dim, neighbor_count=3):
        super(NeighborSelection, self).__init__()
        self.neighbor_count = neighbor_count
        self.fc = nn.Linear(2 * input_dim, 1)  # 用于计算邻居重要性权重

    def forward(self,result_tensor, node_features, neighbors, node_indices):
        node_mapping = {int(node.item()):i for i,node in enumerate(result_tensor)}
        node_mapping2 = {i:int(node.item()) for i,node in enumerate(result_tensor)}



        selected_neighbors = []

        for i,node_idx in enumerate(node_indices):  # 遍历中心节点列表
            node_neighbors = neighbors[i]  # 获取中心节点的邻居集合
            node_feature = node_features(torch.tensor([node_mapping(node_idx)],device='cuda:0'))  # 1 * feature_dim, 获取中心节点特征

            if len(node_neighbors) <= self.neighbor_count:
                selected_neighbors.append(list(node_neighbors))  # 直接返回邻居
            else:
                neighbor_features = node_features(torch.tensor([node_mapping(nd) for nd in node_neighbors],device='cuda:0'))  # 获取邻居的特征 (num_neighbors, feature_dim)
                concatenated_features = torch.cat([node_feature.repeat(len(node_neighbors), 1), neighbor_features], dim=1)  # (num_neighbors, 2 * feature_dim)
                importance_scores = self.fc(concatenated_features)  # (num_neighbors, 1)
                importance_scores = importance_scores.squeeze()  # (num_neighbors,)
                importance_scores = torch.exp(torch.nn.functional.leaky_relu(importance_scores))
                _, top_indices = torch.topk(importance_scores, self.neighbor_count, largest=True, sorted=False)
                selected_neighbors_for_node = {list(node_neighbors)[i] for i in top_indices.tolist()}
                selected_neighbors.append(selected_neighbors_for_node)
        return selected_neighbors



class NodeLayWeight(nn.Module):
    def __init__(self,features_nn):
        super(NodeLayWeight, self).__init__()
        self.features_nn = features_nn
        self.u = nn.Parameter(torch.FloatTensor((self.features_nn(torch.LongTensor([0])).shape[1])*2,1))
        init.xavier_uniform_(self.u)

        self.LeakReLU = nn.LeakyReLU(0.1)

    def forward(self, node,to_neigh,num_sample):
        if type(node)==torch.Tensor:
            node = int(node.item())
        node_neigh = list(to_neigh)
        node_neigh_fea = self.features_nn(torch.LongTensor(list(to_neigh)))
        node_center_fea = self.features_nn(torch.LongTensor([node])).repeat(node_neigh_fea.shape[0],1)
        weight = torch.exp(self.LeakReLU(torch.matmul(torch.cat([node_neigh_fea,node_center_fea],1),self.u)))
        #weight = torch.exp(torch.matmul(torch.cat([node_neigh_fea,node_center_fea],1),self.u))
        weight = weight / sum(weight)
        selectV_I = torch.topk(weight, num_sample, 0, largest=True, sorted=False)
        neigh_index = selectV_I.indices.reshape(1,-1).squeeze().tolist()
        node_neigh = {node_neigh[i] for i in neigh_index}
        return node_neigh
        '''
        print(node_center_fea.shape)
        print(node_neigh_fea.shape)
        torch.Size([1, 128])
        torch.Size([8, 128])
        '''
