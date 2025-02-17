import torch
import torch.nn as nn

class NeighborSelection(nn.Module):
    def __init__(self, input_dim, neighbor_count=3):
        super(NeighborSelection, self).__init__()
        self.neighbor_count = neighbor_count
        self.fc = nn.Linear(2 * input_dim, 1)  # 用于计算邻居重要性权重

    def forward(self, node_features, neighbors, node_indices):
        """
        node_features: 节点特征矩阵 (num_nodes, feature_dim)，每个节点的特征
        neighbors: 每个节点的邻居列表，类型为set，例如 [{2,3,4,5,6}, {1,4,5,6,7}, ...]
        feature_matrix: 节点特征矩阵 (num_nodes, feature_dim)，每个节点的特征
        node_indices: 中心节点的索引列表 (list)，例如 [0, 1, 2, ...]
        """
        selected_neighbors = []

        for node_idx in node_indices:  # 遍历中心节点列表
            node_neighbors = neighbors[node_idx]  # 获取中心节点的邻居集合
            node_feature = node_features(node_idx).unsqueeze(0)  # 1 * feature_dim, 获取中心节点特征

            # 如果邻居数量小于3，直接返回所有邻居
            if len(node_neighbors) <= self.neighbor_count:
                selected_neighbors.append(list(node_neighbors))  # 直接返回邻居
            else:
                # 否则，计算邻居的重要性
                neighbor_features = node_features[list(node_neighbors)]  # 获取邻居的特征 (num_neighbors, feature_dim)

                # 拼接节点特征与邻居特征
                concatenated_features = torch.cat([node_feature.repeat(len(node_neighbors), 1), neighbor_features], dim=1)  # (num_neighbors, 2 * feature_dim)

                # 计算邻居重要性权重
                importance_scores = self.fc(concatenated_features)  # (num_neighbors, 1)

                # 将importance_scores转换为一维向量
                importance_scores = importance_scores.squeeze()  # (num_neighbors,)

                # 使用LeakyReLU激活并进行exp处理
                importance_scores = torch.exp(torch.nn.functional.leaky_relu(importance_scores))

                # 选择重要性最高的邻居
                _, top_indices = torch.topk(importance_scores, self.neighbor_count, largest=True, sorted=False)

                # 选择对应的邻居
                selected_neighbors_for_node = {list(node_neighbors)[i] for i in top_indices.tolist()}
                selected_neighbors.append(selected_neighbors_for_node)

        return selected_neighbors


# 示例数据
# node_features = torch.randn(10, 128)  # 10个节点，每个节点128维特征
# neighbors = [{2, 3, 4, 5, 6}, {1, 4, 5, 6, 7}, {2, 4}, {0, 1, 2, 5}, {0, 2}, {1, 5}, {2, 4}, {3, 7}, {1, 6}, {0, 3}]  # 每个节点的邻居集合
# node_indices = [0, 1, 2, 3]  # 选择节点0, 1, 2, 3作为中心节点进行邻居选择
#
# # 创建模型
# model = NeighborSelection(input_dim=128, neighbor_count=3)
#
# # 获取每个节点的最终邻居
# selected_neighbors = model(node_features, neighbors, node_indices)
#
# # 输出每个节点选出的邻居
# print(selected_neighbors)
