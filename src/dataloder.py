from torch.utils.data import Dataset,DataLoader
import torch
class CustomDataset(Dataset):
    def __init__(self, idx, labels, node_num):
        self.idx = idx  # 存储所有的样本索引
        self.labels = labels  # 存储样本的标签
        self.node_num = node_num  # 节点总数

    def __len__(self):
        return len(self.idx)  # 数据集大小

    def __getitem__(self, idx):
        ids = self.idx[idx]
        targets = self.labels[ids]
        row, col = divmod(ids, self.node_num)
        node_list_batch = torch.tensor([row, col])
        return [node_list_batch, targets]
def construct_loader(idx, label, layer_num, num_nodes,batch_num):
    data_loaders, max_len = [], 0
    for l in range(layer_num):
        dataset_loader = CustomDataset(idx[l], label[l], num_nodes)
        data_loader = DataLoader(dataset_loader, batch_size=batch_num, shuffle=False)
        data_loaders.append(data_loader)
        if max_len < len(data_loader):
            max_len = len(data_loader)
    return data_loaders, max_len