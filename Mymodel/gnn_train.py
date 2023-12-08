import os
import random
import argparse
import subprocess

from collections import defaultdict
import abcore_data as ab
from focal_loss import FocalLoss

import numpy as np
from kmeans import kmeans

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.loader import NeighborSampler

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, accuracy_score


from tqdm import tqdm


class GraphSAGE(nn.Module):
    def __init__(self, in_dim : tuple[int, int], hidden_dim, out_dim, hidden):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_dim[0], hidden_dim, aggr='mean')
        self.conv2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, edge_attr):
       # x 是 OptPairTensor，包含源节点特征 x[0] 和目标节点特征 x[1]
        x_u = x[0]
        x_i = x[1]

        x_u = self.conv1(x_u, edge_index, edge_attr)
        x_u = F.relu(x_u)

        x_i = self.conv1(x_i, edge_index, edge_attr)
        x_i = F.relu(x_i)

        x_u = self.conv2(x_u)
        x_i = self.conv2(x_i)

        return x_u, x_i

# class LoadDataset(Dataset):
#     def __init__(self, data):
#         self.x = data.u_x[data.train_u].cpu()

#     def __len__(self):
#         return self.x.shape[0]

#     def __getitem__(self, idx):
#         return torch.from_numpy(np.array(self.x[idx])).float(), \
#                torch.from_numpy(np.array(idx))

def pretrain_gnn(model, dataset, epochs):
    print(model)
    optimizer = Adam(model.parameters(), lr=1e-3)
    model.train()
    for epoch in range(epochs):
        x_u, x_i = model((dataset.u_x, dataset.i_x), dataset.train_edge_index, dataset.train_edge_label)
        loss = F.mse_loss(, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    root_path, _ = os.path.split(os.path.abspath(__file__))
    torch.save(model.state_dict(), root_path+'/model/ae_pre_train.pkl')
    torch.cuda.empty_cache()
    
def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def pre_train(dataset, hidden_dim, hidden, pre_gnn_epoch):
    setup_seed(2023)
    model = GraphSAGE((dataset.u_x.shape(1), dataset.u_x.shape(2)), hidden_dim, dataset.u_x.shape(1), hidden=hidden).cuda()
    
    pretrain_gnn(model, dataset, pre_gnn_epoch)