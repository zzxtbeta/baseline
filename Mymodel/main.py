#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
@author: guyongrui
@file: main.py
@time: 2023/12/06
"""


import os
import sys
import time
import random
import argparse
import subprocess

from collections import defaultdict
import abcore_data as ab
from focal_loss import FocalLoss

import numpy as np
from kmeans import kmeans
from torch.nn.parameter import Parameter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.loader import NeighborSampler
from torch_sparse import SparseTensor



from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, accuracy_score


from tqdm import tqdm

import logging



def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(GraphSAGE, self).__init__()
        # in_dim = [64, 64]
        self.conv1 = SAGEConv(in_dim[0], hidden_dim, aggr='mean')
        self.conv2 = SAGEConv(hidden_dim, out_dim, aggr='mean')

    def forward(self, x, edge_index, edge_index_min):
       # x 是 OptPairTensor，包含源节点特征和目标节点特征
        x_u = x[0]  # [3286, 64]
        x_i = x[1]  # [3754, 64]

        x_u = self.conv1(x_u, edge_index_min)
        x_u = F.relu(x_u)

        x_i = self.conv1(x_i, edge_index)
        x_i = F.relu(x_i)

        x_u = self.conv2(x_u, edge_index_min)
        x_i = self.conv2(x_i, edge_index)

        return x_u, x_i
    
class GCN_1(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim,)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index, edge_weight):
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x   
    
class GCN_2(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim,)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index, edge_weight):
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x 


class GNN(nn.Module):
    def __init__(self, u_x_dim, i_x_dim, n_clusters, hidden_dim, hidden_layer, dataset):
        super(GNN, self).__init__()
        self.sage = GraphSAGE((u_x_dim, i_x_dim), hidden_dim, u_x_dim)
        self.gcn_1 = GCN_1(u_x_dim, hidden_dim, u_x_dim)
        self.gcn_2 = GCN_2(i_x_dim, hidden_dim, i_x_dim)

        self.cluster_layer = Parameter(torch.Tensor(n_clusters, u_x_dim + i_x_dim))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def forward(self, dataset):
        print("")



def run(dataset):
    global model
    global optimizer
    model = GNN(
            u_x_dim=dataset.u_x.shape[1],
            i_x_dim=dataset.i_x.shape[1],
            n_clusters=args.n_clusters,
            hidden_dim = args.hidden_dim,
            hidden_layer = 1,
            dataset=dataset,
            ).to(device)
    print(model)
    print(args)

    optimizer = Adam(model.parameters(), lr=args.lr)

    # max_node_num = max(dataset.max_u, dataset.max_i)
    # min_node_num = min(dataset.max_u, dataset.max_i)

    # # A edg
    # sparse_matrix_max_1 = SparseTensor(row=dataset.train_edge_index[0], col=dataset.train_edge_index[1], value=dataset.train_edge_label, sparse_sizes=(max_node_num, max_node_num))
    # sparse_matrix_min_1 = SparseTensor(row=dataset.train_edge_index[0], col=dataset.train_edge_index[1], value=dataset.train_edge_label, sparse_sizes=(min_node_num, min_node_num))
    # # A2
    # sparse_matrix_max_2 = SparseTensor(row=dataset.train_edge_index[0], col=dataset.train_edge_index[1], value=torch.where(dataset.train_edge_label == -1, torch.tensor(1), dataset.train_edge_label), sparse_sizes=(max_node_num, max_node_num))
    # sparse_matrix_min_2 = SparseTensor(row=dataset.train_edge_index[0], col=dataset.train_edge_index[1], value=torch.where(dataset.train_edge_label == -1, torch.tensor(1), dataset.train_edge_label), sparse_sizes=(min_node_num, min_node_num))

    # # print(sparse_matrix_max_2)
    # # print(sparse_matrix_max_2.sizes())
    # # print(sparse_matrix_max_2.to_dense())

    # # AWX
    # u_x_a, i_x_a = model.sage((dataset.u_x, dataset.i_x), sparse_matrix_max_1, sparse_matrix_min_1) # [3286,64]  [3754,64]
    u_x_a= model.gcn_1(dataset.u_x, dataset.train_edge_index, dataset.train_edge_label.float())
    i_x_a= model.gcn_1(dataset.i_x, dataset.train_edge_index, dataset.train_edge_label.float())

    # # A2WX

    # u_x_a2, i_x_a2 = model.sage((dataset.u_x, dataset.i_x), sparse_matrix_max_2, sparse_matrix_min_2) # [3286,64]  [3754,64]+
    u_x_a2 = model.gcn_2(dataset.u_x, dataset.train_edge_index, dataset.train_edge_label.float())
    i_x_a2 = model.gcn_2(dataset.i_x, dataset.train_edge_index, dataset.train_edge_label.float())

    # 拼接得到用户特征
    u_emb = torch.cat((u_x_a, u_x_a2), dim=1)
    # 攻击者特征
    u_atk_emb = u_emb[dataset.u_atk_idx,:]


    """
    NeighborSampler:
    # 每一层采样返回结果： (edge_index, e_id, size); L 层采样完成后，返回结果：(batch_size, n_id, adjs)
    # batch_size就是mini-batch的节点数目；
    # n_id：L层采样中遇到的所有的节点的list，其中target节点在list最前端；
    # adjs：第L层到第1层采样结果的list, 包含(edge_index, e_id, size)
    """
    # train_loader = NeighborSampler(dataset.train_edge, sizes=[-1], batch_size=5000, num_workers=6, shuffle=True)
    # # val_loader = NeighborSampler(dataset.val_edge, sizes=[-1], batch_size=5000, num_workers=6,shuffle=False)
    # test_loader = NeighborSampler(dataset.test_edge, sizes=[-1], batch_size=5000, num_workers=6, shuffle=False)
    # assert len(test_loader) == 1
    # batch_num = len(train_loader)


    # 寻找Top k个离攻击者社区最远的用户社区
    normal_u_cluster = kmeans(u_emb, u_atk_emb, args.n_clusters, args.k)
    # model.cluster_layer.data = kmeans(u_emb, u_atk_emb, args.n_clusters, args.k).to(device)
    for i, cluster_nodes in enumerate(normal_u_cluster):
        print(f"Cluster {i + 1} nodes shape:", cluster_nodes)
        print(f"Cluster {i + 1} nodes size:", cluster_nodes.size(0))

    # k个正常用户社区节点embedding
    normal_u_emb = [u_emb[cluster_nodes] for cluster_nodes in normal_u_cluster]
    normal_u_emb = torch.cat(normal_u_emb, dim=0)
    print("normal_u_emb shape:", normal_u_emb.shape)
    print("normal_u_emb size:", normal_u_emb)


    # torch.cuda.empty_cache()
    # max_train_f1 = 0
    # max_test_f1 = 0
    # max_test_acc = 0
    # max_test_pre = 0
    # max_test_recall = 0
    # max_test_auc = 0
    # max_epoch = 0

    # for batch_size, n_id, adjs in test_loader:
    #     test_init_data = init_data(adjs, n_id, train=False)
    #     test_adjs = adjs.to(device)
    #     test_n_id = n_id



if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument('--dirpath', default=BASE_DIR, help='Current Dir')
    parser.add_argument('--device', type=str, default='cpu', help='Devices')
    parser.add_argument('--dataset_name', type=str, default='bitcoin_alpha-1')
    parser.add_argument('--u_emb_size', type=int, default=64, help='Embeding U Size')
    parser.add_argument('--i_emb_size', type=int, default=64, help='Embeding I Size')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight Decay')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning Rate')
    parser.add_argument('--seed', type=int, default=2023, help='Random seed')
    parser.add_argument('--pre_gnn_epoch', default=150, type=int)
    parser.add_argument('--epoch', type=int, default=200, help='Epoch')
    # parser.add_argument('--gnn_layer_num', type=int, default=2, help='GNN Layer')
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--batch_size', type=int, default=500, help='Batch Size')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout')
    parser.add_argument('--gnn', type=str, default='sage')
    parser.add_argument('--n_clusters', default=32, type=int)
    parser.add_argument('--k', default=10, type=int)
    parser.add_argument('--agg', type=str, default='AttentionAggregator', choices=['AttentionAggregator', 'MeanAggregator'], help='Aggregator')
    args = parser.parse_args()

    # TODO
    # 排除两个超参，其他存在hyper_params里
    exclude_hyper_params = ['dirpath', 'device']
    hyper_params = dict(vars(args))
    for exclude_p in exclude_hyper_params:
        del hyper_params[exclude_p]

    hyper_params = "~".join([f"{k}-{v}" for k,v in hyper_params.items()])

    from torch.utils.tensorboard import SummaryWriter
    # https://pytorch.org/docs/stable/tensorboard.html
    tb_writer = SummaryWriter(comment=hyper_params)

    # setup seed
    setup_seed(args.seed)
    # setup device
    device = torch.device(args.device)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # get dataset
    dataset = ab.get_abcore_data(device)

    focal_loss = FocalLoss(2)
    root_path, _ = os.path.split(os.path.abspath(__file__))
    run(dataset)