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
import torch.optim as optim
from torch.optim import Adam
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.transforms import AddSelfLoops, NormalizeScale
from torch.utils.data import DataLoader
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


class GCN_1(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim, add_self_loops=True)
        self.conv2 = GCNConv(hidden_dim, out_dim, add_self_loops=True)

    def forward(self, x, edge_index, edge_weight):
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x

class GCN_2(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index, edge_weight):
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim=1):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


class GNN(nn.Module):
    def __init__(self, u_input_dim, i_input_dim, hidden_dim, output_dim=1):
        super(GNN, self).__init__()
        # GCN layers
        self.gcn_1 = GCN_1(u_input_dim, hidden_dim, u_input_dim)
        self.gcn_2 = GCN_2(u_input_dim, hidden_dim, u_input_dim)
        # MLP layers
        self.mlp = MLP(u_input_dim * 2 + i_input_dim, 128, 32, output_dim)


    def forward(self, u_x, i_x, edge_index, edge_a1_label, edge_a2_label, u_atk_idx):
        # AWX + self.loop()
        u_x_a = self.gcn_1(u_x, edge_index, edge_a1_label.float())
        print("u_x_a shape:", u_x_a.shape)
        i_x_a = self.gcn_1(i_x, edge_index, edge_a1_label.float())
        print("i_x_a shape:", i_x_a.shape)

        # A2WX
        u_x_a2 = self.gcn_2(u_x, edge_index, edge_a2_label.float())
        i_x_a2 = self.gcn_2(i_x, edge_index, edge_a2_label.float())

        # 特征经过A和A_2卷积后拼接得到用户特征
        u_emb = torch.cat((u_x_a, u_x_a2), dim=1)
        print("u_emb shape:", u_emb.shape)

        # 筛选出攻击者特征
        u_atk_emb = u_emb[dataset.u_atk_idx,:]
        print("u_atk_emb shape:", u_atk_emb.shape)

        # 创建一个掩码，将 u_atk_idx 对应的位置置为 False，其余位置置为 True
        mask = torch.ones(u_emb.shape[0], dtype=torch.bool)
        mask[dataset.u_atk_idx] = False

        # 使用掩码从 u_emb 中选择非攻击者节点的特征
        u_emb = u_emb[mask]

        # 拿正常用户和被标记攻击者的特征去做kmeans，与攻击者社区中心比较，寻找Top k个离攻击者社区最远的用户社区
        normal_u_cluster = kmeans(u_emb, u_atk_emb, args.n_clusters, args.k)

        # 将所有 Top k 个正常用户社区中的用户节点索引汇总到一个张量中
        all_normal_u_indices = torch.cat(normal_u_cluster)
        print("All Top k user indices:", all_normal_u_indices.shape)

        # 创建掩码，正常用户索引对应位置为True，其他为False
        u_mask = torch.zeros(u_emb.shape[0], dtype=torch.bool)
        u_mask[all_normal_u_indices] = True
        u_edge_mask = u_mask[edge_index[0]]

        # 利用正常用户节点索引掩码提取正常边
        normal_edge = edge_index[:, u_edge_mask]

        # 根据原本标记出来的攻击者索引提取攻击边
        attack_edge = edge_index[:, dataset.u_atk_idx]

        # 根据 normal_edge 中每条边节点 u 和 i 的节点索引找到对应的 u_emb 和 i_emb
        normal_u_emb = u_emb[normal_edge[0]]
        normal_i_emb = i_x_a[normal_edge[1]]

        attack_u_emb = u_emb[attack_edge[0]]
        attack_i_emb = i_x_a[attack_edge[1]]

        # 将 normal_u_emb 和 normal_i_emb 拼接起来作为正常边特征
        normal_edge_emb = torch.cat([normal_u_emb, normal_i_emb], dim=1)
        attack_edge_emb = torch.cat([attack_u_emb, attack_i_emb], dim=1)
        print("normal_edge_emb shape:", normal_edge_emb.shape)
        print("attack_edge_emb shape:", attack_edge_emb.shape)

        # MLP model
        pred_y = self.mlp(normal_edge_emb)

        return pred_y



def run(dataset, model, optimizer, args):
    # Forward pass
    pred_y = model(dataset.u_x, dataset.i_x, dataset.train_edge_index,
                   dataset.train_edge_a1_label.float(), dataset.train_edge_a2_label.float(),
                   dataset.u_atk_idx)

    # Print and log model output
    print(args)
    print(pred_y.shape)

    # Compute loss
    target = dataset.train_edge_a1_label.float()
    loss = focal_loss(pred_y, target)  # Assuming target is defined

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


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
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--batch_size', type=int, default=500, help='Batch Size')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout')
    parser.add_argument('--gnn', type=str, default='gcn')
    parser.add_argument('--n_clusters', default=32, type=int)
    parser.add_argument('--k', default=12, type=int)
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
     # Model definition
    gnn_model = GNN(u_input_dim=dataset.u_x.shape[1], i_input_dim=dataset.i_x.shape[1], hidden_dim=args.hidden_dim)
    print(f"dataset.u_x.shape[1]: {dataset.u_x.shape[1]}")

    # Optimizer definition
    all_parameters = list(gnn_model.parameters())
    optimizer = Adam(all_parameters, lr=args.lr)

    # Run the training loop
    for epoch in range(args.epoch):
        run(dataset, gnn_model, optimizer, args)
