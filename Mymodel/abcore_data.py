import pyabcore
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from time import time
import torch
import networkx as nx
from collections import Counter
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data.dataset import Dataset

def get_matrix(dataset):
    # 有问题 跑太慢
    max_idx = max(dataset.max_u, dataset.max_i)
    # 创建一个空的方阵邻接矩阵，初始化为0
    adjacency_matrix = torch.zeros((max_idx, max_idx), dtype=torch.float)

    # 填充邻接矩阵
    for row in dataset.train_edge:
        u, i, l = row[0].item(), row[1].item(), row[2].item()
        adjacency_matrix[u, i] = l

    # 将邻接矩阵与自身相乘
    result_matrix = torch.matmul(adjacency_matrix, adjacency_matrix)

    # 归一化矩阵A1
    row_sums_1 = torch.abs(adjacency_matrix).sum(dim=1, keepdim=True)
    column_sums_1 = torch.abs(adjacency_matrix).sum(dim=0, keepdim=True)

    # 归一化矩阵A2
    row_sums_2 = torch.abs(result_matrix).sum(dim=1, keepdim=True)
    column_sums_2 = torch.abs(result_matrix).sum(dim=0, keepdim=True)


    # 避免对全零行和列进行除零操作
    row_sums_1[row_sums_1 == 0] = 1
    column_sums_1[:, column_sums_1[0] == 0] = 1

    row_sums_2[row_sums_2 == 0] = 1
    column_sums_2[:, column_sums_2[0] == 0] = 1

    normalized_result_1 = adjacency_matrix / row_sums_1
    normalized_result_1 = normalized_result_1 / column_sums_1

    normalized_result_2 = result_matrix / row_sums_2
    normalized_result_2 = normalized_result_2 / column_sums_2

    # 获取归一化后的 label 列
    train_data_a1 = pd.DataFrame({'l': normalized_result_1[dataset.train_edge[:, 0], dataset.train_edge[:, 1]].numpy()})
    train_data_a2 = pd.DataFrame({'l': normalized_result_2[dataset.train_edge[:, 0], dataset.train_edge[:, 1]].numpy()})

    # non_zero_count = (train_data_a2['l'] != 0).sum()
    # print("train_data_a2中非零数字的个数:", non_zero_count.item())
    # nan_count = train_data_a2['l'].isna().sum()
    # print("train_data_a2中的 NaN 值的个数:", nan_count.item())

    return train_data_a1, train_data_a2


# load datasets
def get_data():
    root_path, _ = os.path.split(os.path.abspath(__file__))

    train_data = pd.read_csv(root_path+'/datasets/bitcoin/bitcoin_alpha-1_training.txt', names=['u', 'i', 'l', 'k'], delimiter='\t', dtype=int)
    val_data = pd.read_csv(root_path+'/datasets/bitcoin/bitcoin_alpha-1_validation.txt', names=['u', 'i', 'l', 'k'], delimiter='\t', dtype=int)
    test_data = pd.read_csv(root_path+'/datasets/bitcoin/bitcoin_alpha-1_testing.txt', names=['u', 'i','l', 'k'], delimiter='\t', dtype=int)


    df = pd.concat((train_data, val_data, test_data))

    dataset = Dataset()

    dataset.max_u = max(df['u'])+1
    dataset.max_i = max(df['i'])+1

    # df_labels = df[df['l'] != -1]

    # dataset.train_data = train_data[['u', 'i', 'l']]

    dataset.all_edge = np.array(df[['u', 'i']], dtype=np.int32)

    dataset.train_edge = torch.LongTensor(np.array(train_data[['u', 'i', 'l','k']]))

    dataset.val_edge = torch.LongTensor(np.array(val_data[['u', 'i', 'l','k']]))

    dataset.test_edge = torch.LongTensor(np.array(test_data[['u', 'i', 'l', 'k']]))

    dataset.train_u = list(set(train_data['u']))
    dataset.train_u.sort()
    dataset.train_u = torch.LongTensor(dataset.train_u)

    dataset.u_x = torch.FloatTensor(np.load(root_path+'/datasets/node_feature_div/alpha_u_feature.npy'))
    dataset.i_x = torch.FloatTensor(np.load(root_path+'/datasets/node_feature_div/alpha_v_feature.npy'))


    x = torch.cat([dataset.u_x, dataset.i_x], dim=0)
    dataset.x = torch.FloatTensor(x)

    return dataset

# execute α-β core
def get_abcore(dataset, device):
    abcore = pyabcore.Pyabcore(dataset.max_u, dataset.max_i)
    abcore.index(dataset.all_edge)
    a = 1
    b = 1

    dataset.core_u_x = torch.BoolTensor([])
    dataset.core_i_x = torch.BoolTensor([])
    while 1:
        abcore.query(a, b)
        result_u = torch.BoolTensor(abcore.get_left())
        result_i = torch.BoolTensor(abcore.get_right())
        if(result_i.sum() < len(result_i)*0.01):
            print('max b:{}'.format(b-1))
            dataset.max_b = b-1
            break

        dataset.core_u_x = torch.cat((dataset.core_u_x, result_u.unsqueeze(-1)),dim=1)
        dataset.core_i_x = torch.cat((dataset.core_i_x, result_i.unsqueeze(-1)),dim=1)
        b += 1


    dataset.u_x = torch.cat((dataset.core_u_x, dataset.u_x),dim=1).to(device)
    dataset.i_x = torch.cat((dataset.core_i_x,dataset.i_x),dim=1).to(device)
    print("-------------------测试输出--------------------------")
    print(dataset.core_u_x.size())  # [3783, 95]
    print(dataset.u_x.size())  # [3286, 64+95]
    print(dataset.core_i_x.size())  # [3783, 95]
    print(dataset.i_x.size())  # [3754, 64+95]


    a = 1
    b = 1
    # drop the inactive users
    while 1:
        abcore.query(a, b)
        result_u = torch.BoolTensor(abcore.get_left())
        # print(result_u.shape)   # ([3286])
        if(result_u.sum() < len(result_u)*0.3):
            print('inactive threshold a:{}'.format(a-1))
            dataset.min_a = a-1
            a = 1
            break
        a += 1

    # drop the unpopular products
    while 1:
        abcore.query(a, b)
        result_i = torch.BoolTensor(abcore.get_right())
        # print(result_i.shape)   # ([3754])
        if(result_i.sum() < len(result_i)*0.3):
            print('unpopular threshold b:{}'.format(b-1))
            dataset.min_b = b-1
            break
        b += 1

    abcore.query(dataset.min_a, dataset.min_b)
    result_u = torch.BoolTensor(abcore.get_left())
    result_i = torch.BoolTensor(abcore.get_right())

    # 更新train_edge
    selected_edges = []
    for edge in dataset.train_edge:
        # print(f"edge: {edge}")
        node1, node2 = edge[0].item(), edge[1].item()
        # print(node1, node2)
        if result_u[node1] and result_i[node2]:
            selected_edges.append(edge.tolist())
    dataset.train_edge = torch.LongTensor(np.array(selected_edges))
    dataset.train_edge = dataset.train_edge.to(device)
    dataset.train_edge_label = dataset.train_edge[:,2].to(device)
    # 拿到A2 label
    dataset.train_edge_a1_label, dataset.train_edge_a2_label = torch.LongTensor(np.array(get_matrix(dataset))).to(device)

    dataset.train_edge_attack = dataset.train_edge[:,3].to(device)
    dataset.train_edge_index = dataset.train_edge[:,0:2].T.to(device)
    dataset.u_atk_idx = torch.unique(dataset.train_edge[dataset.train_edge[:,3] == 1,0]).to(device)
    # print(dataset.u_atk_idx)
    # print(torch.unique(dataset.train_edge[:,0]))
    # print(dataset.train_edge.shape)
    # print(dataset.train_edge_a2_label.shape)
    # print(dataset.train_edge_a2_label)


    # 更新val_edge
    selected_edges = []
    for edge in dataset.val_edge:
        # print(f"edge: {edge}")
        node1, node2 = edge[0].item(), edge[1].item()
        # print(node1, node2)
        if result_u[node1] and result_i[node2]:
            selected_edges.append(edge.tolist())
    dataset.val_edge = torch.LongTensor(np.array(selected_edges))
    dataset.val_edge = dataset.val_edge.to(device)
    dataset.val_edge_label = dataset.val_edge[:,2].to(device)
    dataset.val_edge_attack = dataset.val_edge[:,3].to(device)
    dataset.val_edge_index = dataset.val_edge[:,0:2].T.to(device)
    # print(dataset.val_edge.shape)


    # 更新test_edge
    selected_edges = []
    for edge in dataset.test_edge:
        # print(f"edge: {edge}")
        node1, node2 = edge[0].item(), edge[1].item()
        # print(node1, node2)
        if result_u[node1] and result_i[node2]:
            selected_edges.append(edge.tolist())
    dataset.test_edge = torch.LongTensor(np.array(selected_edges))
    dataset.test_edge = dataset.test_edge.to(device)
    dataset.test_edge_label = dataset.test_edge[:,2].to(device)
    dataset.test_edge_attack = dataset.test_edge[:,3].to(device)
    dataset.test_edge_index = dataset.test_edge[:,0:2].T.to(device)
    # print(dataset.test_edge.shape)

    return dataset


def get_abcore_data(device):
    dataset= get_data()
    dataset = get_abcore(dataset, device)

    return dataset

if __name__ == '__main__':
    get_abcore_data('cpu')