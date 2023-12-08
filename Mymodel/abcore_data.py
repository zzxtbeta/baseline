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
    dataset.train_edge_attack = dataset.train_edge[:,3].to(device)
    dataset.train_edge_index = dataset.train_edge[:,0:2].T.to(device)
    dataset.u_atk_idx = torch.unique(dataset.train_edge[dataset.train_edge[:,3] == 1,0]).to(device)
    # print(torch.unique(dataset.train_edge[:,0]))
    # print(dataset.train_edge_index.shape)


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