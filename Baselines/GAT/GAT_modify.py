import argparse
import os.path as osp
import random
import time
import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import GATConv
from torch.utils.data.dataset import Dataset
from torch_geometric.nn import Node2Vec
from torch_geometric.data import Data
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, accuracy_score



parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='bitcoin_alpha-1')
parser.add_argument('--hidden_channels', type=int, default=8)
parser.add_argument('--heads', type=int, default=8)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--device', type=str, default='cuda:1', help='Devices')
parser.add_argument('--seed', type=int, default=2023, help='Random seed')
parser.add_argument('--wandb', action='store_true', help='Track experiment')
args = parser.parse_args()

# args.device = 'cuda:1'
device = torch.device(args.device)

def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
# setup seed
setup_seed(args.seed)

# 问题：他这个代码做的是一个节点分类，最后是判断每个节点属于哪个class，然后输入的y是节点的label，我的数据集是做边预测，没有节点标签，不知道怎么改了

def get_data():
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'topic_data', 'bitcoin_datasets')

    train_data = pd.read_csv(path + '/bitcoin_alpha-1_training.txt', names=['u', 'v', 'l','k'], delimiter='\t', dtype=int)
    test_data = pd.read_csv(path + '/bitcoin_alpha-1_testing.txt', names=['u', 'v', 'l', 'k'], delimiter='\t', dtype=int)
    val_data = pd.read_csv(path + '/bitcoin_alpha-1_validation.txt', names=['u', 'v', 'l', 'k'], delimiter='\t', dtype=int)

    df = pd.concat((train_data, val_data, test_data))

    # x = np.array(df['v', 'l','k'], dtype=np.int32)
    edge_index =  torch.tensor(np.transpose(np.array(df[['u', 'v']], dtype=np.int32)))

    one_hot_encoding = torch.eye(3783)  # alpha-3783   otc-5881
    x = one_hot_encoding.clone().detach()


    y = torch.tensor(np.array(df['k'], dtype=np.int32))

    # # 假设每个数据集的行数为 train_size、val_size 和 test_size
    train_size = len(train_data)
    val_size = len(val_data)
    test_size = len(test_data)

    # 创建相应的掩码
    train_mask = torch.BoolTensor(len(df))
    val_mask = torch.BoolTensor(len(df))
    test_mask = torch.BoolTensor(len(df))

    # 设置每个数据集的掩码
    train_mask[:train_size] = True
    train_mask[train_size:] = False
    val_mask[train_size:train_size + val_size] = True
    val_mask[train_size + val_size:] = False
    test_mask[-test_size:] = True
    test_mask[:-test_size] = False

    # train_mask[:train_size] = 1
    # val_mask[train_size:(train_size + val_size)] = 1
    # test_mask[-test_size:] = 1


    data = Data(x=x, y=y, edge_index=edge_index, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    return data

data = get_data().to(device)
print(data)



class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=0.6)
        # On the Pubmed dataset, use `heads` output heads in `conv2`.
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1,
                             concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x

num_features = 3783   # 节点特征维度
num_classes = 2  # 节点分类类别数

# 使用node2vec为每个节点生成初始特征


model = GAT(num_features, args.hidden_channels, num_classes,
            args.heads).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    print("data.train_mask shape:", data.train_mask.shape)
    print("out shape:", out.shape)
    loss = F.cross_entropy(out[data.train_mask], torch.tensor(data.y[data.train_mask], dtype=torch.long))
    # loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)

        # print(mask.dtype)
        # print(data.y[mask].dtype)
        # print(f"pred[mask]:{pred[mask]}")
        # print(f"len(data.y[mask]): {len(data.y[mask])}")
        # print(f"(len(pred[mask]): {len(pred[mask])}")
        # print(f"int(mask.sum()): {int(mask.sum())}")

@torch.no_grad()
def test():
    model.eval()
    pred = model(data.x, data.edge_index).argmax(dim=-1)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
        # accs.append(accuracy_score(data.y[mask], pred[mask])) / int(mask.sum())
    return accs





times = []
best_val_acc = final_test_acc = 0
for epoch in range(1, args.epochs + 1):
    start = time.time()
    loss = train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    log(Epoch=epoch, Loss=loss, Train=train_acc, Val=val_acc, Test=test_acc)
    times.append(time.time() - start)
print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")