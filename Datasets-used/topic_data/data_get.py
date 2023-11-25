import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 读取原始数据集
file_path = "/home/yrgu/topic/baseline/Datasets-used/topic_data/bitcoin_datasets/bitcoinalpha_sorted.csv"  # 请替换成你的数据集文件路径
data = pd.read_csv(file_path, delimiter='\t', header=None, names=['user', 'item', 'rating'])

# 打印初始数据总条数
print(f"Initial total number of records: {len(data)}")

# 1. 统计用户和商品节点的度数，并选择攻击者和目标商品
user_degrees = data.groupby('user').size().reset_index(name='user_degree')
item_degrees = data.groupby('item').size().reset_index(name='item_degree')

attackers_percentage = 0.05
attackers = user_degrees.nsmallest(int(attackers_percentage * len(user_degrees)), 'user_degree')['user'].tolist()

targets_count = 100
targets = item_degrees.nlargest(targets_count, 'item_degree')['item'].tolist()

# 2. 计算目标商品中正边和负边的比例，确定攻击者的差评对象组和好评对象组
target_data = data[data['item'].isin(targets)]
positive_ratio = target_data[target_data['rating'] == 1].shape[0] / target_data.shape[0]
negative_ratio = target_data[target_data['rating'] == -1].shape[0] / target_data.shape[0]

if positive_ratio > 0.9:
    negative_targets = targets
    positive_targets_count = int((1 - negative_ratio) * targets_count)
    positive_targets = item_degrees.nlargest(positive_targets_count, 'item_degree')['item'].tolist()
else:
    positive_targets = targets
    negative_targets_count = int(negative_ratio * targets_count)
    negative_targets = item_degrees.nsmallest(negative_targets_count, 'item_degree')['item'].tolist()

# 3. 生成攻击边
attack_edges = []
for attacker in attackers:
    # 为每个攻击者生成10条攻击边
    for _ in range(10):
        target = np.random.choice(positive_targets + negative_targets)
        sign = np.random.choice([-1, 1])
        attack_edges.append((attacker, target, sign))

# 打印新增攻击边的总数量以及虚假好评和恶意差评的数量
print(f"Total number of attack edges: {len(attack_edges)}")
print(f"Number of malicious positives (fake positives): {len([edge for edge in attack_edges if edge[2] == 1])}")
print(f"Number of malicious negatives (fake negatives): {len([edge for edge in attack_edges if edge[2] == -1])}")

# 4. 将攻击边加入到数据集中
attack_data = pd.DataFrame(attack_edges, columns=['user', 'item', 'rating'])
data = pd.concat([data, attack_data], ignore_index=True)

# 打印处理后的数据总条数
print(f"Processed total number of records: {len(data)}")

# 5. 重新打标签
data['new_rating'] = 0
attack_edges_count = int(0.3 * len(attack_edges))
attack_edges_indices = np.random.choice(len(attack_edges), attack_edges_count, replace=False)
data.loc[data.index.isin(attack_edges_indices), 'new_rating'] = 1

# 统计各个数据集中正常边和攻击边的个数
def print_label_counts(data_set, data_set_name):
    """
    Print the count of 0s and 1s in the 'new_rating' column of a given dataset.

    Parameters:
    - data_set (pd.DataFrame): The dataset to analyze.
    - data_set_name (str): The name of the dataset for printing.

    Returns:
    None
    """
    label_counts = data_set['new_rating'].value_counts()
    print(f"{data_set_name} Label Counts:")
    print(f"Label 0 count: {label_counts.get(0, 0)}")
    print(f"Label 1 count: {label_counts.get(1, 0)}")
    print("\n")

# 6. 划分数据集并保存（使用分层抽样）
for i in range(5):
    # 分层抽样
    train_set, temp = train_test_split(data, test_size=0.15, random_state=None, stratify=data['new_rating'])
    validation_set, test_set = train_test_split(temp, test_size=0.67, random_state=None, stratify=temp['new_rating'])
    
    # 打印各自数据集的条数
    print_label_counts(train_set, f"bitcoin_alpha-{i+1}_training.txt")
    print_label_counts(validation_set, f"bitcoin_alpha-{i+1}_validation.txt")
    print_label_counts(test_set, f"bitcoin_alpha-{i+1}_testing.txt")
    print("------------------------------------------------------------------------")
    
    
    # 保存数据集
    train_set.to_csv(f"bitcoin_alpha-{i+1}_training.txt", sep='\t', index=False, header=False)
    validation_set.to_csv(f"bitcoin_alpha-{i+1}_validation.txt", sep='\t', index=False, header=False)
    test_set.to_csv(f"bitcoin_alpha-{i+1}_testing.txt", sep='\t', index=False, header=False)