import json
import random
from collections import defaultdict

# 读取json文件
with open('data_unbalanced_metainfo.json', 'r') as file:
    data = json.load(file)

# 获取图片路径和标签
img_paths = [item['img_path'] for item in data['data_list']]
gt_labels = [item['gt_label'] for item in data['data_list']]

# 创建一个字典，用于存储每个标签对应的样本索引
label_to_indices = defaultdict(list)
for idx, labels in enumerate(gt_labels):
    for label in labels:
        label_to_indices[label].append(idx)

# 确保每个标签在训练集和验证集中至少有两个样本
train_indices = set()
val_indices = set()
for label, indices in label_to_indices.items():
    if len(indices) < 2:
        raise ValueError(f"The class {label} has less than 2 samples, which is too few.")
    
    random.shuffle(indices)
    split_point = max(1, int(0.9 * len(indices)))
    
    train_indices.update(indices[:split_point])
    val_indices.update(indices[split_point:])

# 将索引转换为列表并进行混洗
train_indices = list(train_indices)
val_indices = list(val_indices)
random.shuffle(train_indices)
random.shuffle(val_indices)

# 提取训练集和验证集的数据
train_data_list = [data['data_list'][i] for i in train_indices]
val_data_list = [data['data_list'][i] for i in val_indices]

# 创建训练集和验证集的字典
train_data = {
    'metainfo': data['metainfo'],
    'data_list': train_data_list
}

val_data = {
    'metainfo': data['metainfo'],
    'data_list': val_data_list
}

# 将训练集和验证集保存为json文件
with open('train_data.json', 'w') as file:
    json.dump(train_data, file, indent=4)

with open('val_data.json', 'w') as file:
    json.dump(val_data, file, indent=4)

# 输出训练集和验证集的大小
print("Training set size:", len(train_data_list))
print("Validation set size:", len(val_data_list))



"""
Training set size: 236814
Validation set size: 56123
"""
