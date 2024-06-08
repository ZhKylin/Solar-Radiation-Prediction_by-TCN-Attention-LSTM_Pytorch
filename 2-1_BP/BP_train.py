# Encoding: UTF-8
# Author: Kylin Zhang
# Time: 2024/5/28 - 10:39

# step2-1-1：构建BP神经网络，并使用划分好的数据集训练


import sys
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
from datetime import datetime
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 导入训练集和验证集
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..', '1_dataset'))
from dataset import x_train, y_train, x_val, y_val
from dataset import scaler
# 数据转为torch张量
x_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(y_train).float()
x_val = torch.from_numpy(x_val).float()
y_val = torch.from_numpy(y_val).float()
x_val, y_val = x_val.to(device), y_val.to(device)
# 分批次训练
batch_size = 1024
dataset = TensorDataset(x_train, y_train)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        pin_memory=True)


# 搭建PB神经网络
class BPModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BPModel, self).__init__()
        self.fc1 = nn.Sequential(
            # BP神经网络输入形状为(Batch_size，Feature_size),将第一维后纬度展平
            nn.Flatten(start_dim=1),
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size*2),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(hidden_size*2, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        out = self.fc3(x)
        return out


# 网络超参数定义
input_size = x_train.shape[1] * x_train.shape[2]  # 展平后的特征数量
hidden_size = 2048
output_size = 1
learn_rate = 0.001

# 实例化模型，定义损失函数和优化器
bp_net = BPModel(input_size, hidden_size, output_size)
bp_net.to(device)
criterion = nn.MSELoss()
criterion.to(device)
optimizer = torch.optim.Adam(bp_net.parameters(), lr=learn_rate)

if __name__ == '__main__':
    # 训练循环
    epoch = 1000
    train_loss = []
    val_loss = []
    for i in range(epoch):
        print(f'---------第{i+1}轮训练开始---------')
        start_time = time.time()
        # 模型训练
        # 前向传播
        total_loss = 0
        for x_train, y_train in dataloader:
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            # 优化器梯度清零
            optimizer.zero_grad()
            # 前向传播
            outputs = bp_net(x_train)
            # 损失计算
            loss = criterion(outputs, y_train)
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'本轮训练总loss值为{total_loss}')
        train_loss.append(total_loss)

        # 模型验证
        with torch.no_grad():
            outputs = bp_net(x_val)
        loss = criterion(outputs, y_val)
        print(f'本轮验证loss值为{loss.item()}')
        val_loss.append(loss.item())

        end_time = time.time()
        print(f'本轮用时{end_time-start_time}秒')

    # 模型保存
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = f'./trained_net/result_{current_time}'
    os.makedirs(save_path, exist_ok=True)
    model_name = f'BP_Net.pth'
    torch.save(bp_net, os.path.join(save_path, model_name))
    print(f'\n模型{model_name}已成功存储在{os.path.abspath(save_path)}中')
    # 训练日志保存(loss值)
    log_name = f'loss.csv'
    log = pd.DataFrame({'train_loss': train_loss, 'val_loss': val_loss})
    log.to_csv(os.path.join(save_path, log_name))
    print(f'\n本次训练日志已成功存储在{os.path.abspath(save_path)}中')

    # 数据可视化
    # 画loss值变化曲线
    plt.figure(figsize=(12, 8))
    plt.semilogy(range(epoch), train_loss, label='训练集loss', linewidth=2)
    plt.semilogy(range(epoch), val_loss, label='验证集loss', linewidth=2)
    plt.title(f'loss曲线_{current_time}', fontsize=24)
    plt.xlabel('训练次数', fontsize=24)
    plt.ylabel('loss值', fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(loc='best', fontsize=24)
    # 保存loss图像
    figure_name = f'训练loss曲线.png'
    plt.savefig(os.path.join(save_path, figure_name))
    print(f'\n本次训练loss图像已成功存储在{os.path.abspath(save_path)}中')

    # 画出验证集的真实数据和对应的预测数据
    with torch.no_grad():
        outputs = bp_net(x_val)
    outputs = outputs.detach().cpu().numpy()
    y_val = y_val.detach().cpu().numpy()
    # 反归一化
    data_range = scaler.data_range_
    data_min = scaler.data_min_
    outputs = outputs * data_range[0] + data_min[0]
    y_val = y_val * data_range[0] + data_min[0]
    # 画图
    plt.figure(figsize=(12, 8))
    plt.plot(y_val, label='真实值', linewidth=1.5)
    plt.plot(outputs, label='预测值', linewidth=0.75)
    plt.title('验证集预测结果', fontsize=24)
    plt.xlabel('时间轴', fontsize=24)
    plt.ylabel('太阳辐射能(W/m^2)', fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(loc='best', fontsize=24)
    # 保存验证集预测结果图像
    figure_name = f'验证集预测结果.png'
    plt.savefig(os.path.join(save_path, figure_name))
    print(f'\n本次验证集预测结果图像已成功存储在{os.path.abspath(save_path)}中')

    plt.show()