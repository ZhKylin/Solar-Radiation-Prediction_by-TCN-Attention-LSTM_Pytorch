# Encoding: UTF-8
# Author: Kylin Zhang
# Time: 2024/6/6 - 23:48

# step2-3-1：构建TCN+Attention+LSTM联用神经网络，并使用划分好的数据集训练


import sys
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
from datetime import datetime
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader


# 自定义网络结构导入
from TCN import TCNNet
from MultiHeadAttention import MultiHeadAttention

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


# 搭建TCN+Attention+LSTM联用神经网络
class TCNAttentionLSTM(nn.Module):
    def __init__(self,
                 feature_size,         # 输入的特征数
                 output_size,          # 输出的特征数
                 hidden_size,          # LSTM网络--隐藏层大小
                 num_heads,            # Attention网络--要分裂的注意力头数
                 num_channels: tuple,  # TCN网络--网络层结构参数(必须是元组)
                 kernel_size=2,        # TCN网络--卷积核大小
                 dropout=0.2           # dropout
                 ):

        super(TCNAttentionLSTM, self).__init__()
        self.hidden_size = hidden_size
        # (batch_size, seq_len, feature_size)

        # layer1: TCN卷积网络进行特征提取
        self.tcn = TCNNet(feature_size, num_channels, kernel_size, dropout)
        # -->(batch_size, seq_len, num_channels[-1])

        # layer2: Attention网络调整特征权重
        # num_heads必须是num_channels[-1]的因数
        assert num_channels[-1] % num_heads == 0
        self.attention1 = MultiHeadAttention(num_channels[-1], num_heads)
        # -->(batch_size, seq_len, num_channels[-1])

        # layer3: 两层LSTM网络, Relu激活函数
        self.lstm1 = nn.LSTM(num_channels[-1], hidden_size, num_layers=2,
                             batch_first=True, dropout=dropout)
        self.relu1 = nn.ReLU()
        # -->(batch_size, seq_len, hidden_size)

        # layer4: Attention网络调整特征权重
        # num_heads必须是hidden_size的因数
        assert hidden_size % num_heads == 0
        self.attention2 = MultiHeadAttention(hidden_size, num_heads)
        # -->(batch_size, seq_len, hidden_size)

        # layer5: 两层LSTM网络, Relu激活函数
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers=2,
                             batch_first=True, dropout=dropout)
        self.relu2 = nn.ReLU()
        # -->(batch_size, seq_len, hidden_size)

        # layer6: 输出层
        # 取每个batch中的最后一个时间序列的全部特征值作为全连接层的输入
        # -->(batch_size, -1(维度自动消失), hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        # -->(batch_size, output_size)

    def forward(self, x):
        # LSTM网络参数初始化
        h_01 = torch.rand(2, x.size(0), self.hidden_size).to(device)
        c_01 = torch.rand(2, x.size(0), self.hidden_size).to(device)
        h_02 = torch.rand(2, x.size(0), self.hidden_size).to(device)
        c_02 = torch.rand(2, x.size(0), self.hidden_size).to(device)

        # 网络搭建
        x = self.tcn(x)
        x = self.attention1(x)
        x, _ = self.lstm1(x, (h_01, c_01))
        x = self.relu1(x)
        x = self.attention2(x)
        x, _ = self.lstm2(x, (h_02, c_02))
        x = self.relu2(x)
        output = self.fc(x[:, -1, :])
        return output


# 网络超参数定义
feature_size = x_train.shape[2]  # 输入的特征数
output_size = 1                  # 输出的特征数
hidden_size = 256                 # LSTM网络--隐藏层大小
num_heads = 16                    # Attention网络--要分裂的注意力头数
num_channels = (64, 128, 256)      # TCN网络--网络层结构参数(必须是元组)
kernel_size = 2                  # TCN网络--卷积核大小
dropout = 0.2                    # dropout
learn_rate = 0.001
# num_heads必须是num_channels[-1]的因数
assert num_channels[-1] % num_heads == 0
# num_heads必须是hidden_size的因数
assert hidden_size % num_heads == 0


# 实例化模型，定义损失函数和优化器
tcn_att_lstm = TCNAttentionLSTM(feature_size, output_size, hidden_size,
                                num_heads, num_channels=num_channels,
                                kernel_size=kernel_size, dropout=dropout)
tcn_att_lstm.to(device)
criterion = nn.MSELoss()
criterion.to(device)
optimizer = torch.optim.Adam(tcn_att_lstm.parameters(), lr=learn_rate)


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
            outputs = tcn_att_lstm(x_train)
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
            outputs = tcn_att_lstm(x_val)
        loss = criterion(outputs, y_val)
        print(f'本轮验证loss值为{loss.item()}')
        val_loss.append(loss.item())
        end_time = time.time()

        print(f'本轮用时{end_time-start_time}秒')

    # 模型保存
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = f'./trained_net/result_{current_time}'
    os.makedirs(save_path, exist_ok=True)
    model_name = f'TCN-Attention-LSTM_Net.pth'
    torch.save(tcn_att_lstm.state_dict(), os.path.join(save_path, model_name))
    print(f'\n模型{model_name}已成功存储在{os.path.abspath(save_path)}中')
    # 模型超参数保存
    para_file_name = f'net_para.csv'
    para_dir = {'feature_size': feature_size, 'output_size': output_size,
                'hidden_size': hidden_size, 'num_heads': num_heads,
                'num_channels': num_channels, 'kernel_size': kernel_size,
                'dropout': dropout}
    formatted_data = [
        [key, ', '.join(map(str, value)) if isinstance(value, tuple) else value]
        for key, value in para_dir.items()
    ]
    net_para = pd.DataFrame(formatted_data, columns=['Parameter', 'Value'])
    net_para.to_csv(os.path.join(save_path, para_file_name), index=False)
    print(f'\n本次训练模型超参数已成功存储在{os.path.abspath(save_path)}中')
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
        outputs = tcn_att_lstm(x_val)
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
