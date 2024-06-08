# Encoding: UTF-8
# Author: Kylin Zhang
# Time: 2024/6/5 - 23:04

# 构建序列建模基准和时序卷积网络


import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm


class Chomp1d(nn.Module):
    # 剪切层：用于去掉未来部分，确保卷积的因果性
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        output = x[:, :, :-self.chomp_size].contiguous()
        return output


class BasicBlock(nn.Module):
    # 定义TCN网络的基础结构块
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation,
                 padding, dropout=0.2):
        super(BasicBlock, self).__init__()
        # 第一层
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding,
                                           dilation=dilation))
        self.chomp1 = Chomp1d(padding)  # 经卷积层后最后一维数据会多出padding，修剪掉
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        # 第二层
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding,
                                           dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )

        # 为了后续的残差连接处理，若输入输出大小不同需要进行下采样对维度进行匹配
        if n_inputs != n_outputs:
            self.downsample = nn.Conv1d(n_inputs, n_outputs, 1)
        else:
            self.downsample = None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        # 如果进行了下采样，同样需要对下采样网络权值初始化
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # x(batch_size, n_input, seq_len)
        output = self.net(x)
        # output(batch_size, n_output, seq_len)
        # 确保res形状与output完全相同
        res = x if self.downsample is None else self.downsample(x)
        # 残差连接
        return self.relu(output + res)


class TCNNet(nn.Module):
    # 搭建堆叠TCN神经网络
    def __init__(self, n_input, num_channels: tuple, kernel_size=2, dropout=0.2):
        super(TCNNet, self).__init__()
        layers = []  # 用于存储TCN网络各层
        num_layers = len(num_channels)  # 层数由num_channels的长度决定
        for i in range(num_layers):
            dilation_size = 2 ** i
            # 输入通道数，第一层是 n_inputs，其它层是前一层的输出通道数
            in_channels = n_input if i == 0 else num_channels[i-1]
            # 输出通道数等于当前层的通道数
            out_channels = num_channels[i]

            # 循环创建TCN网络层
            layers += [BasicBlock(in_channels, out_channels, kernel_size,
                                  stride=1, dilation=dilation_size,
                                  padding=(kernel_size-1) * dilation_size,
                                  dropout=dropout)]
        # 堆叠网络实例化
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x(batch_size, seq_len, feature_size(n_input))
        x = x.transpose(1, 2)  # 为了满足一维卷积的输入格式，需要将特征维度放到第二维
        # --> x(batch_size, n_input, seq_len)
        output = self.net(x)
        # output(batch_size, num_channels[-1], seq_len)
        return output.transpose(1, 2)
        # --> output(batch_size, seq_len, num_channels[-1])


if __name__ == "__main__":
    x = torch.randn(100, 128, 8)
    n_input = 8  # n_input参数必须与输入TCN网络的特征数相同
    num_channels = (16, 32)  # num_channels参数必须输入一个元组,代表每一层输出通道数
    tcn = TCNNet(n_input, num_channels)
    """
    数据结构流(100, 128, 8)输入转置-->(100, 8, 128)经过第一层网络-->
    (100, 16, 128)经过第二层网络-->(100, 32, 128)输出转置-->(100, 128, 32)
    """
    print(tcn(x).shape)  # 输出形状应为(100, 128, 32)
