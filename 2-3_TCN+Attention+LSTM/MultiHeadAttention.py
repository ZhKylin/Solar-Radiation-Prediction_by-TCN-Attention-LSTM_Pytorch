# Encoding: UTF-8
# Author: Kylin Zhang
# Time: 2024/6/4 - 14:20

# 构建多头注意力机制网络


import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, feature_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert feature_size % num_heads == 0
        self.num_heads = num_heads
        self.depth = feature_size // num_heads  # 每个头的维度 = 特征维度/头数目
        self.feature_size = feature_size

        self.w_q = nn.Linear(feature_size, feature_size)  # 查询向量对应的权重矩阵
        self.w_k = nn.Linear(feature_size, feature_size)  # 键向量对应的权重矩阵
        self.w_v = nn.Linear(feature_size, feature_size)  # 值向量对应的权重矩阵
        self.w_o = nn.Linear(feature_size, feature_size)  # 输出向量对应的权重矩阵

        self.layer_norm = nn.LayerNorm(self.feature_size)

    def split(self, x, batch_size):
        # 头分裂函数
        # x(batch_size, seq_len, feature_size)
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        # --> x(batch_size, seq_len, num_heads, depth)
        return x.transpose(1, 2)
        # --> x(batch_size, num_heads，seq_len, depth)

    def forward(self, x):
        batch_size = x.shape[0]

        # 向量头分裂
        q = self.split(self.w_q(x), batch_size)
        k = self.split(self.w_k(x), batch_size)
        v = self.split(self.w_v(x), batch_size)

        # 计算注意力分数
        source = (torch.matmul(q, k.transpose(-1, -2)) /
                  torch.sqrt(torch.tensor(self.feature_size,
                                          dtype=torch.float32)))
        # 计算注意力权重矩阵
        alpha = F.softmax(source, dim=-1)
        # alpha(batch_size, num_heads，seq_len, seq_len)
        # 计算中间结果
        context = torch.matmul(alpha, v)
        # context(batch_size, num_heads，seq_len, depth)

        # 头合并输出
        context = context.transpose(1, 2).contiguous()
        # --> context(batch_size, seq_len, num_heads, depth)
        context = context.view(batch_size, -1, self.feature_size)
        # --> context(batch_size, seq_len, feature_size)

        # 残差连接和层归一化
        output = self.w_o(context)
        output = self.layer_norm(output + x)

        return output


if __name__ == "__main__":
    x = torch.randn(100, 128, 64)
    attention_layer = MultiHeadAttention(64, 4)
    output = attention_layer(x)
    """
     数据结构流:(100, 128, 64)头分裂-->(100, 128, 4, 16)输出转置-->(100, 4, 128, 16)
               分数计算-->(100, 4, 128, 128)中间结果计算-->(100, 4, 128, 16)
               合并前转置-->(100, 128, 4, 16)头合并输出-->(100, 128, 64)
     """
    print(output.shape)  # 输出形状应为(100, 128, 64)
