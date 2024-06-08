# Encoding: UTF-8
# Author: Kylin Zhang
# Time: 2024/5/27 - 20:23

# step1-2：读取处理后的表格并进行训练集、验证集和测试集划分


import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 参数
train_ratio = 0.8  # 训练集占比
val_ratio = 0.1  # 验证集占比
batch_size = 128  # 批处理大小(即一次用多少个连续数据预测一下数据)

# 读取表格文件
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, 'SolarPrediction_processed.csv')
dataset = pd.read_csv(data_path)

# 特征选取
dataset = dataset[['Radiation', 'Temperature', 'Pressure',
                   'Humidity', 'WindDirection(Degrees)',
                   'Speed', 'DayOfYear', 'TimeOfDay(s)', 'DayLength(s)']]

# 训练集和验证集划分
# 训练集 ：验证集 ：测试集 = 8 ：1 ：1
train = dataset.iloc[0:int(len(dataset) * train_ratio), :]
val = dataset.iloc[int(len(dataset) * train_ratio):
                   int(len(dataset) * (train_ratio + val_ratio)), :]
test = dataset.iloc[int(len(dataset) * (train_ratio + val_ratio)):, :]


# 数据归一化
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit_transform(dataset)
train = scaler.transform(train)
val = scaler.transform(val)
test = scaler.transform(test)


# 特征和标签划分
# 用每batch_size的数据预测紧邻的下一个值
x_train = []
y_train = []
x_val = []
y_val = []
x_test = []
y_test = []
for i in range(batch_size, len(train)):
    x_train.append(train[i - batch_size:i, :])
    y_train.append(train[i, 0])
for i in range(batch_size, len(val)):
    x_val.append(val[i - batch_size:i, :])
    y_val.append(val[i, 0])
for i in range(batch_size, len(test)):
    x_test.append(test[i - batch_size:i, :])
    y_test.append(test[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_val, y_val = np.array(x_val), np.array(y_val)
x_test, y_test = np.array(x_test), np.array(y_test)
y_train, y_val, y_test = \
    (y_train.reshape(-1, 1), y_val.reshape(-1, 1), y_test.reshape(-1, 1))


if __name__ == '__main__':
    print(x_train.shape, y_train.shape)
    print(x_val.shape, y_val.shape)
    print(x_test.shape, y_test.shape)

    # data_range = scaler.data_range_
    # data_min = scaler.data_min_
    # y_test = y_test * data_range[0] + data_min[0]
    # print(y_test)