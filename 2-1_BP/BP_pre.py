# Encoding: UTF-8
# Author: Kylin Zhang
# Time: 2024/5/29 - 18:17

# step2-1-2：用训练好的BP网络进行预测


import matplotlib.pyplot as plt
import torch
import os
import sys
import sklearn.metrics as sm
import numpy as np
import pandas as pd
from datetime import datetime

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 导入训练好的模型
from BP_train import BPModel
save_path = './trained_net'
folder_name = 'result_2024-06-08_12-16-33'  # 选择要导入的BP网络模型对应的文件夹名称
model_name = 'BP_Net.pth'
bp_net = torch.load(os.path.join(save_path, folder_name, model_name))
bp_net.to(device)

# 导入测试数据
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..', '1_dataset'))
from dataset import x_test, y_test
from dataset import scaler

# 数据转为torch张量
x_test = torch.from_numpy(x_test).float()
y_test = torch.from_numpy(y_test).float()
x_test, y_test = x_test.to(device), y_test.to(device)

# 使用训练好的网络预测测试集数据
bp_net.eval()
with torch.no_grad():
    outputs = bp_net(x_test)
print('BP网络预测完成')

# 结果存储路径
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
save_path = f'./prediction_result/pre_result_{current_time}'
os.makedirs(save_path, exist_ok=True)

# 预测数据处理
prediction = outputs.detach().cpu().numpy()
real = y_test.detach().cpu().numpy()
# 反归一化
data_range = scaler.data_range_
data_min = scaler.data_min_
prediction = prediction * data_range[0] + data_min[0]
real = real * data_range[0] + data_min[0]
prediction = prediction.reshape(prediction.shape[0])
real = real.reshape(real.shape[0])

# 预测结果可视化
plt.figure(figsize=(12, 8))
plt.plot(real, label='真实值', linewidth=1.5)
plt.plot(prediction, label='预测值', linewidth=0.75)
plt.title('测试集预测结果', fontsize=24)
plt.xlabel('时间轴', fontsize=24)
plt.ylabel('太阳辐射能(W/m^2)', fontsize=24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(loc='best', fontsize=24)
figure_name = f'预测值曲线.png'
plt.savefig(os.path.join(save_path, figure_name))
print(f'\n预测值图像已成功存储在{os.path.abspath(save_path)}中')
plt.show()

# 预测性能指标
r2 = sm.r2_score(real, prediction)
mae = sm.mean_absolute_error(real, prediction)  # 平均绝对误差
rmse = np.sqrt(sm.mean_squared_error(real, prediction))  # 均方根误差
mape = np.mean(np.absolute(real - prediction) / real) * 100  # 均方百分比误差
eval_para = {'r2': r2, 'mae': mae, 'rmse': rmse, 'mape': mape}
print('--------------------------')
print(f'\nR方：{r2}')
print(f'平均绝对误差：{mae}')
print(f'均方根误差:{rmse}')
print(f'均方百分比误差:{mape:.2f}%')
print('--------------------------')

# 预测结果储存
file_name = f'预测结果和评估指标.csv'
result = pd.DataFrame({'prediction': prediction, 'real': real})
result.loc[0, 'r2'] = r2
result.loc[0, 'mae'] = mae
result.loc[0, 'rmse'] = rmse
result.loc[0, 'mape'] = mape
result.to_csv(os.path.join(save_path, file_name))
print(f'\n预测结果已成功存储在{os.path.abspath(save_path)}中')