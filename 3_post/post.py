# Encoding: UTF-8
# Author: Kylin Zhang
# Time: 2024/6/8 - 13:14

# step3：预测结果数据后处理


import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
import sklearn.metrics as sm
import numpy as np

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

"""
                            **后处理程序使用说明**
1.选择要处理的预测结果csv文件
(存储在各网络文件夹中的prediction_result文件夹内,文件名"预测结果和评估指标.csv")
2.复制"预测结果和评估指标.csv"文件的仓库根路径,赋值给csv_data_path变量
(路径形式示例:
①'2-1_BP/prediction_result/pre_result_2024-06-08_12-49-49/预测结果和评估指标.csv'
②'2-2_LSTM/prediction_result/pre_result_2024-06-08_13-31-08/预测结果和评估指标.csv'
③'2-3_TCN+Attention+LSTM/prediction_result/pre_result_2024-06-08_12-08-19/预测结果和评估指标.csv')
"""
# 后处理数据选择
csv_data_path = '2-3_TCN+Attention+LSTM/prediction_result/pre_result_2024-06-08_12-08-19/预测结果和评估指标.csv'

data = pd.read_csv(os.path.join('..', csv_data_path))
parts = csv_data_path.split('/')
net_type = parts[0]
net_type = net_type[4:]

prediction = data['prediction']
real = data['real']
prediction = prediction.to_numpy(prediction)
real = real.to_numpy(real)

# period 1
start_num1 = 250
end_num1 = 850
pre1 = prediction[start_num1: end_num1]
real1 = real[start_num1: end_num1]

# period 2
start_num2 = 2000
end_num2 = 2600
pre2 = prediction[start_num2: end_num2]
real2 = real[start_num2: end_num2]

current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
save_path = f'./post_result/{net_type}_{current_time}'
os.makedirs(save_path, exist_ok=True)

# period 1 plot
plt.figure(figsize=(12, 8))
plt.plot(real1, label='真实值', linewidth=2)
plt.plot(pre1, label='预测值', linewidth=2)
plt.title('测试集预测结果', fontsize=24)
plt.xlabel('时间轴', fontsize=24)
plt.ylabel('太阳辐射能(W/m^2)', fontsize=24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(loc='best', fontsize=24)
figure_name1 = f'预测值曲线阶段1.png'
plt.savefig(os.path.join(save_path, figure_name1))

# period 2 plot
plt.figure(figsize=(12, 8))
plt.plot(real2, label='真实值', linewidth=2)
plt.plot(pre2, label='预测值', linewidth=2)
plt.title('测试集预测结果', fontsize=24)
plt.xlabel('时间轴', fontsize=24)
plt.ylabel('太阳辐射能(W/m^2)', fontsize=24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(loc='best', fontsize=24)
figure_name2 = f'预测值曲线阶段2.png'
plt.savefig(os.path.join(save_path, figure_name2))
print(f'\n预测值图像已成功存储在{os.path.abspath(save_path)}中')

plt.show()


# 预测性能指标
def smape(real, pred):
    # 对称平均绝对百分比误差
    denominator = (np.abs(real) + np.abs(pred)) / 2
    smape_value = np.mean(np.abs(real - pred) / denominator) * 100
    return smape_value


def rae(real, pred):
    # 相对绝对误差
    numerator = np.sum(np.abs(real - pred))
    denominator = np.sum(np.abs(real - np.mean(real)))
    rae_value = numerator / denominator
    return rae_value * 100

# period 1
r21 = sm.r2_score(real1, pre1)
mae1 = sm.mean_absolute_error(real1, pre1)  # 平均绝对误差
rmse1 = np.sqrt(sm.mean_squared_error(real1, pre1))  # 均方根误差
mape1 = np.mean(np.absolute(real1 - pre1) / real1) * 100  # 均方百分比误差
smape1 = smape(real1, pre1)  # 对称平均绝对百分比误差
rae1 = rae(real1, pre1)  # 相对绝对误差
print(f'------------{net_type}阶段1------------')
print(f'R方：{r21}')
print(f'平均绝对误差：{mae1}')
print(f'均方根误差:{rmse1}')
print(f'均方百分比误差:{mape1:.2f}%')
print(f'对称平均绝对百分比误差:{smape1:.2f}%')
print(f'相对绝对误差:{rae1:.2f}%')
print('------------------------------\n')

# period 2
r22 = sm.r2_score(real2, pre2)
mae2 = sm.mean_absolute_error(real2, pre2)  # 平均绝对误差
rmse2 = np.sqrt(sm.mean_squared_error(real2, pre2))  # 均方根误差
mape2 = np.mean(np.absolute(real2 - pre2) / real2) * 100  # 均方百分比误差
smape2 = smape(real2, pre2)  # 对称平均绝对百分比误差
rae2= rae(real2, pre2)  # 相对绝对误差
print(f'------------{net_type}阶段2------------')
print(f'R方：{r22}')
print(f'平均绝对误差：{mae2}')
print(f'均方根误差:{rmse2}')
print(f'均方百分比误差:{mape2:.2f}%')
print(f'对称平均绝对百分比误差:{smape2:.2f}%')
print(f'相对绝对误差:{rae2:.2f}%')
print('------------------------------')

# 预测结果储存
file_name = f'两阶段预测评估指标.csv'
eval_para = {'r2': (r21, r22), 'mae': (mae1, mae2),
             'rmse': (rmse1, rmse2), 'mape': (mape1, mape2),
             'smape': (smape1, smape2), 'rae': (rae1, rae2)}
result = pd.DataFrame(eval_para, index=['Period1', 'Period2']).T
result.to_csv(os.path.join(save_path, file_name))
print(f'\n处理结果已成功存储在{os.path.abspath(save_path)}中')