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
路径形式示例:
①'2-1_BP/prediction_result/pre_result_2024-06-08_12-49-49/预测结果和评估指标.csv'
②'2-2_LSTM/prediction_result/pre_result_2024-06-08_13-31-08/预测结果和评估指标.csv'
③'2-3_TCN+Attention+LSTM/prediction_result/pre_result_2024-06-08_12-08-19/预测结果和评估指标.csv'
"""
# 后处理数据选择
csv_data_path = '2-3_TCN+Attention+LSTM/prediction_result/pre_result_2024-06-08_12-08-19/预测结果和评估指标.csv'

data = pd.read_csv(os.path.join('..', csv_data_path))
parts = csv_data_path.split('/')
net_type = parts[0]
net_type = net_type[4:]

pre = data['prediction']
real = data['real']
pre = pre.to_numpy()
real = real.to_numpy()

# period 1
start_num1 = 250
end_num1 = 850
pre1 = pre[start_num1: end_num1]
real1 = real[start_num1: end_num1]

# period 2
start_num2 = 2000
end_num2 = 2600
pre2 = pre[start_num2: end_num2]
real2 = real[start_num2: end_num2]

dataset = [(pre, real), (pre1, real1), (pre2, real2)]

current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
save_path = f'./post_result/{net_type}_{current_time}'
os.makedirs(save_path, exist_ok=True)

i = 1
for pre_p, real_p in dataset[1:]:
    # 画图
    plt.figure(figsize=(12, 8))
    plt.plot(real_p, label='真实值', linewidth=2)
    plt.plot(pre_p, label='预测值', linewidth=2)
    plt.title(f'测试集阶段{i}预测结果', fontsize=24)
    plt.xlabel('时间轴', fontsize=24)
    plt.ylabel('太阳辐射能(W/m^2)', fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(loc='best', fontsize=24)
    # 存储
    figure_name = f'预测值曲线阶段{i}.png'
    plt.savefig(os.path.join(save_path, figure_name))
    plt.show()
    i += 1


# 预测性能指标
def smape_f(real, pred):
    # 对称平均绝对百分比误差
    denominator = (np.abs(real) + np.abs(pred)) / 2
    smape_value = np.mean(np.abs(real - pred) / denominator) * 100
    return smape_value


def rae_f(real, pred):
    # 相对绝对误差
    numerator = np.sum(np.abs(real - pred))
    denominator = np.sum(np.abs(real - np.mean(real)))
    rae_value = numerator / denominator
    return rae_value * 100


i = 0
eval_para = {'r2': [], 'mae': [], 'rmse': [], 'mape': [], 'smape': [], 'rae': []}
for pre_p, real_p in dataset:
    # 评价指标计算
    r2 = sm.r2_score(real_p, pre_p)
    mae = sm.mean_absolute_error(real_p, pre_p)  # 平均绝对误差
    rmse = np.sqrt(sm.mean_squared_error(real_p, pre_p))  # 均方根误差
    mape = np.mean(np.absolute(real_p - pre_p) / real_p) * 100  # 均方百分比误差
    smape = smape_f(real_p, pre_p)  # 对称平均绝对百分比误差
    rae = rae_f(real_p, pre_p)  # 相对绝对误差
    eval_para['r2'].append(r2)
    eval_para['mae'].append(mae)
    eval_para['rmse'].append(rmse)
    eval_para['mape'].append(mape)
    eval_para['smape'].append(smape)
    eval_para['rae'].append(rae)

    if i == 0:
        print(f'----------{net_type}总测试数据集----------')
    else:
        print(f'------------{net_type}阶段{i}------------')
    print(f'R方：{r2}')
    print(f'平均绝对误差:{mae}')
    print(f'均方根误差:{rmse}')
    print(f'均方百分比误差:{mape:.2f}%')
    print(f'对称平均绝对百分比误差:{smape:.2f}%')
    print(f'相对绝对误差:{rae:.2f}%')
    print('----------------------------------------\n')
    i += 1

# 预测结果储存
file_name = f'预测评估指标.csv'
result = pd.DataFrame(eval_para, index=['Total_Test', 'Period1', 'Period2']).T
result.to_csv(os.path.join(save_path, file_name))
print(f'\n处理结果已成功存储在{os.path.abspath(save_path)}中')