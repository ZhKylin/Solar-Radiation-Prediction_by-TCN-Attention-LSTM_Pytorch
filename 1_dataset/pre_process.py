# Encoding: UTF-8
# Author: Kylin Zhang
# Time: 2024/5/27 - 16:49

# step1-1：对原始数据表格进行特征提取，得到文件SolarPrediction_processed.csv


import pandas as pd
from pytz import timezone
import pytz

# 读取表格文件
dataset = pd.read_csv('SolarPrediction_original.csv')
dataset = dataset.sort_values(['UNIXTime'], ascending=True)
dataset = dataset.ffill()

# 对表格时间数据进行处理
# 将时间戳改为日期作为索引，并将时区转化为当地时区
hawaii = timezone('Pacific/Honolulu')
dataset.index = pd.to_datetime(dataset['UNIXTime'], unit='s')
dataset.index = dataset.index.tz_localize(pytz.utc).tz_convert(hawaii)
# 将日期拆分到新列，并删除原日期列
dataset['MonthOfYear'] = dataset.index.strftime('%m').astype(int)
dataset['WeekOfYear'] = dataset.index.strftime('%U').astype(int)
dataset['DayOfYear'] = dataset.index.strftime('%j').astype(int)
dataset.drop('Data', inplace=True, axis=1)
# 将时间拆分到新列，并删除原时间列
dataset['TimeOfDay(h)'] = dataset.index.hour
dataset['TimeOfDay(m)'] = dataset.index.hour*60 + dataset.index.minute
dataset['TimeOfDay(s)'] = (dataset.index.hour*60*60 + dataset.index.minute*60
                           + dataset.index.second)
dataset.drop('Time', inplace=True, axis=1)
# 将日出日落时间换为当天日照时长，并删除原日出日落时间列
dataset['TimeSunRise'] = pd.to_datetime(dataset['TimeSunRise'], format='%H:%M:%S')
dataset['TimeSunSet'] = pd.to_datetime(dataset['TimeSunSet'], format='%H:%M:%S')
dataset['DayLength(s)'] = (dataset['TimeSunSet'].dt.hour*60*60
                           + dataset['TimeSunSet'].dt.minute*60
                           + dataset['TimeSunSet'].dt.second
                           - dataset['TimeSunRise'].dt.hour*60*60
                           - dataset['TimeSunRise'].dt.minute*60
                           - dataset['TimeSunRise'].dt.second)
dataset.drop(['TimeSunRise', 'TimeSunSet'], inplace=True, axis=1)

# 存储新列表
dataset.to_csv('SolarPrediction_processed.csv', index=False)