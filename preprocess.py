import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import os, random

# read dataset
test_path='./Preprocess_Data'
dataframe = pd.read_csv("./data/0_ori/Hospital.csv", encoding = "ISO-8859-1")
dataframe.fillna(value=0, inplace=True)

print("----------preprocess-----------")
'''刪除欄位：
1 IPDAT 輸入日期
2 IDCODE 歸戶代號
3 OPDNO 門診號
5 RGSDPT 掛號科別
6 EMGADMDAT 急診入院日期
7 EMGDGDAT 急診出院日期
8 CSN 住院號
9 ITID 離院動態代號
82 EMGDEAD 急診死亡
83 HSPDEAD 住院死亡
84 DEADDAT 死亡時間
85 DEADSINCEEMG 自急診入院死亡時間
86 Dead within 24hr
87 Dead within 72hr
88 Dead within 168hr
90 Dead within 6hr
91 Dead within 12hr
92 Dead within 48hr
95 HSPADMDAT 住院日期
96 HSPDGDAT 住院出院日期
都是0的資料:
31 Peptic ulcer disease
40 AIDS/HIV
50 RDWSD (%) /B
74 EMGCRTICAL
'''
column=[1,2,3,5,6,7,8,9,31,40,50,74,82,83,84,85,86,87,88,90,91,92,95,96]
dataframe.drop(dataframe.columns[column], inplace=True, axis=1)
# 將bool改為int
dataframe[dataframe.columns[56:70]] = dataframe[dataframe.columns[56:70]].astype('int')
dataframe.rename(columns={'Finally dead':'label'}, inplace=True)
dataframe['label']=dataframe['label'].astype('int')

# column rename
columns = dataframe.columns.tolist()
for i, col in enumerate(columns):
    try:
        columns[i] = col.replace('/', '.')
    except:
        pass

# 將label移至最後面
dataframe.columns = columns
columns.remove('label')
columns.append('label')
dataframe = dataframe[columns]

# 平衡label
label_0_indices = dataframe[dataframe['label'] == 0].index.tolist()
label_1_indices = dataframe[dataframe['label'] == 1].index.tolist()
random.seed(42)  # 设置随机种子以确保可复现的结果
random_indices = random.sample(label_0_indices, len(label_1_indices))
balanced_df = pd.concat([dataframe.loc[random_indices], dataframe.loc[label_1_indices]])
del label_0_indices, label_1_indices, dataframe

# label 取出
df_label = balanced_df['label']
balanced_df.drop(columns=['label'], inplace=True)

# 计算每个列的最小值和最大值
min_values = balanced_df.min()
max_values = balanced_df.max()

# Min-Max 标准化到 0~1 范围
balanced_normalized_df = (balanced_df - min_values) / (max_values - min_values)

# label 放回
balanced_normalized_df['label'] = df_label

# 確認資料型態
print('balanced set')
print(balanced_normalized_df.info())
print(balanced_normalized_df['label'].value_counts())

#分割訓練、測試
train_df, test_df = train_test_split(balanced_normalized_df, test_size=0.2, random_state=42)

# 確認資料型態
print('training set')
print(train_df['label'].value_counts())
print('testing set')
print(test_df['label'].value_counts())

# 儲存
save_path = os.path.join('data', '1_preprocess')
os.makedirs(save_path, exist_ok=True)
train_df.to_csv(os.path.join(save_path, 'train.csv'), index=None)
test_df.to_csv(os.path.join(save_path, 'test.csv'), index=None)