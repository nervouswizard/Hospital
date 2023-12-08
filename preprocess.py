import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os, random
from observe import _plot

# 關閉隨機性
seed = 4222
random.seed(seed)
np.random.seed(seed)

def random_select(df0, df1):
    label_0_indices = df0.index.tolist()
    label_1_indices = df1.index.tolist()
    random_indices = random.sample(label_0_indices, len(label_1_indices))
    return pd.concat([df0[df0.index.isin(random_indices)], df1]), df0[~df0.index.isin(random_indices)]

def antizero_select(df0, df1):
    # 將 zero precentage 放入資料集
    df0_zero_percentage = ((df0 == 0).sum(axis=1)-1) / df0.shape[1]
    df0['zero'] = df0_zero_percentage
    
    # 將label移至最後面
    columns = df0.columns.tolist()
    columns.remove('label')
    columns.append('label')
    df0 = df0[columns]

    # 依zero precentage排序
    df0 = df0.sort_values(by='zero')
    # 將zero precentage分成10個區間
    num_bins = 10
    hist, bins = np.histogram(df0_zero_percentage, bins=num_bins, range=(0,1))
    df0_split = {}
    last_idx = 0
    for i in range(len(hist)):
        if hist[i] == 0: continue
        df0_split[round(bins[i], 1)] = df0[last_idx:last_idx+hist[i]]
        last_idx += hist[i]

    num_even = 2587
    df0_split[0.4] = df0_split[0.4].sample(n=num_even)
    df0_split[0.5] = df0_split[0.5].sample(n=num_even)
    df0_split[0.6] = df0_split[0.6].sample(n=num_even)
    df0_split[0.7] = df0_split[0.7].sample(n=num_even)

    sampled_data = pd.concat([df0_split[i/10] for i in range(2,10)])
    last = df0[~df0.index.isin(sampled_data.index)]
    del df0

    df1_zero_percentage = (df1 == 0).sum(axis=1) / df1.shape[1]
    df1['zero'] = df1_zero_percentage

    # 將label移至最後面
    columns = df1.columns.tolist()
    columns.remove('label')
    columns.append('label')
    df1 = df1[columns]

    return pd.concat([sampled_data, df1]), last

def label_balance(df, savepath,t = 'random'):
    os.makedirs(savepath, exist_ok=True)
    # label為0的資料集
    df0 = df[df['label'] == 0].copy()
    df1 = df[df['label'] == 1].copy()

    _plot(df0, savepath, '0')
    _plot(df1, savepath, '1')

    # random選取label為0的, 不會增加欄位
    if t == 'random':
        return random_select(df0, df1)
    # 選取採樣資料內0比較少的, 會增加一個欄位 : zero 
    elif t == 'antizero':
        return antizero_select(df0, df1)

def preprocess(filepath, savepath, filename):
    dataframe = pd.read_csv(os.path.join(filepath, filename), encoding = "ISO-8859-1")
    dataframe.fillna(value=0, inplace=True)

    # 特徵選擇
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

    # label 取出
    df_label = dataframe['label']
    dataframe.drop(columns=['label'], inplace=True)

    # 计算每个列的最小值和最大值
    min_values = dataframe.min()
    max_values = dataframe.max()

    # Min-Max 标准化到 0~1 范围
    normalized_df = (dataframe - min_values) / (max_values - min_values)
    del dataframe

    # label 放回
    normalized_df['label'] = df_label

    # 確認資料型態
    print('normalized dataframe')
    print(normalized_df.info())
    print(normalized_df['label'].value_counts())

    # 儲存
    os.makedirs(os.path.join(filepath, 'preprocessed'), exist_ok=True)
    normalized_df.to_csv(os.path.join(filepath, 'preprocessed', filename), index=None)

    # 平衡label
    balanced_df, last_df = label_balance(normalized_df, os.path.join(filepath, 'histogram'), 'random')
    del normalized_df

    #分割訓練、測試
    train_df, test_df = train_test_split(balanced_df, test_size=0.2, random_state=seed)
    test_df = pd.concat([test_df, last_df])

    # 確認資料型態
    print('training set')
    print(train_df.shape)
    print(train_df['label'].value_counts())
    print('testing set')
    print(test_df.shape)
    print(test_df['label'].value_counts())

    # 儲存
    train_df.to_csv(os.path.join(savepath, 'train.csv'), index=None)
    test_df.to_csv(os.path.join(savepath, 'test.csv'), index=None)

if __name__ == '__main__':
    filepath = os.path.join('data', '0_ori')
    savepath = os.path.join('data', '1_preprocess')
    os.makedirs(savepath, exist_ok=True)
    filename = 'Hospital.csv'
    preprocess(filepath, savepath, filename)