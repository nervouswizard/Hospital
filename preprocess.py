import pandas as pd
import numpy as np
import re
import math
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import os

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
'''
column=[1,2,3,5,6,7,8,9,82,83,84,85,86,87,88,90,91,92,95,96]
dataframe.drop(dataframe.columns[column], inplace=True, axis=1)
# 將bool改為int
dataframe[dataframe.columns[59:74]] = dataframe[dataframe.columns[59:74]].astype('int')
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

# 確認資料型態
print(dataframe.info())

save_path = os.path.join('data', '0_ori', 'preprocessed')
os.makedirs(save_path, exist_ok=True)
dataframe.to_csv(os.path.join(save_path, 'Hospital.csv'), index=False)

# split training and test data set

train_data, test_data = train_test_split(dataframe, train_size=0.8)

save_path = os.path.join('data', '1_preprocess')
os.makedirs(save_path, exist_ok=True)
train_data.to_csv(os.path.join(save_path, 'train.csv'), index=False)
test_data.to_csv(os.path.join(save_path, 'test.csv'), index=False)

exit()

#dataframe['K (mEq/L) /B'][88406]=1.31

"""for i in dataframe[34:59].columns:
    dataframe[i].astype(str).str.strip('<>')"""
#dataframe[dataframe.columns[34:59]].replace('><-', '', regex=True, inplace=True) #front
#dataframe[dataframe.columns[0:59]].replace(' ', '', regex=True, inplace=True) #front
'''error='K (mEq/L) /B'
import time
for i in range(85000,len(dataframe[error])):
    print(dataframe[error][i])
    print(i)
    print()
    time.sleep(0.05)
'''

'''
for i in dataframe.columns:#K (mEq/L) /B
        dataframe[i] = dataframe[i].astype('float')'''
print("----------output-----------")
#train_data, test_data = train_test_split(dataframe, train_size=0.8)
'''
dataframe['Finally dead']=dataframe['Finally dead'].astype('int')

true_data=dataframe[dataframe["Finally dead"]==1]

false_data=dataframe[dataframe["Finally dead"]==0]

del true_data["Finally dead"],false_data["Finally dead"],dataframe["Finally dead"]

scaler = preprocessing.MinMaxScaler(feature_range=(0.0, 255.0))
names = dataframe.columns
b = scaler.fit_transform(true_data)
true_data = pd.DataFrame(b, columns=names)

b = scaler.fit_transform(false_data)
false_data = pd.DataFrame(b, columns=names)

path=f'{test_path}/original/'
createFolder(path)

np.savetxt(f'{path}/trn_true.txt', true_data.values, fmt='%s', delimiter='\t')
np.savetxt(f'{path}/trn_false.txt', false_data.values, fmt='%s', delimiter='\t')
'''
print('----------split train test Data-----------')
train_data, test_data = train_test_split(dataframe, train_size=0.8)

Label_trn=train_data['label'].astype('int')
Label_tst=test_data['label'].astype('int')

'''print('----------SMOTE-----------')   
   
Label=train_data["Finally dead"]
del train_data["Finally dead"]
smo = SMOTE(sampling_strategy=1)
X_smo, y_smo = smo.fit_resample(train_data, Label)
from collections import Counter
print(Counter(y_smo))
X_smo["Finally dead"] = y_smo
train_data=X_smo'''


print('----------Normalization RandomForest-----------')
scaler = preprocessing.MinMaxScaler(feature_range=(0.0, 1.0))
names = dataframe.columns

b = scaler.fit_transform(train_data)
RF_trn = pd.DataFrame(b, columns=names)

b = scaler.fit_transform(test_data)
RF_tst = pd.DataFrame(b, columns=names)

print('----------split_Label-----------')
true_trn=train_data[train_data["Finally dead"]==1]
true_tst=test_data[test_data["Finally dead"]==1]
false_trn=train_data[train_data["Finally dead"]==0]
false_tst=test_data[test_data["Finally dead"]==0]

del true_trn["Finally dead"],true_tst["Finally dead"],false_trn["Finally dead"],false_tst["Finally dead"],dataframe["Finally dead"],RF_trn['Finally dead'],RF_tst['Finally dead']

print('----------Normalization Picture-----------')
scaler = preprocessing.MinMaxScaler(feature_range=(0.0, 255.0))
names = dataframe.columns
b = scaler.fit_transform(true_trn)
true_trn = pd.DataFrame(b, columns=names)

b = scaler.fit_transform(true_tst)
true_tst = pd.DataFrame(b, columns=names)

b = scaler.fit_transform(false_trn)
false_trn = pd.DataFrame(b, columns=names)

b = scaler.fit_transform(false_tst)
false_tst = pd.DataFrame(b, columns=names)



'''print('----------correlation_calculate-----------')

import scipy.stats

temp_len=len(dataframe.columns)
r=[[]for _ in range(temp_len)]
r1=[[]for _ in range(temp_len)]
r2=[[]for _ in range(temp_len)]
for i in range(0,temp_len):
    temp_x=list(map(float,dataframe.iloc[:,i]))
    
    for j in range(i+1,temp_len):
        temp_y=list(map(float,dataframe.iloc[:,j]))
        temp_pearson=scipy.stats.pearsonr(temp_x, temp_y)[0]
        temp_spearman=scipy.stats.spearmanr(temp_x, temp_y)[0]
        temp_kendall=scipy.stats.kendalltau(temp_x, temp_y)[0]
        if(np.isnan(temp_pearson)):
            temp_pearson=0
        if(np.isnan(temp_spearman)):
            temp_spearman=0
        if(np.isnan(temp_kendall)):
            temp_kendall=0
        r[i].append(temp_pearson)
        r1[i].append(temp_spearman)
        r2[i].append(temp_kendall)
        if(i!=j):
            r[j].append(temp_pearson)
            r1[j].append(temp_spearman)
            r2[j].append(temp_kendall)
        
    print("i:{0}/{1}".format(i,temp_len-1))

import csv   
# 開啟輸出的 CSV 檔案
with open('pearson.csv', 'w', newline='') as csvfile:
# 建立 CSV 檔寫入器
    writer = csv.writer(csvfile)   
    for i in range(0,temp_len):
        writer.writerow(r[i])
with open('spearman.csv', 'w', newline='') as csvfile:
# 建立 CSV 檔寫入器
    writer = csv.writer(csvfile)   
    for i in range(0,temp_len):
        writer.writerow(r1[i])
with open('kendall.csv', 'w', newline='') as csvfile:
# 建立 CSV 檔寫入器
    writer = csv.writer(csvfile)   
    for i in range(0,temp_len):
        writer.writerow(r2[i])
'''
#to numpy
'''true_df=true_df.to_numpy()
false_df=false_df.to_numpy()'''
 




print("----------output-----------")
np.savetxt('trn_true_test.txt', true_trn.values, fmt='%s', delimiter='\t')
np.savetxt('trn_false_test.txt', false_trn.values, fmt='%s', delimiter='\t')
np.savetxt('tst_true_test.txt', true_tst.values, fmt='%s', delimiter='\t')
np.savetxt('tst_false_test.txt', false_tst.values, fmt='%s', delimiter='\t')

np.savetxt('forest_trn.txt', RF_trn.values, fmt='%s', delimiter='\t')
np.savetxt('forest_tst.txt', RF_tst.values, fmt='%s', delimiter='\t')
np.savetxt('forest_trn_label.txt', Label_trn.values, fmt='%s', delimiter='\t')
np.savetxt('forest_tst_label.txt', Label_tst.values, fmt='%s', delimiter='\t')

print("----------done-----------")
import os
os.system("pause")