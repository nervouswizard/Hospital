import pandas as pd
import matplotlib.pyplot as plt
import os

def _plot(dataframe, savepath, t):
    zero_percentage = (dataframe == 0).sum(axis=1) / dataframe.shape[1]
    zero_percentage = zero_percentage.sort_values()

    # 繪製圖表
    plt.figure(figsize=(10, 5))
    plt.plot(range(dataframe.shape[0]), zero_percentage * 100, color='blue')
    plt.xlabel('Row Index')
    plt.ylabel('Percentage of Zeros')
    plt.title(f'Accumulated percentage of Zeros in Each Row (Label == {t})')
    plt.savefig(f'label{t}.png')

def plot_zero_persentage(dataframe, savepath):
    df0 = dataframe[dataframe['label'] == 0]
    _plot(df0, savepath, 0)
    df1 = dataframe[dataframe['label'] == 1]
    _plot(df1, savepath, 1)


def observe(filepath, savepath, filename):
    dataframe = pd.read_csv(os.path.join(filepath, filename), encoding = "ISO-8859-1")
    dataframe.fillna(value=0, inplace=True)

    # 特徵選擇
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

    

if __name__ == '__main__':
   filepath = os.path.join('data', '0_ori')
   savepath = os.path.join('data', '0_ori', 'ovserve')
   filename = 'Hospital.csv'
   observe(filepath, savepath, filename)